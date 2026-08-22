from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from tests.eggroll_reference import (
    apply_reference_pair_loop_update,
    reference_candidate_deltas,
    reference_descent_gradients,
)
from train.eggroll_updates import (
    apply_factorized_update,
    assemble_factorized_descent_gradients,
)


class UpdateShapeRecorder(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.shapes: list[tuple[int, ...]] = []

    def __torch_dispatch__(
        self,
        func: object,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        result = func(*args, **(kwargs or {}))  # type: ignore[operator]
        self._record(result)
        return result

    def _record(self, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            self.shapes.append(tuple(value.shape))
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._record(item)
        elif isinstance(value, dict):
            for item in value.values():
                self._record(item)


def _assert_nested_close(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_close(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_close(actual_item, expected_item)
        return
    assert actual == expected


def _parameter_copies() -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    originals = [
        torch.arange(35, dtype=torch.float32).reshape(7, 5) / 10.0,
        torch.linspace(-0.5, 0.5, 7),
        torch.arange(12, dtype=torch.float32).reshape(3, 4) / 8.0,
        torch.tensor(0.25),
    ]
    return (
        [nn.Parameter(value.clone()) for value in originals],
        [nn.Parameter(value.clone()) for value in originals],
    )


def test_factorized_gradients_and_adam_update_match_materialized_reference() -> None:
    reference_parameters, factorized_parameters = _parameter_copies()
    reference_optimizer = torch.optim.Adam(reference_parameters, lr=0.007)
    factorized_optimizer = torch.optim.Adam(factorized_parameters, lr=0.007)
    fitnesses = [2.5, -0.5, 0.75, 1.5, -2.0, 0.25, 3.0, -1.0]
    settings = {
        "base_seed": 771,
        "fitnesses": fitnesses,
        "sigma": 0.08,
        "rank": 3,
    }

    expected_gradients = reference_descent_gradients(
        reference_parameters,
        **settings,
    )
    actual_gradients = assemble_factorized_descent_gradients(
        factorized_parameters,
        **settings,
    )

    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    apply_reference_pair_loop_update(
        reference_parameters,
        reference_optimizer,
        **settings,
    )
    apply_factorized_update(
        factorized_parameters,
        factorized_optimizer,
        **settings,
    )

    for actual, expected in zip(
        factorized_parameters, reference_parameters, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    _assert_nested_close(
        factorized_optimizer.state_dict(),
        reference_optimizer.state_dict(),
    )


def test_matrix_assembly_creates_only_one_final_dense_gradient() -> None:
    population_size = 8
    pair_count = population_size // 2
    out_features = 7
    in_features = 5
    parameters = [
        nn.Parameter(torch.zeros(out_features, in_features)),
        nn.Parameter(torch.zeros(6)),
    ]
    recorder = UpdateShapeRecorder()

    with recorder:
        gradients = assemble_factorized_descent_gradients(
            parameters,
            base_seed=90,
            fitnesses=[3.0, -1.0, 2.0, 0.0, -0.5, 1.0, 4.0, -2.0],
            sigma=0.1,
            rank=2,
        )

    assert gradients[0].shape == (out_features, in_features)
    assert recorder.shapes.count((out_features, in_features)) == 1
    assert (pair_count, out_features, in_features) not in recorder.shapes
    assert (population_size, out_features, in_features) not in recorder.shapes


def test_factorized_update_ascends_toward_higher_fitness_candidate() -> None:
    parameter = nn.Parameter(torch.zeros(3, 5))
    optimizer = torch.optim.SGD([parameter], lr=0.2)
    base_seed = 29
    positive_delta = reference_candidate_deltas(
        [parameter],
        seed=base_seed,
        sign=1.0,
        sigma=0.12,
        rank=2,
    )[0]

    gradients = apply_factorized_update(
        [parameter],
        optimizer,
        base_seed=base_seed,
        fitnesses=[4.0, -2.0],
        sigma=0.12,
        rank=2,
    )

    assert torch.sum(gradients[0] * positive_delta).item() < 0.0
    assert torch.sum(parameter.detach() * positive_delta).item() > 0.0
