from __future__ import annotations

import copy
import math
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from tests.eggroll_reference import (
    apply_reference_pair_loop_update,
    reference_candidate_deltas,
    reference_descent_gradients,
)
from train.eggroll_updates import (
    _pair_descent_scores,
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


# covers: train/eggroll-execution :: Pseudo-gradient assembly aggregates low-rank factors :: Factorized matrix update matches reference update
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

    for actual, expected in zip(factorized_parameters, reference_parameters, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    _assert_nested_close(
        factorized_optimizer.state_dict(),
        reference_optimizer.state_dict(),
    )


# covers: train/eggroll-execution :: Pseudo-gradient assembly aggregates low-rank factors :: Assembly has one dense matrix result
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


# covers: train/eggroll-execution :: Pseudo-gradient assembly aggregates low-rank factors :: Update follows fitness ascent
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


def test_pair_scores_use_population_variance_normalization() -> None:
    fitnesses = torch.tensor([1.0, 2.0, 5.0, 9.0])

    actual, population_size = _pair_descent_scores(fitnesses)

    denominator = math.sqrt(9.6875 + 1e-5)
    expected_normalized = torch.tensor(
        [-3.25 / denominator, -2.25 / denominator, 0.75 / denominator, 4.75 / denominator]
    )
    expected_scores = torch.tensor(
        [expected_normalized[1] - expected_normalized[0], expected_normalized[3] - expected_normalized[2]]
    )
    assert population_size == 4
    torch.testing.assert_close(actual, expected_scores, rtol=1e-6, atol=1e-6)


# covers: train/eggroll-execution :: Pseudo-gradient assembly aggregates low-rank factors :: Non-finite candidate fitness
def test_non_finite_fitness_rejects_before_parameter_or_optimizer_mutation() -> None:
    parameter = nn.Parameter(torch.tensor([[0.25, -0.5], [1.0, 0.75]]))
    optimizer = torch.optim.Adam([parameter], lr=0.03)
    apply_factorized_update(
        [parameter],
        optimizer,
        base_seed=3,
        fitnesses=[2.0, -1.0],
        sigma=0.1,
        rank=1,
    )
    for non_finite in (float("nan"), float("inf"), float("-inf")):
        parameter_before = parameter.detach().clone()
        optimizer_before = copy.deepcopy(optimizer.state_dict())

        with pytest.raises(ValueError) as error:
            apply_factorized_update(
                [parameter],
                optimizer,
                base_seed=4,
                fitnesses=[2.0, non_finite],
                sigma=0.1,
                rank=1,
            )

        assert str(error.value) == "fitnesses must be finite"
        assert torch.equal(parameter.detach(), parameter_before)
        assert optimizer.state_dict()["param_groups"] == optimizer_before["param_groups"]
        _assert_nested_close(optimizer.state_dict(), optimizer_before)
