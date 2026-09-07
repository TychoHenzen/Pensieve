from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from train.eggroll_factorized import factorized_linear
from train.eggroll_perturbations import (
    DenseVectorNoise,
    MatrixFactors,
    sample_perturbation_directions,
)


class AllocationShapeRecorder(TorchDispatchMode):
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


def _candidate_directions(
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    candidate_count: int,
) -> tuple[list[MatrixFactors], list[DenseVectorNoise]]:
    factors: list[MatrixFactors] = []
    bias_noises: list[DenseVectorNoise] = []
    for candidate_index in range(candidate_count):
        weight_direction, bias_direction = sample_perturbation_directions(
            [weight, bias],
            seed=100 + candidate_index,
            sigma=0.13,
            rank=2,
        )
        assert isinstance(weight_direction, MatrixFactors)
        assert isinstance(bias_direction, DenseVectorNoise)
        factors.append(weight_direction)
        bias_noises.append(bias_direction)
    return factors, bias_noises


def _materialized_outputs(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    factors: list[MatrixFactors],
    bias_noises: list[DenseVectorNoise],
    signs: list[float],
) -> torch.Tensor:
    outputs = []
    candidate_inputs = inputs.ndim > 2
    for candidate_index, (factor, noise, sign) in enumerate(
        zip(factors, bias_noises, signs, strict=True)
    ):
        current_inputs = inputs[candidate_index] if candidate_inputs else inputs
        outputs.append(
            F.linear(
                current_inputs,
                weight + factor.materialize(sign),
                bias + noise.materialize(sign),
            )
        )
    return torch.stack(outputs)


def test_shared_input_matches_materialized_encoder_candidates_without_dense_batch() -> None:
    torch.manual_seed(4)
    candidate_count = 4
    out_features = 7
    in_features = 5
    inputs = torch.randn(6, in_features)
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)
    signs = [1.0, -1.0, 1.0, -1.0]
    factors, bias_noises = _candidate_directions(
        weight,
        bias,
        candidate_count=candidate_count,
    )
    expected = _materialized_outputs(
        inputs,
        weight,
        bias,
        factors,
        bias_noises,
        signs,
    )

    recorder = AllocationShapeRecorder()
    with recorder:
        actual = factorized_linear(
            inputs,
            weight,
            factors,
            signs,
            bias=bias,
            bias_noises=bias_noises,
        )

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert (candidate_count, out_features, in_features) not in recorder.shapes


def test_candidate_inputs_match_materialized_latent_candidates() -> None:
    torch.manual_seed(5)
    candidate_count = 4
    out_features = 7
    in_features = 5
    inputs = torch.randn(candidate_count, 3, in_features)
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)
    signs = [1.0, -1.0, -1.0, 1.0]
    factors, bias_noises = _candidate_directions(
        weight,
        bias,
        candidate_count=candidate_count,
    )
    expected = _materialized_outputs(
        inputs,
        weight,
        bias,
        factors,
        bias_noises,
        signs,
    )

    recorder = AllocationShapeRecorder()
    with recorder:
        actual = factorized_linear(
            inputs,
            weight,
            factors,
            signs,
            bias=bias,
            bias_noises=bias_noises,
        )

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert (candidate_count, out_features, in_features) not in recorder.shapes


@pytest.mark.parametrize(
    ("inputs", "factor_count", "signs", "message"),
    [
        (torch.zeros(2, 3, 5), 3, [1.0, -1.0, 1.0], "candidate input count"),
        (torch.zeros(2, 5), 2, [1.0], "sign count"),
    ],
)
def test_factorized_linear_validates_candidate_counts(
    inputs: torch.Tensor,
    factor_count: int,
    signs: list[float],
    message: str,
) -> None:
    weight = torch.zeros(7, 5)
    factors = [
        MatrixFactors(a=torch.zeros(7, 2), b=torch.zeros(5, 2), scale=0.1)
        for _ in range(factor_count)
    ]

    with pytest.raises(ValueError, match=message):
        factorized_linear(inputs, weight, factors, signs)


def test_factorized_linear_rejects_incompatible_factor_shapes() -> None:
    with pytest.raises(ValueError, match="matrix A shape"):
        factorized_linear(
            torch.zeros(2, 5),
            torch.zeros(7, 5),
            [MatrixFactors(a=torch.zeros(6, 2), b=torch.zeros(5, 2), scale=0.1)],
            [1.0],
        )
