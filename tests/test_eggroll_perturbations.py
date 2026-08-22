from __future__ import annotations

import math

import pytest
import torch

from tests.eggroll_reference import (
    ReferenceMatrixDirection,
    ReferenceVectorDirection,
    sample_reference_directions,
)
from train.eggroll_perturbations import (
    DenseVectorNoise,
    MatrixFactors,
    sample_antithetic_pair,
    sample_perturbation_directions,
)


def test_sampler_matches_official_reference_draw_order_and_scale() -> None:
    seed = 442
    sigma = 0.18
    rank = 3
    parameters = [
        torch.zeros(2, 4),
        torch.zeros(3),
        torch.zeros(3, 2),
        torch.zeros(()),
    ]

    actual = sample_perturbation_directions(
        parameters,
        seed=seed,
        sigma=sigma,
        rank=rank,
    )
    reference = sample_reference_directions(
        parameters,
        seed=seed,
        sigma=sigma,
        rank=rank,
    )

    assert len(actual) == len(reference)
    for actual_direction, reference_direction in zip(
        actual, reference, strict=True
    ):
        if isinstance(reference_direction, ReferenceMatrixDirection):
            assert isinstance(actual_direction, MatrixFactors)
            assert actual_direction.b.shape[0] == reference_direction.b.shape[0]
            assert torch.equal(actual_direction.b, reference_direction.b)
            assert torch.equal(actual_direction.a, reference_direction.a)
            assert actual_direction.scale == sigma / math.sqrt(rank)
        else:
            assert isinstance(reference_direction, ReferenceVectorDirection)
            assert isinstance(actual_direction, DenseVectorNoise)
            assert torch.equal(actual_direction.noise, reference_direction.noise)
            assert actual_direction.scale == sigma


def test_matrix_factors_reconstruct_the_materialized_reference_delta() -> None:
    sigma = 0.27
    rank = 2
    parameter = torch.zeros(3, 5)
    direction = sample_perturbation_directions(
        [parameter],
        seed=901,
        sigma=sigma,
        rank=rank,
    )[0]
    reference = sample_reference_directions(
        [parameter],
        seed=901,
        sigma=sigma,
        rank=rank,
    )[0]

    assert isinstance(direction, MatrixFactors)
    assert isinstance(reference, ReferenceMatrixDirection)
    assert direction.b.shape == (5, rank)
    assert direction.a.shape == (3, rank)
    assert torch.equal(
        direction.materialize(1.0),
        reference.materialize(1.0),
    )


def test_antithetic_candidates_share_draws_with_opposite_signs() -> None:
    parameters = [torch.zeros(2, 3), torch.zeros(4)]

    positive, negative = sample_antithetic_pair(
        parameters,
        seed=31,
        sigma=0.12,
        rank=2,
    )

    assert positive.directions is negative.directions
    assert positive.sign == 1.0
    assert negative.sign == -1.0
    for positive_delta, negative_delta in zip(
        positive.materialize(), negative.materialize(), strict=True
    ):
        assert torch.equal(negative_delta, -positive_delta)


def test_vector_draw_matches_reference_after_preceding_matrix_draws() -> None:
    parameters = [torch.zeros(4, 2), torch.zeros(3, 5), torch.zeros(7)]
    actual = sample_perturbation_directions(
        parameters,
        seed=78,
        sigma=0.05,
        rank=4,
    )
    reference = sample_reference_directions(
        parameters,
        seed=78,
        sigma=0.05,
        rank=4,
    )

    assert isinstance(actual[-1], DenseVectorNoise)
    assert isinstance(reference[-1], ReferenceVectorDirection)
    assert torch.equal(actual[-1].noise, reference[-1].noise)
    assert torch.equal(
        actual[-1].materialize(-1.0),
        reference[-1].materialize(-1.0),
    )


@pytest.mark.parametrize("rank", [0, -1, True, 1.5])
def test_sampler_rejects_invalid_rank(rank: object) -> None:
    with pytest.raises(ValueError, match="rank must be a positive integer"):
        sample_perturbation_directions(
            [torch.zeros(2, 3)],
            seed=1,
            sigma=0.1,
            rank=rank,  # type: ignore[arg-type]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sampler_moves_cpu_draws_to_the_parameter_device() -> None:
    parameters = [
        torch.zeros(2, 3, device="cuda"),
        torch.zeros(4, device="cuda"),
    ]

    directions = sample_perturbation_directions(
        parameters,
        seed=9,
        sigma=0.1,
        rank=2,
    )

    assert isinstance(directions[0], MatrixFactors)
    assert directions[0].a.device.type == "cuda"
    assert directions[0].b.device.type == "cuda"
    assert isinstance(directions[1], DenseVectorNoise)
    assert directions[1].noise.device.type == "cuda"
