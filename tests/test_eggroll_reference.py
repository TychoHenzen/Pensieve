from __future__ import annotations

import math

import torch
from torch import nn

from tests.eggroll_reference import (
    ReferenceMatrixDirection,
    ReferenceVectorDirection,
    apply_reference_pair_loop_update,
    evaluate_materialized_candidates,
    materialize_reference_candidate,
    reference_candidate_deltas,
    sample_reference_directions,
)


def test_reference_sampler_uses_b_first_then_a_in_registry_order() -> None:
    seed = 812
    sigma = 0.3
    rank = 2
    matrix = torch.zeros(2, 3)
    vector = torch.zeros(4)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    expected_matrix_draw = torch.randn(5, rank, generator=generator)
    expected_vector_draw = torch.randn(4, generator=generator)

    matrix_direction, vector_direction = sample_reference_directions(
        [matrix, vector],
        seed=seed,
        sigma=sigma,
        rank=rank,
    )

    assert isinstance(matrix_direction, ReferenceMatrixDirection)
    assert torch.equal(matrix_direction.b, expected_matrix_draw[:3])
    assert torch.equal(matrix_direction.a, expected_matrix_draw[3:])
    expected_delta = (
        expected_matrix_draw[3:] @ expected_matrix_draw[:3].T
    ) * (sigma / math.sqrt(rank))
    assert torch.equal(matrix_direction.materialize(1.0), expected_delta)

    assert isinstance(vector_direction, ReferenceVectorDirection)
    assert torch.equal(vector_direction.noise, expected_vector_draw)
    assert torch.equal(
        vector_direction.materialize(1.0),
        expected_vector_draw * sigma,
    )


def test_materialized_candidates_are_antithetic_and_keep_candidate_order() -> None:
    parameters = [torch.zeros(2, 3), torch.zeros(2)]
    directions = sample_reference_directions(
        parameters,
        seed=71,
        sigma=0.2,
        rank=2,
    )

    positive = materialize_reference_candidate(directions, sign=1.0)
    negative = materialize_reference_candidate(directions, sign=-1.0)

    for positive_delta, negative_delta in zip(positive, negative, strict=True):
        assert torch.equal(negative_delta, -positive_delta)

    observed = evaluate_materialized_candidates(
        parameters,
        base_seed=71,
        population_size=4,
        sigma=0.2,
        rank=2,
        evaluate=lambda deltas: [delta.clone() for delta in deltas],
    )
    assert len(observed) == 4
    for pair_start in (0, 2):
        for positive_delta, negative_delta in zip(
            observed[pair_start], observed[pair_start + 1], strict=True
        ):
            assert torch.equal(negative_delta, -positive_delta)


def test_minimizing_optimizer_moves_toward_higher_fitness_candidate() -> None:
    parameter = nn.Parameter(torch.zeros(2, 3))
    optimizer = torch.optim.SGD([parameter], lr=0.25)
    base_seed = 19
    positive_delta = reference_candidate_deltas(
        [parameter],
        seed=base_seed,
        sign=1.0,
        sigma=0.1,
        rank=2,
    )[0]

    gradients = apply_reference_pair_loop_update(
        [parameter],
        optimizer,
        base_seed=base_seed,
        fitnesses=[3.0, -1.0],
        sigma=0.1,
        rank=2,
    )

    assert torch.sum(gradients[0] * positive_delta).item() < 0.0
    assert torch.sum(parameter.detach() * positive_delta).item() > 0.0
