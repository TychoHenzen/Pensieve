from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from tests.eggroll_reference import (
    ReferenceMatrixDirection,
    ReferenceVectorDirection,
    apply_reference_pair_loop_update,
    capture_eggroll_step_snapshot,
    evaluate_materialized_candidates,
    materialize_reference_candidate,
    reference_candidate_deltas,
    restore_eggroll_step_snapshot,
    sample_reference_directions,
)
from workspace.concept_slots import Workspace


def _assert_state_values_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual.cpu(), expected.cpu())
        return
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_values_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_state_values_equal(actual_item, expected_item)
        return
    assert actual == expected


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


def test_step_snapshot_restores_all_mutable_state_without_aliasing() -> None:
    parameters = [
        nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3)),
        nn.Parameter(torch.tensor([1.0, -2.0])),
    ]
    optimizer = torch.optim.Adam(parameters, lr=0.03)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad()

    workspace = Workspace(slot_count=1)
    workspace.write_slots(
        torch.arange(workspace.slots.numel(), dtype=torch.float32).reshape_as(
            workspace.slots
        )
    )
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(404)

    snapshot = capture_eggroll_step_snapshot(parameters, optimizer, workspace)
    saved_parameter = snapshot.parameter_values[0].clone()
    saved_workspace = snapshot.workspace_slots.clone()
    snapshot_adam_tensor = next(
        value
        for state in snapshot.optimizer_state["state"].values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    saved_adam_tensor = snapshot_adam_tensor.clone()

    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(50.0)
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                value.add_(70.0)
    optimizer.param_groups[0]["lr"] = 9.0
    workspace.slots.add_(80.0)
    random.random()
    np.random.random()
    torch.rand(3)
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            torch.rand(1, device=f"cuda:{device_index}")

    restore_eggroll_step_snapshot(snapshot, parameters, optimizer, workspace)

    for parameter, expected in zip(
        parameters, snapshot.parameter_values, strict=True
    ):
        assert torch.equal(parameter.detach(), expected)
        assert parameter.data_ptr() != expected.data_ptr()
    _assert_state_values_equal(optimizer.state_dict(), snapshot.optimizer_state)
    assert torch.equal(workspace.slots, snapshot.workspace_slots)
    assert workspace.slots.data_ptr() != snapshot.workspace_slots.data_ptr()
    _assert_state_values_equal(random.getstate(), snapshot.python_rng_state)
    _assert_state_values_equal(np.random.get_state(), snapshot.numpy_rng_state)
    assert torch.equal(torch.get_rng_state(), snapshot.torch_cpu_rng_state)
    if torch.cuda.is_available():
        actual_cuda_states = torch.cuda.get_rng_state_all()
        assert len(actual_cuda_states) == len(snapshot.torch_cuda_rng_states)
        for actual, expected in zip(
            actual_cuda_states, snapshot.torch_cuda_rng_states, strict=True
        ):
            assert torch.equal(actual.cpu(), expected)

    restored_adam_tensor = next(
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    assert restored_adam_tensor.data_ptr() != snapshot_adam_tensor.data_ptr()
    with torch.no_grad():
        parameters[0].add_(1.0)
        workspace.slots.add_(1.0)
        restored_adam_tensor.add_(1.0)
    assert torch.equal(snapshot.parameter_values[0], saved_parameter)
    assert torch.equal(snapshot.workspace_slots, saved_workspace)
    assert torch.equal(
        snapshot_adam_tensor,
        saved_adam_tensor,
    )
