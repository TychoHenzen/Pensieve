"""Materialized EGGROLL reference helpers for tests and benchmarks only."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import numpy as np
import torch


class SnapshotWorkspace(Protocol):
    def snapshot(self) -> torch.Tensor: ...

    def restore(self, state: torch.Tensor) -> None: ...


@dataclass(frozen=True)
class EggrollStepSnapshot:
    """Complete mutable state needed to replay one EGGROLL step."""

    parameter_values: tuple[torch.Tensor, ...]
    optimizer_state: dict[str, Any]
    workspace_slots: torch.Tensor
    python_rng_state: object
    numpy_rng_state: tuple[str, np.ndarray, int, int, float]
    torch_cpu_rng_state: torch.Tensor
    torch_cuda_rng_states: tuple[torch.Tensor, ...]


def _clone_state_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {
            copy.deepcopy(key): _clone_state_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clone_state_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_value(item) for item in value)
    return copy.deepcopy(value)


def _clone_numpy_rng_state(
    state: tuple[str, np.ndarray, int, int, float],
) -> tuple[str, np.ndarray, int, int, float]:
    return (state[0], state[1].copy(), state[2], state[3], state[4])


def capture_eggroll_step_snapshot(
    parameters: Sequence[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    workspace: SnapshotWorkspace,
) -> EggrollStepSnapshot:
    """Clone all state that can affect or be changed by one EGGROLL step."""
    numpy_rng_state = _clone_numpy_rng_state(np.random.get_state())
    cuda_rng_states = (
        tuple(state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    return EggrollStepSnapshot(
        parameter_values=tuple(
            parameter.detach().clone() for parameter in parameters
        ),
        optimizer_state=_clone_state_value(optimizer.state_dict()),
        workspace_slots=workspace.snapshot().detach().clone(),
        python_rng_state=copy.deepcopy(random.getstate()),
        numpy_rng_state=numpy_rng_state,
        torch_cpu_rng_state=torch.get_rng_state().detach().cpu().clone(),
        torch_cuda_rng_states=cuda_rng_states,
    )


def restore_eggroll_step_snapshot(
    snapshot: EggrollStepSnapshot,
    parameters: Sequence[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    workspace: SnapshotWorkspace,
) -> None:
    """Restore a snapshot without sharing its mutable tensor storage."""
    if len(parameters) != len(snapshot.parameter_values):
        raise ValueError("parameter registry length differs from snapshot")

    with torch.no_grad():
        for parameter, saved_value in zip(
            parameters, snapshot.parameter_values, strict=True
        ):
            if parameter.shape != saved_value.shape:
                raise ValueError("parameter shape differs from snapshot")
            parameter.copy_(saved_value.to(device=parameter.device))

    optimizer.load_state_dict(_clone_state_value(snapshot.optimizer_state))
    workspace.restore(snapshot.workspace_slots.clone())
    random.setstate(copy.deepcopy(snapshot.python_rng_state))
    np.random.set_state(_clone_numpy_rng_state(snapshot.numpy_rng_state))
    torch.set_rng_state(snapshot.torch_cpu_rng_state.clone())
    if snapshot.torch_cuda_rng_states:
        torch.cuda.set_rng_state_all(
            [state.clone() for state in snapshot.torch_cuda_rng_states]
        )


@dataclass(frozen=True)
class ReferenceMatrixDirection:
    """One official EGGROLL matrix direction before its antithetic sign."""

    a: torch.Tensor
    b: torch.Tensor
    scale: float

    def materialize(self, sign: float) -> torch.Tensor:
        return (self.a @ self.b.T) * (sign * self.scale)


@dataclass(frozen=True)
class ReferenceVectorDirection:
    """One dense non-matrix direction before its antithetic sign."""

    noise: torch.Tensor
    scale: float

    def materialize(self, sign: float) -> torch.Tensor:
        return self.noise * (sign * self.scale)


ReferenceDirection = ReferenceMatrixDirection | ReferenceVectorDirection
CandidateResult = TypeVar("CandidateResult")


def sample_reference_directions(
    parameters: Sequence[torch.Tensor],
    *,
    seed: int,
    sigma: float,
    rank: int,
) -> list[ReferenceDirection]:
    """Draw canonical directions from one CPU generator in registry order."""
    if rank < 1:
        raise ValueError("rank must be at least 1")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    directions: list[ReferenceDirection] = []

    for parameter in parameters:
        if parameter.ndim == 2:
            out_features, in_features = parameter.shape
            draw = torch.randn(
                in_features + out_features,
                rank,
                generator=generator,
                device="cpu",
            ).to(parameter.device)
            b = draw[:in_features]
            a = draw[in_features:]
            directions.append(
                ReferenceMatrixDirection(
                    a=a,
                    b=b,
                    scale=sigma / math.sqrt(rank),
                )
            )
        else:
            noise = torch.randn(
                parameter.shape,
                generator=generator,
                device="cpu",
            ).to(parameter.device)
            directions.append(ReferenceVectorDirection(noise=noise, scale=sigma))

    return directions


def materialize_reference_candidate(
    directions: Sequence[ReferenceDirection],
    *,
    sign: float,
) -> list[torch.Tensor]:
    """Materialize one signed candidate from previously sampled directions."""
    return [direction.materialize(sign) for direction in directions]


def reference_candidate_deltas(
    parameters: Sequence[torch.Tensor],
    *,
    seed: int,
    sign: float,
    sigma: float,
    rank: int,
) -> list[torch.Tensor]:
    """Sample and materialize one signed reference candidate."""
    directions = sample_reference_directions(
        parameters,
        seed=seed,
        sigma=sigma,
        rank=rank,
    )
    return materialize_reference_candidate(directions, sign=sign)


def evaluate_materialized_candidates(
    parameters: Sequence[torch.Tensor],
    *,
    base_seed: int,
    population_size: int,
    sigma: float,
    rank: int,
    evaluate: Callable[[list[torch.Tensor]], CandidateResult],
) -> list[CandidateResult]:
    """Evaluate candidates in positive-then-negative antithetic pair order."""
    if population_size < 2 or population_size % 2 != 0:
        raise ValueError("population_size must be a positive even number")

    results: list[CandidateResult] = []
    for pair_index in range(population_size // 2):
        directions = sample_reference_directions(
            parameters,
            seed=base_seed + pair_index,
            sigma=sigma,
            rank=rank,
        )
        results.append(
            evaluate(materialize_reference_candidate(directions, sign=1.0))
        )
        results.append(
            evaluate(materialize_reference_candidate(directions, sign=-1.0))
        )
    return results


def reference_descent_gradients(
    parameters: Sequence[torch.Tensor],
    *,
    base_seed: int,
    fitnesses: Sequence[float] | torch.Tensor,
    sigma: float,
    rank: int,
) -> list[torch.Tensor]:
    """Assemble the pair-loop gradient supplied to a minimizing optimizer."""
    if isinstance(fitnesses, torch.Tensor):
        fitness_tensor = fitnesses
    else:
        fitness_tensor = torch.tensor(fitnesses, dtype=torch.float32)
    if fitness_tensor.ndim != 1:
        raise ValueError("fitnesses must be one-dimensional")

    population_size = fitness_tensor.numel()
    if population_size < 2 or population_size % 2 != 0:
        raise ValueError("fitnesses must contain a positive even population")

    normalized = (fitness_tensor - fitness_tensor.mean()) / (
        fitness_tensor.std() + 1e-5
    )
    gradients = [torch.zeros_like(parameter) for parameter in parameters]

    for pair_index in range(population_size // 2):
        positive_deltas = reference_candidate_deltas(
            parameters,
            seed=base_seed + pair_index,
            sign=1.0,
            sigma=sigma,
            rank=rank,
        )
        score = (
            normalized[2 * pair_index + 1] - normalized[2 * pair_index]
        ).item()
        for gradient, delta in zip(gradients, positive_deltas, strict=True):
            gradient.add_(delta, alpha=score)

    scale = math.sqrt(population_size) / population_size
    for gradient in gradients:
        gradient.mul_(scale)
    return gradients


def apply_reference_pair_loop_update(
    parameters: Sequence[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    base_seed: int,
    fitnesses: Sequence[float] | torch.Tensor,
    sigma: float,
    rank: int,
) -> list[torch.Tensor]:
    """Apply one materialized reference update and return its descent gradients."""
    optimizer.zero_grad()
    gradients = reference_descent_gradients(
        parameters,
        base_seed=base_seed,
        fitnesses=fitnesses,
        sigma=sigma,
        rank=rank,
    )
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient
    optimizer.step()
    return gradients
