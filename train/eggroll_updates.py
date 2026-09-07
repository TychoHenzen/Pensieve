"""Factorized pseudo-gradient assembly for EGGROLL updates."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from train.eggroll_perturbations import (
    DenseVectorNoise,
    MatrixFactors,
    PerturbationDirection,
    sample_perturbation_directions,
)


def _pair_descent_scores(
    fitnesses: Sequence[float] | torch.Tensor,
) -> tuple[torch.Tensor, int]:
    fitness_tensor = fitnesses if isinstance(fitnesses, torch.Tensor) else torch.tensor(fitnesses, dtype=torch.float32)
    if fitness_tensor.ndim != 1:
        raise ValueError("fitnesses must be one-dimensional")

    population_size = fitness_tensor.numel()
    if population_size < 2 or population_size % 2 != 0:
        raise ValueError("fitnesses must contain a positive even population")
    if not torch.isfinite(fitness_tensor).all():
        raise ValueError("fitnesses must be finite")

    normalized = (fitness_tensor - fitness_tensor.mean()) / torch.sqrt(fitness_tensor.var(unbiased=False) + 1e-5)
    return normalized[1::2] - normalized[0::2], population_size


def _matrix_gradient(
    directions: Sequence[MatrixFactors],
    pair_scores: torch.Tensor,
    population_scale: float,
) -> torch.Tensor:
    stacked_a = torch.stack([direction.a for direction in directions])
    stacked_b = torch.stack([direction.b for direction in directions])
    coefficients = pair_scores.to(
        device=stacked_a.device,
        dtype=stacked_a.dtype,
    )
    coefficients = coefficients * torch.tensor(
        [direction.scale for direction in directions],
        device=stacked_a.device,
        dtype=stacked_a.dtype,
    )
    coefficients = coefficients * population_scale

    weighted_a = stacked_a * coefficients[:, None, None]
    out_features = stacked_a.shape[1]
    in_features = stacked_b.shape[1]
    left = weighted_a.permute(1, 0, 2).reshape(out_features, -1)
    right = stacked_b.permute(0, 2, 1).reshape(-1, in_features)
    return left @ right


def _vector_gradient(
    directions: Sequence[DenseVectorNoise],
    pair_scores: torch.Tensor,
    population_scale: float,
) -> torch.Tensor:
    stacked_noise = torch.stack([direction.noise for direction in directions])
    coefficients = pair_scores.to(
        device=stacked_noise.device,
        dtype=stacked_noise.dtype,
    )
    coefficients = coefficients * torch.tensor(
        [direction.scale for direction in directions],
        device=stacked_noise.device,
        dtype=stacked_noise.dtype,
    )
    coefficients = coefficients * population_scale
    coefficient_shape = (len(directions),) + (1,) * (stacked_noise.ndim - 1)
    return (stacked_noise * coefficients.reshape(coefficient_shape)).sum(dim=0)


def assemble_factorized_descent_gradients(
    parameters: Sequence[torch.Tensor],
    *,
    base_seed: int,
    fitnesses: Sequence[float] | torch.Tensor,
    sigma: float,
    rank: int,
) -> list[torch.Tensor]:
    """Build registry-ordered gradients for a minimizing optimizer."""
    pair_scores, population_size = _pair_descent_scores(fitnesses)
    pair_directions = [
        sample_perturbation_directions(
            parameters,
            seed=base_seed + pair_index,
            sigma=sigma,
            rank=rank,
        )
        for pair_index in range(population_size // 2)
    ]
    population_scale = math.sqrt(population_size) / population_size
    gradients: list[torch.Tensor] = []

    for parameter_index, parameter in enumerate(parameters):
        directions: list[PerturbationDirection] = [pair[parameter_index] for pair in pair_directions]
        if parameter.ndim == 2:
            matrix_directions = []
            for direction in directions:
                if not isinstance(direction, MatrixFactors):
                    raise TypeError("matrix parameter received non-matrix direction")
                matrix_directions.append(direction)
            gradients.append(_matrix_gradient(matrix_directions, pair_scores, population_scale))
        else:
            vector_directions = []
            for direction in directions:
                if not isinstance(direction, DenseVectorNoise):
                    raise TypeError("non-matrix parameter received matrix direction")
                vector_directions.append(direction)
            gradients.append(_vector_gradient(vector_directions, pair_scores, population_scale))

    return gradients


def apply_factorized_update(
    parameters: Sequence[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    base_seed: int,
    fitnesses: Sequence[float] | torch.Tensor,
    sigma: float,
    rank: int,
) -> list[torch.Tensor]:
    """Assign factorized gradients and apply one minimizing optimizer step."""
    gradients = assemble_factorized_descent_gradients(
        parameters,
        base_seed=base_seed,
        fitnesses=fitnesses,
        sigma=sigma,
        rank=rank,
    )
    optimizer.zero_grad()
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient
    optimizer.step()
    return gradients
