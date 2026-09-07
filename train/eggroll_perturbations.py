"""Seeded factorized perturbations for EGGROLL candidate evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MatrixFactors:
    """An unsigned low-rank direction for a matrix shaped ``(out, in)``."""

    a: torch.Tensor
    b: torch.Tensor
    scale: float

    def materialize(self, sign: float) -> torch.Tensor:
        """Materialize this one direction for reference comparisons."""
        return (self.a @ self.b.T) * (sign * self.scale)


@dataclass(frozen=True)
class DenseVectorNoise:
    """An unsigned dense direction for a non-matrix parameter."""

    noise: torch.Tensor
    scale: float

    def materialize(self, sign: float) -> torch.Tensor:
        """Materialize this one direction for reference comparisons."""
        return self.noise * (sign * self.scale)


PerturbationDirection = MatrixFactors | DenseVectorNoise


@dataclass(frozen=True)
class SignedPerturbationSet:
    """A signed view over one registry-ordered set of sampled directions."""

    directions: tuple[PerturbationDirection, ...]
    sign: float

    def materialize(self) -> list[torch.Tensor]:
        """Materialize each parameter delta without forming a candidate batch."""
        return [direction.materialize(self.sign) for direction in self.directions]


def sample_perturbation_directions(
    parameters: Sequence[torch.Tensor],
    *,
    seed: int,
    sigma: float,
    rank: int,
) -> tuple[PerturbationDirection, ...]:
    """Draw one official EGGROLL direction in parameter registry order."""
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError("rank must be a positive integer")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    directions: list[PerturbationDirection] = []

    for parameter in parameters:
        if parameter.ndim == 2:
            out_features, in_features = parameter.shape
            draw = torch.randn(
                in_features + out_features,
                rank,
                generator=generator,
                device="cpu",
            ).to(parameter.device)
            directions.append(
                MatrixFactors(
                    b=draw[:in_features],
                    a=draw[in_features:],
                    scale=sigma / math.sqrt(rank),
                )
            )
        else:
            noise = torch.randn(
                parameter.shape,
                generator=generator,
                device="cpu",
            ).to(parameter.device)
            directions.append(DenseVectorNoise(noise=noise, scale=sigma))

    return tuple(directions)


def antithetic_candidates(
    directions: tuple[PerturbationDirection, ...],
) -> tuple[SignedPerturbationSet, SignedPerturbationSet]:
    """Return positive and negative views that share one sampled direction."""
    return (
        SignedPerturbationSet(directions=directions, sign=1.0),
        SignedPerturbationSet(directions=directions, sign=-1.0),
    )


def sample_antithetic_pair(
    parameters: Sequence[torch.Tensor],
    *,
    seed: int,
    sigma: float,
    rank: int,
) -> tuple[SignedPerturbationSet, SignedPerturbationSet]:
    """Sample one direction and expose its two antithetic candidates."""
    directions = sample_perturbation_directions(
        parameters,
        seed=seed,
        sigma=sigma,
        rank=rank,
    )
    return antithetic_candidates(directions)
