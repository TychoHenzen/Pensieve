"""VICReg-style collapse guards for the latent core's workspace slots.

Two regularization terms discourage the collapse failure modes the latent
loop is prone to:

- `variance_loss` penalizes slot dimensions whose spread across the batch of
  slots falls below a target standard deviation, which keeps slots from all
  converging to the same vector.
- `covariance_loss` penalizes correlation between distinct feature
  dimensions, which keeps those dimensions from becoming redundant encodings
  of the same information.

`EMAProjection` provides a slow-moving target copy of a learned projection
layer, used to compute a stop-gradient target for self-supervised losses
without the target trivially tracking the online network step for step.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

DEFAULT_VARIANCE_WEIGHT = 1.0
DEFAULT_COVARIANCE_WEIGHT = 1.0
DEFAULT_VARIANCE_THRESHOLD = 1.0
DEFAULT_EMA_DECAY = 0.99


def post_loop_slot_variance(slots: torch.Tensor) -> torch.Tensor:
    """Return the mean population variance across slot vectors.

    `slots` may be `(slots, dimensions)` or
    `(batch, slots, dimensions)`. The slot axis is always the second-to-last
    axis, and all remaining axes are averaged into one scalar.
    """
    return slots.var(dim=-2, unbiased=False).mean()


def slot_variance_penalty(
    slots: torch.Tensor,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> torch.Tensor:
    """Return one bounded collapse penalty per example or candidate.

    The slot axis is second-to-last. A two-dimensional slot tensor returns
    one scalar. Batched candidate tensors retain their leading dimensions.
    """
    if slots.ndim < 2:
        raise ValueError("slots must have at least a slot and feature dimension")
    per_dimension_variance = slots.var(dim=-2, unbiased=False)
    per_dimension_std = torch.sqrt(per_dimension_variance + 1e-4)
    return torch.clamp(variance_threshold - per_dimension_std, min=0.0).mean(
        dim=-1
    )


class VICRegLoss(nn.Module):
    """Variance + covariance collapse-prevention regularizer over slots."""

    def __init__(
        self,
        variance_weight: float = DEFAULT_VARIANCE_WEIGHT,
        covariance_weight: float = DEFAULT_COVARIANCE_WEIGHT,
        variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    ) -> None:
        super().__init__()
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.variance_threshold = variance_threshold

    def variance_loss(self, slots: torch.Tensor) -> torch.Tensor:
        """Hinge penalty on per-dimension standard deviation across slots.

        `slots` is (N, D): N slot vectors of dimension D. Standard deviation
        is computed across the N slots for each of the D dimensions; any
        dimension whose std falls below `variance_threshold` is penalized.
        """
        std = torch.sqrt(slots.var(dim=0, unbiased=False) + 1e-4)
        hinge = torch.clamp(self.variance_threshold - std, min=0.0)
        return hinge.mean()

    def covariance_loss(self, slots: torch.Tensor) -> torch.Tensor:
        """Penalty on off-diagonal entries of the feature covariance matrix.

        `slots` is (N, D). The covariance matrix is (D, D), computed over the
        N slots as observations. Off-diagonal entries measure redundancy
        between distinct feature dimensions.
        """
        num_slots, num_dims = slots.shape
        centered = slots - slots.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / max(num_slots - 1, 1)
        off_diag_mask = 1.0 - torch.eye(num_dims, device=slots.device)
        return (cov.pow(2) * off_diag_mask).sum() / num_dims

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """Weighted sum of the variance and covariance collapse penalties."""
        return (
            self.variance_weight * self.variance_loss(slots)
            + self.covariance_weight * self.covariance_loss(slots)
        )


class EMAProjection(nn.Module):
    """Exponential-moving-average target copy of a learned projection.

    Holds a frozen copy of `projection`'s weights, updated by `update()` as
    an EMA of the source projection's current weights. `forward` runs the
    target copy under `no_grad` and detaches its output, so it can be used
    as a stop-gradient target in a self-supervised loss.
    """

    def __init__(self, projection: nn.Linear, decay: float = DEFAULT_EMA_DECAY) -> None:
        super().__init__()
        self.decay = decay
        self.source = projection
        self.target = copy.deepcopy(projection)
        for param in self.target.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self) -> None:
        """Update the target weights toward the source weights by `decay`."""
        for target_param, source_param in zip(
            self.target.parameters(), self.source.parameters()
        ):
            target_param.mul_(self.decay).add_(source_param, alpha=1.0 - self.decay)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project `hidden_states` with the EMA target, detached from the graph."""
        with torch.no_grad():
            projected = self.target(hidden_states)
        return projected.detach()
