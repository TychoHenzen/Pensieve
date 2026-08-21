from __future__ import annotations

import torch

from eval.instrumentation import InstrumentationFields

SLOT_DIM = 896
ABLATION_SLOT_COUNTS = (1, 4, 8, 16, 32, 64)
DEFAULT_SLOT_COUNT = 16


class Workspace:
    """Holds a configurable number of concept slots, each of dimension 896."""

    def __init__(self, slot_count: int = DEFAULT_SLOT_COUNT) -> None:
        if slot_count not in ABLATION_SLOT_COUNTS:
            raise ValueError(
                f"slot_count must be one of {ABLATION_SLOT_COUNTS}, got {slot_count}"
            )
        self.slot_count = slot_count
        self.slots: torch.Tensor = torch.zeros(slot_count, SLOT_DIM)

    def read_slots(self) -> torch.Tensor:
        return self.slots

    def write_slots(self, values: torch.Tensor) -> None:
        if values.shape != self.slots.shape:
            raise ValueError(
                f"expected shape {tuple(self.slots.shape)}, got {tuple(values.shape)}"
            )
        self.slots = values

    def snapshot(self) -> torch.Tensor:
        return self.slots.clone()

    def restore(self, state: torch.Tensor) -> None:
        if state.shape != self.slots.shape:
            raise ValueError(
                f"expected shape {tuple(self.slots.shape)}, got {tuple(state.shape)}"
            )
        self.slots = state.clone()

    def collapse_stats(self) -> dict[str, float]:
        """Per-dimension variance and inter-slot covariance, for collapse detection.

        Variance is the mean, across the slot feature dimensions, of the
        variance of that dimension's values across slots. It is near zero
        when all slots hold (nearly) the same vector, and grows as slots
        diverge from one another.

        Covariance is the mean absolute off-diagonal entry of the slot x slot
        covariance matrix (slots treated as observations over the feature
        dimensions), summarizing how correlated distinct slots are with one
        another.
        """
        if self.slot_count < 2:
            return {"variance": 0.0, "covariance": 0.0}
        variance = self.slots.var(dim=0).mean().item()
        centered = self.slots - self.slots.mean(dim=0, keepdim=True)
        cov = (centered @ centered.T) / max(self.slots.shape[-1] - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        covariance = off_diag.abs().mean().item()
        return {"variance": variance, "covariance": covariance}

    def instrumentation_fields(self) -> InstrumentationFields:
        """Collapse-detection stats packaged as instrumentation schema fields."""
        stats = self.collapse_stats()
        return InstrumentationFields(memory_write_magnitude=stats["variance"])
