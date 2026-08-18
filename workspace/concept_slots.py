from __future__ import annotations

import torch

SLOT_DIM = 768
ABLATION_SLOT_COUNTS = (1, 4, 8, 16, 32, 64)
DEFAULT_SLOT_COUNT = 16


class Workspace:
    """Holds a configurable number of concept slots, each of dimension 768."""

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
        """Stub: variance and covariance across slots. Filled in by task 1.3."""
        if self.slot_count < 2:
            return {"variance": 0.0, "covariance": 0.0}
        variance = self.slots.var(dim=0).mean().item()
        centered = self.slots - self.slots.mean(dim=0, keepdim=True)
        cov = (centered @ centered.T) / max(SLOT_DIM - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        covariance = off_diag.abs().mean().item()
        return {"variance": variance, "covariance": covariance}
