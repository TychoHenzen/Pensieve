from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentationFields:
    halting_steps: int | None = None
    memory_write_magnitude: float | None = None
    channel_bandwidth: float | None = None
    consolidation_gain: float | None = None
