from __future__ import annotations

METRICS_VERSION = "1"

from dataclasses import dataclass

from eval.subject import CostCounters


@dataclass(frozen=True)
class ProbeLogEntry:
    position: int
    probe_id: str
    task_id: str
    teaching_position: int
    correct: bool
    cost_counters: CostCounters
