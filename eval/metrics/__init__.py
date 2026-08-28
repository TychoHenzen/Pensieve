from __future__ import annotations

from dataclasses import dataclass

from eval.subject import CostCounters

METRICS_VERSION = "1"


@dataclass(frozen=True)
class ProbeLogEntry:
    position: int
    probe_id: str
    task_id: str
    teaching_position: int
    correct: bool
    cost_counters: CostCounters
