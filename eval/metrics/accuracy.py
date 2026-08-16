from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from eval.metrics import ProbeLogEntry


@dataclass(frozen=True)
class TaskAccuracyResult:
    per_task: dict[str, float]
    per_probe_class: dict[str, float]
    pooled: float


def task_accuracy(log: list[ProbeLogEntry]) -> TaskAccuracyResult:
    task_correct: dict[str, int] = defaultdict(int)
    task_total: dict[str, int] = defaultdict(int)
    probe_correct: dict[str, int] = defaultdict(int)
    probe_total: dict[str, int] = defaultdict(int)
    pooled_correct = 0
    pooled_total = 0

    for entry in log:
        task_total[entry.task_id] += 1
        probe_total[entry.probe_id] += 1
        pooled_total += 1
        if entry.correct:
            task_correct[entry.task_id] += 1
            probe_correct[entry.probe_id] += 1
            pooled_correct += 1

    per_task = {
        task_id: task_correct[task_id] / total
        for task_id, total in task_total.items()
    }
    per_probe_class = {
        probe_id: probe_correct[probe_id] / total
        for probe_id, total in probe_total.items()
    }
    pooled = pooled_correct / pooled_total if pooled_total else 0.0

    return TaskAccuracyResult(
        per_task=per_task,
        per_probe_class=per_probe_class,
        pooled=pooled,
    )
