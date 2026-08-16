from __future__ import annotations

from collections import defaultdict

from eval.metrics import ProbeLogEntry


def retention_matrix(
    phases: list[list[ProbeLogEntry]], task_order: list[str]
) -> list[list[float | None]]:
    """Build the retention matrix R[i][j] from pre-split measurement phases.

    Each inner list in *phases* holds the probe entries measured after
    task j finished (phase j).  R[i][j] is the accuracy on task i
    probes within phase j.  Entries where j < i are None because task i
    had not been introduced yet.
    """
    num_tasks = len(task_order)
    task_index = {tid: idx for idx, tid in enumerate(task_order)}

    matrix: list[list[float | None]] = [
        [None] * num_tasks for _ in range(num_tasks)
    ]

    for phase_j, phase_entries in enumerate(phases):
        if phase_j >= num_tasks:
            break
        totals: dict[str, int] = defaultdict(int)
        corrects: dict[str, int] = defaultdict(int)
        for entry in phase_entries:
            totals[entry.task_id] += 1
            if entry.correct:
                corrects[entry.task_id] += 1
        for tid, total in totals.items():
            row = task_index.get(tid)
            if row is None or phase_j < row:
                continue
            matrix[row][phase_j] = corrects[tid] / total

    return matrix
