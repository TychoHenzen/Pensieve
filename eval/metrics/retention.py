"""Retention matrix: per-task accuracy at each subsequent measurement phase.

A split-classify-style stream teaches tasks in sequence and, after each
task's teaching examples, probes every task seen so far (task 0, then
task 1, ... up to the task just taught) before moving on. Each such
probe block is a measurement phase. `retention_matrix` groups the probe
log into those phases and reports, for each task i and phase j, the
accuracy on task i's probes within phase j.

Phases are recovered from `task_id` order alone, using the fact that
phase j must contain exactly j + 1 distinct tasks (task 0 through task
j), because the generator probes tasks 0..j in ascending order before
moving to the next teaching block. `retention_matrix` first collapses
the log into runs of consecutive identical `task_id`s (a single task's
probes stay together within a phase), then greedily assigns runs to
phases: once a phase has accumulated its required distinct-task count,
the next run starts a new phase. This does not require knowing how
many probes were issued per task, so it works whether `probes_per_task`
is 1 or many.

R[i][j] is populated only when task i had been taught by phase j (i.e.
j >= i), since a phase before task i's introduction contains no probes
for it. Unmeasured entries stay `None`.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import groupby

from eval.metrics import ProbeLogEntry


def retention_matrix(
    log: list[ProbeLogEntry], task_order: list[str]
) -> list[list[float | None]]:
    num_tasks = len(task_order)
    task_index = {task_id: index for index, task_id in enumerate(task_order)}

    runs = [
        (task_id, list(entries))
        for task_id, entries in groupby(log, key=lambda entry: entry.task_id)
    ]

    phases: list[list[ProbeLogEntry]] = []
    phase_index = 0
    distinct_in_phase = 0
    last_task_in_phase: str | None = None
    for task_id, entries in runs:
        if phase_index >= num_tasks:
            break
        if not phases or len(phases) <= phase_index:
            phases.append([])
        if task_id != last_task_in_phase:
            distinct_in_phase += 1
            last_task_in_phase = task_id
        phases[phase_index].extend(entries)
        if distinct_in_phase == phase_index + 1:
            phase_index += 1
            distinct_in_phase = 0
            last_task_in_phase = None

    matrix: list[list[float | None]] = [
        [None] * num_tasks for _ in range(num_tasks)
    ]

    for phase_index, phase in enumerate(phases):
        if phase_index >= num_tasks:
            break
        totals: dict[str, int] = defaultdict(int)
        corrects: dict[str, int] = defaultdict(int)
        for entry in phase:
            totals[entry.task_id] += 1
            if entry.correct:
                corrects[entry.task_id] += 1
        for task_id, total in totals.items():
            row = task_index.get(task_id)
            if row is None:
                continue
            matrix[row][phase_index] = corrects[task_id] / total

    return matrix
