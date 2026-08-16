from __future__ import annotations

from eval.metrics import ProbeLogEntry
from eval.metrics.retention import retention_matrix
from eval.subject import CostCounters

TASK_ORDER = ["t0", "t1", "t2"]
PROBES_PER_TASK = 2


def _entry(
    position: int,
    probe_id: str,
    task_id: str,
    correct: bool,
    teaching_position: int = 0,
) -> ProbeLogEntry:
    return ProbeLogEntry(
        position=position,
        probe_id=probe_id,
        task_id=task_id,
        teaching_position=teaching_position,
        correct=correct,
        cost_counters=CostCounters(),
    )


def _split_classify_phases(correct_for: object) -> list[list[ProbeLogEntry]]:
    """Build split-classify-style measurement phases.

    Phase j probes tasks 0..j in ascending order, PROBES_PER_TASK
    probes per task.  correct_for(phase, task_index) decides correctness.
    """
    phases: list[list[ProbeLogEntry]] = []
    position = 0
    for phase in range(len(TASK_ORDER)):
        entries: list[ProbeLogEntry] = []
        for task_index in range(phase + 1):
            task_id = TASK_ORDER[task_index]
            correct = correct_for(phase, task_index)
            for probe_num in range(PROBES_PER_TASK):
                entries.append(
                    _entry(
                        position,
                        f"p-{task_id}-{phase}-{probe_num}",
                        task_id,
                        correct,
                    )
                )
                position += 1
        phases.append(entries)
    return phases


# covers: eval/metrics :: Retention matrix :: perfect-memory retention is all ones
def test_perfect_memory_retention_is_all_ones():
    phases = _split_classify_phases(lambda phase, task_index: True)
    matrix = retention_matrix(phases, TASK_ORDER)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            if col_index < row_index:
                assert value is None
            else:
                assert value == 1.0


# covers: eval/metrics :: Retention matrix :: task-wiper shows diagonal ones and below-diagonal at chance
def test_task_wiper_retention_diagonal_ones_off_diagonal_zero():
    phases = _split_classify_phases(lambda phase, task_index: task_index == phase)
    matrix = retention_matrix(phases, TASK_ORDER)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            if col_index < row_index:
                assert value is None
            elif col_index == row_index:
                assert value == 1.0
            else:
                assert value == 0.0
