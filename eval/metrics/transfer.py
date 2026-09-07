from __future__ import annotations

from eval.metrics import ProbeLogEntry


def backward_transfer(R: list[list[float | None]]) -> list[float]:
    """Compute backward transfer per task from a retention matrix.

    R[i][j] is accuracy on task i after task j finishes (j >= i).
    BWT_i = R[i][T-1] - R[i][i], skipping the last task and any
    entries where the final-position accuracy is unavailable.
    """
    num_tasks = len(R)
    if num_tasks == 0:
        return []

    last_index = num_tasks - 1
    results: list[float] = []
    for i in range(num_tasks - 1):
        final_accuracy = R[i][last_index]
        initial_accuracy = R[i][i]
        if final_accuracy is None or initial_accuracy is None:
            continue
        results.append(final_accuracy - initial_accuracy)
    return results


def forward_transfer(
    log: list[ProbeLogEntry],
    task_order: list[str],
    chance_rates: dict[str, float],
) -> list[float]:
    """Compute forward transfer per task (excluding the first task).

    FWT_i = accuracy on task i probes seen before task i was taught,
    minus chance_rates[task_id]. Tasks with no such probes score 0.0.
    """
    results: list[float] = []
    for task_id in task_order[1:]:
        task_entries = [entry for entry in log if entry.task_id == task_id]
        if not task_entries:
            results.append(0.0)
            continue

        teaching_position = task_entries[0].teaching_position
        pre_teaching_entries = [entry for entry in task_entries if entry.position < teaching_position]
        if not pre_teaching_entries:
            results.append(0.0)
            continue

        correct_count = sum(1 for entry in pre_teaching_entries if entry.correct)
        accuracy = correct_count / len(pre_teaching_entries)
        results.append(accuracy - chance_rates[task_id])
    return results
