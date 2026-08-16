from __future__ import annotations

from eval.metrics import ProbeLogEntry
from eval.metrics.accuracy import task_accuracy
from eval.subject import CostCounters


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


# covers: eval/metrics :: Task accuracy :: perfect-memory oracle scores 1.0 on every task
def test_all_correct_yields_one_per_task_and_pooled():
    log = [
        _entry(0, "p-cat", "assoc", True),
        _entry(1, "p-dog", "assoc", True),
        _entry(2, "p-cow", "assoc", True),
    ]
    result = task_accuracy(log)
    assert result.per_task == {"assoc": 1.0}
    assert result.pooled == 1.0


def test_mixed_correctness_across_two_tasks():
    log = [
        _entry(0, "p-cat", "assoc", True),
        _entry(1, "p-dog", "assoc", False),
        _entry(2, "p-cow", "assoc", True),
        _entry(3, "p-split1", "split", True),
        _entry(4, "p-split2", "split", False),
    ]
    result = task_accuracy(log)
    assert result.per_task["assoc"] == 2 / 3
    assert result.per_task["split"] == 1 / 2
    assert result.pooled == 3 / 5


def test_all_incorrect_yields_zero_pooled():
    log = [
        _entry(0, "p-cat", "assoc", False),
        _entry(1, "p-dog", "assoc", False),
    ]
    result = task_accuracy(log)
    assert result.per_task == {"assoc": 0.0}
    assert result.pooled == 0.0


def test_single_task_has_one_per_task_entry():
    log = [
        _entry(0, "p-cat", "assoc", True),
        _entry(1, "p-dog", "assoc", False),
    ]
    result = task_accuracy(log)
    assert len(result.per_task) == 1
    assert "assoc" in result.per_task


def test_multiple_probe_classes_group_correctly():
    log = [
        _entry(0, "p-cat", "assoc", True),
        _entry(1, "p-cat", "assoc", False),
        _entry(2, "p-dog", "assoc", True),
    ]
    result = task_accuracy(log)
    assert result.per_probe_class["p-cat"] == 0.5
    assert result.per_probe_class["p-dog"] == 1.0
