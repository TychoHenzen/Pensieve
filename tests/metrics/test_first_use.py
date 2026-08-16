from __future__ import annotations

from eval.metrics import ProbeLogEntry
from eval.metrics.first_use import time_to_first_use
from eval.subject import CostCounters


def _entry(
    position: int,
    probe_id: str,
    task_id: str,
    teaching_position: int,
    correct: bool,
) -> ProbeLogEntry:
    return ProbeLogEntry(
        position=position,
        probe_id=probe_id,
        task_id=task_id,
        teaching_position=teaching_position,
        correct=correct,
        cost_counters=CostCounters(),
    )


# covers: eval/metrics :: Time to first use :: perfect-memory oracle first use equals distance to next probe
def test_perfect_memory_first_use_equals_distance_to_next_probe():
    log = [
        _entry(0, "p0", "t0", teaching_position=0, correct=False),
        _entry(1, "p1", "t0", teaching_position=1, correct=False),
        _entry(2, "p2", "t0", teaching_position=2, correct=False),
        _entry(5, "p0", "t0", teaching_position=0, correct=True),
        _entry(6, "p1", "t0", teaching_position=1, correct=True),
        _entry(7, "p2", "t0", teaching_position=2, correct=True),
    ]
    result = time_to_first_use(log, cutoff=100)
    assert result.per_fact == {"p0": 5, "p1": 5, "p2": 5}
    assert result.median == 5.0


# covers: eval/metrics :: Time to first use :: cut-off share is zero for perfect memory
def test_perfect_memory_cutoff_share_is_zero():
    log = [
        _entry(0, "p0", "t0", teaching_position=0, correct=False),
        _entry(5, "p0", "t0", teaching_position=0, correct=True),
    ]
    result = time_to_first_use(log, cutoff=100)
    assert result.cutoff_share == 0.0


def test_mixed_never_correct_facts_get_cutoff_and_positive_share():
    log = [
        _entry(5, "p0", "t0", teaching_position=0, correct=True),
        _entry(3, "p1", "t0", teaching_position=0, correct=False),
    ]
    result = time_to_first_use(log, cutoff=10)
    assert result.per_fact == {"p0": 5, "p1": 10}
    assert result.cutoff_share == 0.5
    assert result.median == 7.5


def test_all_incorrect_gives_full_cutoff_share_and_median():
    log = [
        _entry(1, "p0", "t0", teaching_position=0, correct=False),
        _entry(2, "p1", "t0", teaching_position=0, correct=False),
    ]
    result = time_to_first_use(log, cutoff=10)
    assert result.cutoff_share == 1.0
    assert result.median == 10.0
    assert result.per_fact == {"p0": 10, "p1": 10}
