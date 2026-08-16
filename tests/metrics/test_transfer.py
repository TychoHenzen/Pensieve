from __future__ import annotations

from eval.metrics import ProbeLogEntry
from eval.metrics.transfer import backward_transfer, forward_transfer
from eval.subject import CostCounters


def _entry(
    position: int,
    probe_id: str,
    task_id: str,
    correct: bool,
    teaching_position: int,
) -> ProbeLogEntry:
    return ProbeLogEntry(
        position=position,
        probe_id=probe_id,
        task_id=task_id,
        teaching_position=teaching_position,
        correct=correct,
        cost_counters=CostCounters(),
    )


# covers: eval/metrics :: Backward transfer :: task-wiper backward transfer is strongly negative
def test_task_wiper_backward_transfer_is_strongly_negative():
    R = [
        [1.0, 0.0, 0.0],
        [None, 1.0, 0.0],
        [None, None, 1.0],
    ]
    result = backward_transfer(R)
    assert result == [-1.0, -1.0]
    assert all(value < -0.5 for value in result)


# covers: eval/metrics :: Backward transfer :: perfect-memory backward transfer is zero
def test_perfect_memory_backward_transfer_is_zero():
    R = [
        [1.0, 1.0, 1.0],
        [None, 1.0, 1.0],
        [None, None, 1.0],
    ]
    result = backward_transfer(R)
    assert result == [0.0, 0.0]


# covers: eval/metrics :: Forward transfer :: chance oracle forward transfer is zero
def test_chance_oracle_forward_transfer_is_zero():
    task_order = ["t0", "t1"]
    chance_rates = {"t1": 0.5}
    log = [
        _entry(0, "p-t0-a", "t0", True, teaching_position=0),
        _entry(1, "p-t0-b", "t0", False, teaching_position=0),
        # probes for t1 seen before t1 is taught (teaching_position=10),
        # correct at chance rate (0.5)
        _entry(2, "p-t1-a", "t1", True, teaching_position=10),
        _entry(3, "p-t1-b", "t1", False, teaching_position=10),
        _entry(4, "p-t1-c", "t1", True, teaching_position=10),
        _entry(5, "p-t1-d", "t1", False, teaching_position=10),
        _entry(10, "p-t1-taught", "t1", True, teaching_position=10),
    ]
    result = forward_transfer(log, task_order, chance_rates)
    assert len(result) == 1
    assert abs(result[0]) < 1e-9
