from __future__ import annotations

from eval.metrics import ProbeLogEntry
from eval.metrics.compute import compute_per_input
from eval.subject import CostCounters


def _entry(position: int, counters: CostCounters) -> ProbeLogEntry:
    return ProbeLogEntry(
        position=position,
        probe_id=f"p-{position}",
        task_id="assoc",
        teaching_position=0,
        correct=True,
        cost_counters=counters,
    )


# covers: eval/metrics :: Compute per input :: three distinct counters
def test_three_distinct_counters_are_independently_derived() -> None:
    log = [
        _entry(0, CostCounters(steps=3, flops=100, wall_seconds=1.0)),
        _entry(1, CostCounters(steps=7, flops=250, wall_seconds=2.5)),
        _entry(2, CostCounters(steps=12, flops=400, wall_seconds=4.0)),
    ]

    result = compute_per_input(log)

    # Each sequence is the difference between consecutive probes' counters,
    # with the first probe measured from the zero-valued initial counters.
    assert result.steps == [3, 4, 5]
    assert result.flops == [100, 150, 150]
    assert result.wall_seconds == [1.0, 1.5, 1.5]


# covers: eval/metrics :: Compute per input :: three distinct counters
def test_counters_stay_independent_when_one_does_not_move() -> None:
    log = [
        _entry(0, CostCounters(steps=5, flops=42, wall_seconds=0.5)),
        _entry(1, CostCounters(steps=8, flops=42, wall_seconds=0.5)),
    ]

    result = compute_per_input(log)

    # A probe may advance steps while FLOPs and wall time are unchanged;
    # the three counters are reported independently of one another.
    assert result.steps == [5, 3]
    assert result.flops == [42, 0]
    assert result.wall_seconds == [0.5, 0.0]
