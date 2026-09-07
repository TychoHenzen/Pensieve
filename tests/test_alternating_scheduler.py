"""Tests for average-variance hysteresis optimizer scheduling."""

from __future__ import annotations

from dataclasses import dataclass

from train.alternating_scheduler import VarianceHysteresisScheduler
from train.training_results import ExperimentPosition


@dataclass(frozen=True)
class FakeResult:
    update_method: str
    position: ExperimentPosition
    shared_variance: float
    consumed_record_count: int = 1


class FakeEngine:
    """Records scheduler calls and returns configured variance values."""

    def __init__(
        self,
        update_method: str,
        variances: list[float],
        max_consumed_records: int = 1,
    ) -> None:
        self.update_method = update_method
        self.variances = variances
        self.max_consumed_records = max_consumed_records
        self.calls: list[tuple[object, ExperimentPosition]] = []

    def train_step(self, example: object, position: ExperimentPosition) -> FakeResult:
        self.calls.append((example, position))
        variance = self.variances.pop(0) if self.variances else 0.015
        consumed_record_count = (
            len(example) if isinstance(example, tuple) and example and isinstance(example[0], tuple) else 1
        )
        return FakeResult(
            self.update_method,
            position,
            variance,
            consumed_record_count,
        )


def _scheduler(
    eggroll_variances: list[float], gradient_variances: list[float]
) -> tuple[VarianceHysteresisScheduler[FakeResult, object], FakeEngine, FakeEngine]:
    eggroll = FakeEngine("eggroll", eggroll_variances)
    gradient = FakeEngine("gradient", gradient_variances)
    scheduler = VarianceHysteresisScheduler(
        phase_steps=2,
        variance_lower_threshold=0.01,
        variance_upper_threshold=0.02,
        eggroll_engine=eggroll,
        gradient_engine=gradient,
    )
    return scheduler, eggroll, gradient


# covers: train/alternating-cycle :: Average-variance hysteresis optimizer control :: Hysteresis band retains the active method
def test_eggroll_stays_active_while_average_variance_is_below_upper_threshold() -> None:
    scheduler, eggroll, gradient = _scheduler([0.005, 0.015, 0.02], [])

    for example in range(3):
        scheduler.train_step(example, epoch=1, example_position=example)

    assert len(eggroll.calls) == 3
    assert gradient.calls == []
    assert eggroll.calls[2][1] == ExperimentPosition("eggroll", 2, 3, 1, 2, 1, 3, 1)


# covers: train/alternating-cycle :: Average-variance hysteresis optimizer control :: Eggroll restores variance
def test_eggroll_switches_to_gradient_when_window_average_reaches_upper_threshold() -> None:
    scheduler, eggroll, gradient = _scheduler([0.015, 0.025], [0.015])

    for example in range(3):
        scheduler.train_step(example, epoch=1, example_position=example)

    assert len(eggroll.calls) == 2
    assert gradient.calls == [(2, ExperimentPosition("gradient", 2, 3, 1, 2, 1))]


# covers: train/alternating-cycle :: Average-variance hysteresis optimizer control :: Gradient detects collapse
def test_gradient_holds_in_hysteresis_band_and_returns_below_lower_threshold() -> None:
    scheduler, eggroll, gradient = _scheduler(
        [0.03, 0.03, 0.015],
        [0.015, 0.015, 0.005, 0.005],
    )

    for example in range(7):
        scheduler.train_step(example, epoch=1, example_position=example)

    assert len(gradient.calls) == 4
    assert eggroll.calls[-1] == (
        6,
        ExperimentPosition("eggroll", 4, 7, 1, 6, 1, 3, 1),
    )


def test_epoch_boundary_does_not_reset_the_variance_window() -> None:
    scheduler, eggroll, gradient = _scheduler([0.03, 0.03, 0.015], [0.015])

    scheduler.train_step((1, 0), epoch=1, example_position=0)
    scheduler.train_step((2, 0), epoch=2, example_position=0)
    scheduler.train_step((2, 1), epoch=2, example_position=1)

    assert eggroll.calls[1] == (
        (2, 0),
        ExperimentPosition("eggroll", 1, 2, 2, 0, 2, 2, 1),
    )
    assert gradient.calls[0] == (
        (2, 1),
        ExperimentPosition("gradient", 2, 3, 2, 1, 1),
    )


def test_restore_preserves_active_phase_window_position_and_variance_sum() -> None:
    scheduler, _, gradient = _scheduler([], [0.015])

    scheduler.restore(
        active_phase="gradient",
        completed_steps=9,
        completed_phase_steps=1,
        phase_variance_sum=0.014,
    )
    result = scheduler.train_step("next", epoch=3, example_position=4)

    assert result.position == ExperimentPosition("gradient", 5, 10, 3, 4, 2)
    assert gradient.calls == [("next", result.position)]
    assert scheduler.active_phase == "gradient"
    assert scheduler.completed_phase_steps == 0
    assert scheduler.phase_variance_sum == 0.0


# covers: train/alternating-cycle :: Average-variance hysteresis optimizer control :: Eggroll batch ends at the observation boundary
def test_eggroll_batch_stops_at_the_observation_window_boundary() -> None:
    eggroll = FakeEngine("eggroll", [0.03], max_consumed_records=8)
    gradient = FakeEngine("gradient", [0.015])
    scheduler = VarianceHysteresisScheduler(
        phase_steps=3,
        variance_lower_threshold=0.01,
        variance_upper_threshold=0.02,
        eggroll_engine=eggroll,
        gradient_engine=gradient,
    )

    result = scheduler.train_records(
        [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")],
        epoch=1,
        example_position=0,
    )
    following = scheduler.train_records([("d", "4")], epoch=1, example_position=3)

    assert eggroll.calls == [
        (
            (("a", "1"), ("b", "2"), ("c", "3")),
            ExperimentPosition("eggroll", 1, 3, 1, 2, 3, 1, 3),
        )
    ]
    assert result.consumed_record_count == 3
    assert scheduler.active_phase == "gradient"
    assert following.position == ExperimentPosition("gradient", 2, 4, 1, 3, 1)


# covers: train/alternating-cycle :: Average-variance hysteresis optimizer control :: Invalid variance controller
def test_invalid_variance_controller_is_rejected() -> None:
    eggroll = FakeEngine("eggroll", [])
    gradient = FakeEngine("gradient", [])

    try:
        VarianceHysteresisScheduler(
            phase_steps=0,
            eggroll_engine=eggroll,
            gradient_engine=gradient,
        )
    except ValueError as error:
        assert str(error) == "phase_steps must be at least 1"
    else:
        raise AssertionError("invalid variance controller was accepted")
