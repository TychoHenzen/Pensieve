"""Tests for fixed-budget alternating optimizer scheduling."""

from __future__ import annotations

from dataclasses import dataclass

from train import run_alternating
from train.alternating_scheduler import FixedBudgetScheduler
from train.training_results import ExperimentPosition


@dataclass
class FakeEngine:
    """Records scheduler calls without loading a model or dataset."""

    update_method: str

    def __post_init__(self) -> None:
        self.calls: list[tuple[object, ExperimentPosition]] = []

    def train_step(self, example: object, position: ExperimentPosition) -> str:
        self.calls.append((example, position))
        return self.update_method


def test_starts_with_a_500_step_eggroll_phase() -> None:
    eggroll = FakeEngine("eggroll")
    gradient = FakeEngine("gradient")
    scheduler = FixedBudgetScheduler(
        phase_steps=500,
        eggroll_engine=eggroll,
        gradient_engine=gradient,
    )

    for example in range(1, 501):
        assert scheduler.train_step(example, epoch=1, example_position=example) == "eggroll"

    assert gradient.calls == []
    assert eggroll.calls[0] == (
        1,
        ExperimentPosition("eggroll", 1, 1, 1, 1, 1),
    )
    assert eggroll.calls[-1] == (
        500,
        ExperimentPosition("eggroll", 1, 500, 1, 500, 500),
    )


def test_step_501_starts_gradient_phase_on_next_example() -> None:
    eggroll = FakeEngine("eggroll")
    gradient = FakeEngine("gradient")
    scheduler = FixedBudgetScheduler(
        phase_steps=500,
        eggroll_engine=eggroll,
        gradient_engine=gradient,
    )

    for example in range(1, 502):
        scheduler.train_step(example, epoch=1, example_position=example)

    assert len(eggroll.calls) == 500
    assert gradient.calls == [
        (
            501,
            ExperimentPosition("gradient", 2, 501, 1, 501, 1),
        )
    ]


# covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Phase boundary within an epoch
def test_phase_boundary_keeps_the_1089_example_epoch_in_order() -> None:
    eggroll = FakeEngine("eggroll")
    gradient = FakeEngine("gradient")
    scheduler = FixedBudgetScheduler(
        phase_steps=500,
        eggroll_engine=eggroll,
        gradient_engine=gradient,
    )

    for example_position in range(1_089):
        scheduler.train_step(
            (1, example_position),
            epoch=1,
            example_position=example_position,
        )

    all_calls = sorted(
        eggroll.calls + gradient.calls,
        key=lambda call: call[1].global_step,
    )
    calls_by_step = {position.global_step: call for call, position in all_calls}

    assert [example for example, _ in all_calls] == [
        (1, position) for position in range(1_089)
    ]
    assert calls_by_step[500] == (1, 499)
    assert calls_by_step[501] == (1, 500)
    assert gradient.calls[0] == (
        (1, 500),
        ExperimentPosition("gradient", 2, 501, 1, 500, 1),
    )
    assert gradient.calls[-1] == (
        (1, 999),
        ExperimentPosition("gradient", 2, 1_000, 1, 999, 500),
    )
    assert eggroll.calls[-1] == (
        (1, 1_088),
        ExperimentPosition("eggroll", 3, 1_089, 1, 1_088, 89),
    )


# covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Epoch boundary within a phase
def test_1089_example_epoch_boundary_preserves_phase_budget() -> None:
    eggroll = FakeEngine("eggroll")
    gradient = FakeEngine("gradient")
    scheduler = FixedBudgetScheduler(
        phase_steps=500,
        eggroll_engine=eggroll,
        gradient_engine=gradient,
    )

    for epoch in range(1, 3):
        for example_position in range(1_089):
            scheduler.train_step(
                (epoch, example_position),
                epoch=epoch,
                example_position=example_position,
            )

    all_calls = sorted(
        eggroll.calls + gradient.calls,
        key=lambda call: call[1].global_step,
    )
    assert [example for example, _ in all_calls] == [
        (epoch, position)
        for epoch in range(1, 3)
        for position in range(1_089)
    ]
    assert all_calls[1_088] == (
        (1, 1_088),
        ExperimentPosition("eggroll", 3, 1_089, 1, 1_088, 89),
    )
    assert all_calls[1_089] == (
        (2, 0),
        ExperimentPosition("eggroll", 3, 1_090, 2, 0, 90),
    )


# covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Full default run
def test_default_five_epochs_visit_all_5445_positions_without_phase_reset() -> None:
    epochs = run_alternating._parse_args([]).epochs
    eggroll = FakeEngine("eggroll")
    gradient = FakeEngine("gradient")
    scheduler = FixedBudgetScheduler(
        phase_steps=500,
        eggroll_engine=eggroll,
        gradient_engine=gradient,
    )

    for epoch in range(1, epochs + 1):
        for example_position in range(1_089):
            scheduler.train_step(
                (epoch, example_position),
                epoch=epoch,
                example_position=example_position,
            )

    all_calls = sorted(
        eggroll.calls + gradient.calls,
        key=lambda call: call[1].global_step,
    )
    visits = {(position.epoch, position.example_position) for _, position in all_calls}
    calls_by_global_step = {
        position.global_step: (example, position) for example, position in all_calls
    }
    assert epochs == 5
    assert len(all_calls) == 5_445
    assert len(visits) == 5_445
    assert [example for example, _ in all_calls] == [
        (epoch, position)
        for epoch in range(1, 6)
        for position in range(1_089)
    ]
    for phase_end in range(500, 5_001, 500):
        assert calls_by_global_step[phase_end][1].phase_step == 500
        assert calls_by_global_step[phase_end + 1][1].phase_step == 1
        assert (
            calls_by_global_step[phase_end + 1][1].cycle
            == calls_by_global_step[phase_end][1].cycle + 1
        )
    assert calls_by_global_step[1_090] == (
        (2, 0),
        ExperimentPosition("eggroll", 3, 1_090, 2, 0, 90),
    )
    assert calls_by_global_step[2_179] == (
        (3, 0),
        ExperimentPosition("eggroll", 5, 2_179, 3, 0, 179),
    )
    assert calls_by_global_step[5_445][1].phase_step == 445


def test_returns_to_eggroll_after_500_gradient_steps() -> None:
    eggroll = FakeEngine("eggroll")
    gradient = FakeEngine("gradient")
    scheduler = FixedBudgetScheduler(
        phase_steps=500,
        eggroll_engine=eggroll,
        gradient_engine=gradient,
    )

    for example in range(1, 1002):
        scheduler.train_step(example, epoch=2, example_position=example)

    assert gradient.calls[-1] == (
        1000,
        ExperimentPosition("gradient", 2, 1000, 2, 1000, 500),
    )
    assert eggroll.calls[-1] == (
        1001,
        ExperimentPosition("eggroll", 3, 1001, 2, 1001, 1),
    )
