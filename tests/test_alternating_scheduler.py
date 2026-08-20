"""Tests for fixed-budget alternating optimizer scheduling."""

from __future__ import annotations

from dataclasses import dataclass

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


def test_phase_boundary_keeps_the_epoch_dataset_cursor() -> None:
    eggroll = FakeEngine("eggroll")
    gradient = FakeEngine("gradient")
    scheduler = FixedBudgetScheduler(
        phase_steps=2,
        eggroll_engine=eggroll,
        gradient_engine=gradient,
    )

    for epoch in range(1, 3):
        for example_position in range(5):
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
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
    ]
    assert gradient.calls[0] == (
        (1, 2),
        ExperimentPosition("gradient", 2, 3, 1, 2, 1),
    )


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
