"""Tests for resumable alternating-training checkpoints."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pytest
import torch
from torch import nn

import train.alternating_checkpoint as alternating_checkpoint
from train.alternating_checkpoint import (
    AlternatingCheckpoint,
    CHECKPOINT_VERSION,
    CheckpointSchedule,
    capture_checkpoint,
    latest_checkpoint,
    load_checkpoint,
    save_boundary_checkpoints,
    save_checkpoint,
)


SCHEDULE_CONFIG = {
    "dataset_selection": "gsm8k-train",
    "epochs": 5,
    "phase_steps": 500,
    "model_shape": {"hidden_size": 64, "num_slots": 8},
    "gradient_optimizer": {"learning_rate": 1e-4},
    "eggroll_optimizer": {"learning_rate": 1e-3},
    "eggroll_population": {"size": 128, "sigma": 0.01, "rank": 4},
    "held_out_selection": {"split": "test", "count": 128},
    "logging_frequency": 10,
}


def test_checkpoint_round_trips_resumption_state(tmp_path) -> None:
    model = nn.Linear(2, 1)
    eggroll_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    gradient_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    _step(eggroll_optimizer, model, gradient=1.0)
    _step(gradient_optimizer, model, gradient=2.0)
    schedule = CheckpointSchedule("gradient", 17, 517, 2, 16)
    model_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    random.seed(123)
    torch.manual_seed(456)
    expected_python_random = random.getstate()
    expected_torch_random = torch.get_rng_state()

    checkpoint = capture_checkpoint(
        model_state=model_state,
        eggroll_optimizer=eggroll_optimizer,
        gradient_optimizer=gradient_optimizer,
        schedule=schedule,
        run_config={"phase_steps": 500, "epochs": 5},
        held_out_selection=[{"question": "q", "answer": "1"}],
        metrics={"answer_exact_match": 0.25},
    )
    path = tmp_path / "boundary.pt"
    save_checkpoint(path, checkpoint)
    loaded = load_checkpoint(path)

    assert loaded.version == CHECKPOINT_VERSION
    assert loaded.schedule == schedule
    assert loaded.run_config == {"phase_steps": 500, "epochs": 5}
    assert loaded.held_out_selection == [{"question": "q", "answer": "1"}]
    assert loaded.metrics == {"answer_exact_match": 0.25}
    assert loaded.python_random_state == expected_python_random
    assert torch.equal(loaded.torch_random_state, expected_torch_random)
    assert set(loaded.model_state) == {"weight", "bias"}
    assert all(torch.equal(loaded.model_state[name], value) for name, value in model_state.items())
    assert _same_state(loaded.eggroll_optimizer_state, eggroll_optimizer.state_dict())
    assert _same_state(loaded.gradient_optimizer_state, gradient_optimizer.state_dict())


def test_phase_boundary_resume_starts_next_example_in_saved_phase(tmp_path) -> None:
    scheduler = FakeResumableScheduler(phase_steps=2)
    schedule = scheduler.run_until_phase_boundary(["first", "second", "third"])
    checkpoint = capture_checkpoint(
        model_state={},
        eggroll_optimizer=_optimizer(),
        gradient_optimizer=_optimizer(),
        schedule=schedule,
        run_config={"phase_steps": 2},
        held_out_selection=[],
        metrics=[],
    )
    path = tmp_path / "phase-boundary.pt"
    save_checkpoint(path, checkpoint)

    resumed_scheduler = FakeResumableScheduler(phase_steps=2)
    resumed_scheduler.resume(load_checkpoint(path).schedule, ["first", "second", "third"])

    assert schedule == CheckpointSchedule(
        active_phase="gradient",
        completed_phase_steps=0,
        global_step=2,
        epoch=1,
        dataset_position=1,
    )
    assert resumed_scheduler.calls == [("third", "gradient", 2, 3)]


def test_epoch_boundary_resume_starts_next_epoch_with_unfinished_phase_budget(tmp_path) -> None:
    scheduler = FakeResumableScheduler(phase_steps=5)
    schedule = scheduler.run_until_epoch_boundary(["first", "second", "third"])
    checkpoint = capture_checkpoint(
        model_state={},
        eggroll_optimizer=_optimizer(),
        gradient_optimizer=_optimizer(),
        schedule=schedule,
        run_config={"phase_steps": 5},
        held_out_selection=[],
        metrics=[],
    )
    path = tmp_path / "epoch-boundary.pt"
    save_checkpoint(path, checkpoint)

    resumed_scheduler = FakeResumableScheduler(phase_steps=5)
    loaded_schedule = load_checkpoint(path).schedule
    resumed_scheduler.resume(loaded_schedule, ["first", "second", "third"])

    assert loaded_schedule == CheckpointSchedule(
        active_phase="eggroll",
        completed_phase_steps=3,
        global_step=3,
        epoch=1,
        dataset_position=2,
    )
    assert resumed_scheduler.calls == [("first", "eggroll", 0, 4)]
    assert resumed_scheduler.resume_positions == [(2, 0, 3)]


@pytest.mark.parametrize(
    ("setting", "override"),
    [
        ("dataset_selection", "gsm8k-train-subset"),
        ("phase_steps", 250),
        ("model_shape", {"hidden_size": 128, "num_slots": 8}),
        ("gradient_optimizer", {"learning_rate": 2e-4}),
        ("eggroll_optimizer", {"learning_rate": 2e-3}),
        ("eggroll_population", {"size": 64, "sigma": 0.01, "rank": 4}),
        ("held_out_selection", {"split": "test", "count": 64}),
    ],
)
def test_resume_rejects_each_changed_schedule_setting_before_training(
    setting: str, override: object
) -> None:
    resume_config = dict(SCHEDULE_CONFIG)
    resume_config[setting] = override
    training_updates: list[str] = []

    with pytest.raises(ValueError, match=setting):
        _validate_then_update(resume_config, completed_epochs=2, updates=training_updates)

    assert training_updates == []


def test_resume_rejects_epoch_target_below_completed_progress_before_training() -> None:
    resume_config = dict(SCHEDULE_CONFIG, epochs=1)
    training_updates: list[str] = []

    with pytest.raises(ValueError, match="epochs"):
        _validate_then_update(resume_config, completed_epochs=2, updates=training_updates)

    assert training_updates == []


def test_resume_error_names_every_conflicting_setting() -> None:
    resume_config = dict(
        SCHEDULE_CONFIG,
        phase_steps=250,
        model_shape={"hidden_size": 128, "num_slots": 8},
        held_out_selection={"split": "test", "count": 64},
    )

    with pytest.raises(ValueError) as error:
        _validate_then_update(resume_config, completed_epochs=2, updates=[])

    assert "phase_steps" in str(error.value)
    assert "model_shape" in str(error.value)
    assert "held_out_selection" in str(error.value)


def test_resume_accepts_matching_schedule_and_display_only_changes() -> None:
    resume_config = dict(SCHEDULE_CONFIG, epochs=8, logging_frequency=1)
    training_updates: list[str] = []

    _validate_then_update(resume_config, completed_epochs=2, updates=training_updates)

    assert training_updates == ["updated"]


def test_latest_checkpoint_uses_stored_global_step_not_filename_order(tmp_path) -> None:
    lower_step = _checkpoint(global_step=9, epoch=100)
    higher_step = _checkpoint(global_step=10, epoch=2)
    save_checkpoint(tmp_path / "epoch-100.pt", lower_step)
    save_checkpoint(tmp_path / "phase-10.pt", higher_step)

    assert latest_checkpoint(tmp_path) == tmp_path / "phase-10.pt"


def test_latest_checkpoint_returns_none_for_missing_directory(tmp_path) -> None:
    assert latest_checkpoint(tmp_path / "missing") is None


def test_latest_checkpoint_returns_none_for_empty_directory(tmp_path) -> None:
    assert latest_checkpoint(tmp_path) is None


def test_coincident_boundaries_serialize_once_and_expose_both_names(
    tmp_path, monkeypatch
) -> None:
    checkpoint = _checkpoint(global_step=12, epoch=3)
    save_calls = 0
    real_save = torch.save

    def counting_save(*args, **kwargs) -> None:
        nonlocal save_calls
        save_calls += 1
        real_save(*args, **kwargs)

    monkeypatch.setattr(torch, "save", counting_save)

    paths = save_boundary_checkpoints(
        tmp_path,
        checkpoint,
        phase_boundary=True,
        epoch_boundary=True,
    )

    assert paths == (tmp_path / "phase-12.pt", tmp_path / "epoch-3.pt")
    assert save_calls == 1
    assert all(path.exists() for path in paths)
    assert paths[0].samefile(paths[1])
    assert load_checkpoint(paths[0]).schedule == checkpoint.schedule
    assert load_checkpoint(paths[1]).schedule == checkpoint.schedule


def _step(optimizer: torch.optim.Optimizer, model: nn.Module, *, gradient: float) -> None:
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, gradient)
    optimizer.step()
    optimizer.zero_grad()


def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD([nn.Parameter(torch.zeros(()))], lr=0.1)


def _checkpoint(*, global_step: int, epoch: int) -> AlternatingCheckpoint:
    return AlternatingCheckpoint(
        model_state={},
        eggroll_optimizer_state={},
        gradient_optimizer_state={},
        schedule=CheckpointSchedule("eggroll", 0, global_step, epoch, 0),
        run_config={},
        held_out_selection=[],
        metrics={},
        python_random_state=random.getstate(),
        torch_random_state=torch.get_rng_state(),
    )


def _validate_then_update(
    resume_config: dict[str, object], *, completed_epochs: int, updates: list[str]
) -> None:
    validate_resume_config = getattr(
        alternating_checkpoint,
        "validate_resume_config",
        lambda **_: None,
    )
    validate_resume_config(
        checkpoint_config=SCHEDULE_CONFIG,
        resume_config=resume_config,
        completed_epochs=completed_epochs,
    )
    updates.append("updated")


@dataclass
class FakeResumableScheduler:
    """Minimal command-side scheduler fake for checkpoint resume behavior."""

    phase_steps: int
    calls: list[tuple[object, str, int, int]] = field(default_factory=list)
    resume_positions: list[tuple[int, int, int]] = field(default_factory=list)

    def run_until_phase_boundary(self, examples: list[object]) -> CheckpointSchedule:
        for dataset_position, example in enumerate(examples[: self.phase_steps]):
            self.calls.append((example, "eggroll", dataset_position, dataset_position + 1))
        return CheckpointSchedule(
            active_phase="gradient",
            completed_phase_steps=0,
            global_step=self.phase_steps,
            epoch=1,
            dataset_position=self.phase_steps - 1,
        )

    def run_until_epoch_boundary(self, examples: list[object]) -> CheckpointSchedule:
        for dataset_position, example in enumerate(examples):
            self.calls.append((example, "eggroll", dataset_position, dataset_position + 1))
        return CheckpointSchedule(
            active_phase="eggroll",
            completed_phase_steps=len(examples),
            global_step=len(examples),
            epoch=1,
            dataset_position=len(examples) - 1,
        )

    def resume(self, schedule: CheckpointSchedule, examples: list[object]) -> None:
        next_position = schedule.dataset_position + 1
        next_epoch = schedule.epoch
        if next_position == len(examples):
            next_epoch += 1
            next_position = 0
        self.resume_positions.append(
            (next_epoch, next_position, schedule.completed_phase_steps)
        )
        self.calls.append(
            (
                examples[next_position],
                schedule.active_phase,
                next_position,
                schedule.global_step + 1,
            )
        )


def _same_state(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_state(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_state(*pair) for pair in zip(left, right))
    return left == right
