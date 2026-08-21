"""Tests for resumable alternating-training checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
from torch import nn

import train.alternating_checkpoint as alternating_checkpoint
from train.alternating_checkpoint import (
    AlternatingCheckpoint,
    CHECKPOINT_VERSION,
    CheckpointSchedule,
    build_alternating_checkpoint,
    capture_checkpoint,
    latest_checkpoint,
    load_checkpoint,
    save_boundary_checkpoints,
    save_checkpoint,
)
from test_stage0_checkpoint_resume import _metadata, _plain, _tensor_values


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
    metadata = _metadata()
    tensors = _tensor_values(metadata)
    checkpoint = capture_checkpoint(metadata=metadata, tensors=tensors)
    path = tmp_path / "boundary.ckpt"
    save_checkpoint(path, checkpoint)
    loaded = load_checkpoint(path)

    assert CHECKPOINT_VERSION == 2
    assert _plain(loaded.metadata) == metadata
    assert set(loaded.tensors) == set(tensors)
    assert all(loaded.tensors[name].equal(value) for name, value in tensors.items())


def test_typed_builder_captures_real_model_optimizer_and_rng_state(
    tmp_path, monkeypatch
) -> None:
    parameters = {
        name: nn.Parameter(torch.tensor([float(index)]))
        for index, name in enumerate(alternating_checkpoint.stage0_parameter_paths())
    }
    gradient = torch.optim.Adam(parameters.values(), lr=1e-4)
    eggroll = torch.optim.Adam(parameters.values(), lr=1e-3)
    for optimizer in (gradient, eggroll):
        for parameter in parameters.values():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad()
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    fixture = _metadata()

    checkpoint = build_alternating_checkpoint(
        identity=fixture["identity"],
        selections=fixture["selections"],
        model_state=parameters,
        eggroll_optimizer=eggroll,
        gradient_optimizer=gradient,
        schedule=CheckpointSchedule("gradient", 0, 2, 1, 1),
        phase_steps=2,
        next_dataset_position=2,
        metrics={"loss": 1.25},
    )
    path = tmp_path / "phase-2.ckpt"
    save_checkpoint(path, checkpoint)
    loaded = load_checkpoint(path)

    assert loaded.metadata["schema_version"] == 2
    assert loaded.metadata["schedule"]["next_dataset_position"] == 2
    assert set(loaded.metadata["optimizer_manifests"][0]["tensor_references"]) == set(parameters)
    assert all(
        loaded.metadata["optimizer_manifests"][0]["tensor_references"][name]
        for name in parameters
    )
    assert all(
        torch.equal(loaded.tensors[f"model.{name}"], parameter.detach())
        for name, parameter in parameters.items()
    )


def test_phase_boundary_resume_starts_next_example_in_saved_phase(tmp_path) -> None:
    scheduler = FakeResumableScheduler(phase_steps=2)
    schedule = scheduler.run_until_phase_boundary(["first", "second", "third"])
    metadata = _metadata()
    metadata["schedule"] = {
        "active_phase": schedule.active_phase,
        "completed_phase_steps": schedule.completed_phase_steps,
        "phase_steps": 2,
        "global_step": schedule.global_step,
        "epoch": schedule.epoch,
        "next_dataset_position": schedule.dataset_position + 1,
    }
    checkpoint = capture_checkpoint(metadata=metadata, tensors=_tensor_values(metadata))
    path = tmp_path / "phase-boundary.ckpt"
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
    metadata = _metadata(epoch_boundary=True)
    checkpoint = capture_checkpoint(metadata=metadata, tensors=_tensor_values(metadata))
    path = tmp_path / "epoch-boundary.ckpt"
    save_checkpoint(path, checkpoint)

    resumed_scheduler = FakeResumableScheduler(phase_steps=5)
    loaded_schedule = load_checkpoint(path).schedule
    resumed_scheduler.resume(loaded_schedule, ["first", "second", "third"])

    assert loaded_schedule == CheckpointSchedule(
        active_phase="eggroll",
        completed_phase_steps=3,
        global_step=3,
        epoch=1,
        dataset_position=-1,
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
    save_checkpoint(tmp_path / "epoch-100.ckpt", lower_step)
    save_checkpoint(tmp_path / "phase-10.ckpt", higher_step)

    assert latest_checkpoint(tmp_path) == tmp_path / "phase-10.ckpt"


def test_latest_checkpoint_returns_none_for_missing_directory(tmp_path) -> None:
    assert latest_checkpoint(tmp_path / "missing") is None


def test_latest_checkpoint_returns_none_for_empty_directory(tmp_path) -> None:
    assert latest_checkpoint(tmp_path) is None


def test_coincident_boundaries_serialize_once_and_expose_both_names(
    tmp_path, monkeypatch
) -> None:
    checkpoint = _checkpoint(global_step=12, epoch=3)
    save_calls = 0
    real_save = alternating_checkpoint.save_checkpoint

    def counting_save(*args, **kwargs) -> None:
        nonlocal save_calls
        save_calls += 1
        real_save(*args, **kwargs)

    monkeypatch.setattr(alternating_checkpoint, "save_checkpoint", counting_save)

    paths = save_boundary_checkpoints(
        tmp_path,
        checkpoint,
        phase_boundary=True,
        epoch_boundary=True,
    )

    assert paths == (tmp_path / "phase-12.ckpt", tmp_path / "epoch-3.ckpt")
    assert save_calls == 1
    assert all(path.exists() for path in paths)
    assert paths[0].samefile(paths[1])
    assert load_checkpoint(paths[0]).schedule == checkpoint.schedule
    assert load_checkpoint(paths[1]).schedule == checkpoint.schedule


def _checkpoint(*, global_step: int, epoch: int) -> AlternatingCheckpoint:
    metadata = _metadata()
    metadata["schedule"]["global_step"] = global_step
    metadata["schedule"]["epoch"] = epoch
    return capture_checkpoint(metadata=metadata, tensors=_tensor_values(metadata))


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
        if schedule.dataset_position < 0:
            next_epoch += 1
        elif next_position == len(examples):
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
