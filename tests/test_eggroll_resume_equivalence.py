from __future__ import annotations

import inspect
import random
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.eggroll_reference import capture_eggroll_step_snapshot
from tests.test_eggroll_step_equivalence import (
    _assert_final_snapshots_close,
    _make_trainer,
)
from tests.test_stage0_checkpoint_resume import RUN_CONFIG, _metadata
from train import run_alternating, run_eggroll
from train.alternating_checkpoint import (
    CheckpointSchedule,
    build_alternating_checkpoint,
    load_checkpoint,
    save_checkpoint,
    stage0_parameter_paths,
)
from train.eggroll_trainer import EggrollTrainer


def _named_parameters(trainer: EggrollTrainer) -> dict[str, torch.nn.Parameter]:
    return dict(zip(stage0_parameter_paths(), trainer.trainable_params, strict=True))


def _assert_results_close(actual: object, expected: object) -> None:
    for field in fields(actual):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if isinstance(expected_value, float):
            assert actual_value == pytest.approx(expected_value, rel=1e-4, abs=1e-4)
        else:
            assert actual_value == expected_value


# covers: train/eggroll-execution::Optimized execution preserves EGGROLL training semantics::Resume stays deterministic
def test_alternating_checkpoint_resume_matches_uninterrupted_next_eggroll_step(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    random.seed(911)
    np.random.seed(912)
    torch.manual_seed(913)

    uninterrupted = _make_trainer()
    uninterrupted_gradient = torch.optim.Adam(
        uninterrupted.trainable_params,
        lr=1e-4,
    )
    uninterrupted.train_step("boundary question", "1")
    fixture = _metadata()
    checkpoint = build_alternating_checkpoint(
        identity=fixture["identity"],
        selections=fixture["selections"],
        model_state=_named_parameters(uninterrupted),
        eggroll_optimizer=uninterrupted.optimizer,
        gradient_optimizer=uninterrupted_gradient,
        schedule=CheckpointSchedule("eggroll", 1, 1, 1, 0),
        phase_steps=2,
        next_dataset_position=1,
        metrics={"language_model_loss": 1.0, "shared_variance": 0.5},
        run_config=RUN_CONFIG,
    )
    checkpoint_path = tmp_path / "phase-1.ckpt"
    save_checkpoint(checkpoint_path, checkpoint)
    loaded = load_checkpoint(checkpoint_path)

    uninterrupted_result = uninterrupted.train_step("next question", "1")
    uninterrupted_final = capture_eggroll_step_snapshot(
        uninterrupted.trainable_params,
        uninterrupted.optimizer,
        uninterrupted.workspace,
    )

    resumed = _make_trainer()
    resumed_gradient = torch.optim.Adam(resumed.trainable_params, lr=7e-4)
    run_alternating._restore_checkpoint_state(
        loaded,
        SimpleNamespace(trainable_params=_named_parameters(resumed)),
        resumed_gradient,
        resumed.optimizer,
    )
    resumed_result = resumed.train_step("next question", "1")
    resumed_final = capture_eggroll_step_snapshot(
        resumed.trainable_params,
        resumed.optimizer,
        resumed.workspace,
    )

    _assert_results_close(resumed_result, uninterrupted_result)
    _assert_final_snapshots_close(resumed_final, uninterrupted_final)


# covers: train/eggroll-execution::Optimized execution preserves EGGROLL training semantics::Existing commands select the optimized path
def test_both_training_commands_bind_only_the_optimized_eggroll_trainer() -> None:
    assert run_eggroll.EggrollTrainer is EggrollTrainer
    assert run_alternating.EggrollTrainer is EggrollTrainer

    for command_module in (run_eggroll, run_alternating):
        source = inspect.getsource(command_module)
        assert "tests.eggroll_reference" not in source
        assert "materialized_linear" not in source
        assert "apply_reference_pair_loop_update" not in source
