from __future__ import annotations

import inspect
import random
from dataclasses import fields

import numpy as np
import pytest
import torch

from tests.eggroll_reference import (
    capture_eggroll_step_snapshot,
    restore_eggroll_step_snapshot,
)
from tests.test_eggroll_step_equivalence import (
    _all_trainable_params,
    _assert_final_snapshots_close,
    _deterministic_fitness_batch,
    _make_trainer,
)
from train import run_alternating, run_eggroll
from train.eggroll_trainer import EggrollTrainer


def _assert_results_close(actual: object, expected: object) -> None:
    for field in fields(actual):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if isinstance(expected_value, float):
            assert actual_value == pytest.approx(expected_value, rel=1e-4, abs=1e-4)
        else:
            assert actual_value == expected_value


# covers: train/eggroll-execution::Optimized execution preserves EGGROLL training semantics::Resume stays deterministic
def test_eggroll_snapshot_resume_matches_uninterrupted_next_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    random.seed(911)
    np.random.seed(912)
    torch.manual_seed(913)

    uninterrupted = _make_trainer()
    uninterrupted.train_fitness_batch(_deterministic_fitness_batch(0))
    checkpoint = capture_eggroll_step_snapshot(
        _all_trainable_params(uninterrupted),
        uninterrupted.optimizer,
        uninterrupted.workspace,
    )

    uninterrupted_result = uninterrupted.train_fitness_batch(
        _deterministic_fitness_batch(3)
    )
    uninterrupted_final = capture_eggroll_step_snapshot(
        _all_trainable_params(uninterrupted),
        uninterrupted.optimizer,
        uninterrupted.workspace,
    )

    resumed = _make_trainer()
    restore_eggroll_step_snapshot(
        checkpoint,
        _all_trainable_params(resumed),
        resumed.optimizer,
        resumed.workspace,
    )
    resumed_result = resumed.train_fitness_batch(_deterministic_fitness_batch(3))
    resumed_final = capture_eggroll_step_snapshot(
        _all_trainable_params(resumed),
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
