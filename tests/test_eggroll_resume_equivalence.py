from __future__ import annotations

import copy
import inspect
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.eggroll_reference import (
    capture_eggroll_step_snapshot,
)
from tests.test_eggroll_step_equivalence import (
    _all_trainable_params,
    _deterministic_fitness_batch,
    _make_trainer,
)
from tests.test_stage0_checkpoint_resume import RUN_CONFIG, _metadata
from train import eggroll_trainer as trainer_module
from train import run_alternating, run_eggroll
from train.alternating_checkpoint import (
    CheckpointSchedule,
    build_alternating_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from train.eggroll_trainer import EggrollTrainer
from train.stage0_checkpoint import ALLOWED_MODEL_PARAMETER_PATHS


# covers: train/eggroll-execution :: Optimized execution preserves EGGROLL training semantics :: Resume stays deterministic
def test_eggroll_snapshot_resume_matches_uninterrupted_next_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    random.seed(911)
    np.random.seed(912)
    torch.manual_seed(913)

    uninterrupted = _make_trainer()
    uninterrupted.train_fitness_batch(_deterministic_fitness_batch(0))
    uninterrupted_parameters = dict(
        zip(
            ALLOWED_MODEL_PARAMETER_PATHS,
            _all_trainable_params(uninterrupted),
            strict=True,
        )
    )
    fixture = _metadata()
    run_config = copy.deepcopy(RUN_CONFIG)
    run_config["phase_steps"] = 3
    run_config["eggroll_optimizer"]["learning_rate"] = 0.009
    run_config["eggroll_population"]["fitness_batch_size"] = 3
    gradient = torch.optim.Adam(uninterrupted_parameters.values(), lr=1e-4)
    checkpoint = build_alternating_checkpoint(
        identity=fixture["identity"],
        selections=fixture["selections"],
        model_state=uninterrupted_parameters,
        eggroll_optimizer=uninterrupted.optimizer,
        gradient_optimizer=gradient,
        schedule=CheckpointSchedule(
            "eggroll",
            0,
            3,
            1,
            2,
            eggroll_optimizer_calls=1,
        ),
        phase_steps=3,
        next_dataset_position=0,
        metrics={"loss": 1.0},
        run_config=run_config,
    )
    checkpoint_path = tmp_path / "epoch-1.ckpt"
    save_checkpoint(checkpoint_path, checkpoint)
    loaded = load_checkpoint(checkpoint_path)

    real_sampler = trainer_module.sample_antithetic_pair
    active_seed_trace: list[list[int]] = [[]]

    def record_seed(*args: object, seed: int, **kwargs: object):
        active_seed_trace[0].append(seed)
        return real_sampler(*args, seed=seed, **kwargs)

    monkeypatch.setattr(trainer_module, "sample_antithetic_pair", record_seed)

    uninterrupted_result = uninterrupted.train_fitness_batch(_deterministic_fitness_batch(3))
    uninterrupted_seeds = list(active_seed_trace[0])
    uninterrupted_final = capture_eggroll_step_snapshot(
        _all_trainable_params(uninterrupted),
        uninterrupted.optimizer,
        uninterrupted.workspace,
    )

    resumed = _make_trainer()
    resumed_parameters = dict(
        zip(
            ALLOWED_MODEL_PARAMETER_PATHS,
            _all_trainable_params(resumed),
            strict=True,
        )
    )
    resumed_gradient = torch.optim.Adam(resumed_parameters.values(), lr=1e-4)
    run_alternating._restore_checkpoint_state(
        loaded,
        SimpleNamespace(trainable_params=resumed_parameters),
        resumed_gradient,
        resumed.optimizer,
    )
    active_seed_trace[0] = []
    resumed_result = resumed.train_fitness_batch(_deterministic_fitness_batch(3))
    resumed_seeds = list(active_seed_trace[0])
    resumed_final = capture_eggroll_step_snapshot(
        _all_trainable_params(resumed),
        resumed.optimizer,
        resumed.workspace,
    )

    assert resumed_result.position == uninterrupted_result.position
    assert resumed_result.language_model_loss == pytest.approx(
        uninterrupted_result.language_model_loss, rel=1e-4, abs=1e-4
    )
    assert resumed_result.total_objective == pytest.approx(uninterrupted_result.total_objective, rel=1e-4, abs=1e-4)
    assert resumed_result.regularizer_loss == pytest.approx(uninterrupted_result.regularizer_loss, rel=1e-4, abs=1e-4)
    assert resumed_result.shared_variance == pytest.approx(uninterrupted_result.shared_variance, rel=1e-4, abs=1e-4)
    assert resumed_result.consumed_record_count == uninterrupted_result.consumed_record_count
    assert resumed_result.next_example_position == uninterrupted_result.next_example_position
    assert resumed_result.optimizer_call_count == uninterrupted_result.optimizer_call_count
    assert resumed_seeds == uninterrupted_seeds
    for resumed_parameter, uninterrupted_parameter in zip(
        resumed_final.parameter_values,
        uninterrupted_final.parameter_values,
        strict=True,
    ):
        torch.testing.assert_close(
            resumed_parameter,
            uninterrupted_parameter,
            rtol=1e-4,
            atol=1e-4,
        )
    assert resumed_final.optimizer_state == uninterrupted_final.optimizer_state
    assert resumed_final.python_rng_state == uninterrupted_final.python_rng_state
    assert np.array_equal(resumed_final.numpy_rng_state[1], uninterrupted_final.numpy_rng_state[1])
    assert resumed_final.numpy_rng_state[:1] == uninterrupted_final.numpy_rng_state[:1]
    assert resumed_final.numpy_rng_state[2:] == uninterrupted_final.numpy_rng_state[2:]
    assert torch.equal(
        resumed_final.torch_cpu_rng_state,
        uninterrupted_final.torch_cpu_rng_state,
    )
    assert resumed_final.torch_cuda_rng_states == uninterrupted_final.torch_cuda_rng_states
    assert all(
        torch.equal(loaded.tensors[f"model.{name}"], checkpoint.tensors[f"model.{name}"])
        for name in ALLOWED_MODEL_PARAMETER_PATHS
    )


# covers: train/eggroll-execution::Optimized execution preserves EGGROLL training semantics::Existing commands select the optimized path
def test_both_training_commands_bind_only_the_optimized_eggroll_trainer() -> None:
    assert run_eggroll.EggrollTrainer is EggrollTrainer
    assert run_alternating.EggrollTrainer is EggrollTrainer

    for command_module in (run_eggroll, run_alternating):
        source = inspect.getsource(command_module)
        assert "tests.eggroll_reference" not in source
        assert "materialized_linear" not in source
        assert "apply_reference_pair_loop_update" not in source
