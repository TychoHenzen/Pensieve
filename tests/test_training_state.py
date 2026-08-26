from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import nn


sentence_transformers = ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", sentence_transformers)

from train.eggroll_trainer import EggrollTrainer
from train.trainer import LatentCoreTrainer
from train.training_state import (
    EGGROLL_PARAMETER_PATHS,
    GRADIENT_PARAMETER_PATHS,
    TrainingState,
)


class FakeWorkspace:
    def __init__(self, slot_count: int) -> None:
        self.slot_count = slot_count


class FakeEncoder(nn.Module):
    def __init__(
        self, slot_count: int, device: str, sentence_model: object | None = None
    ) -> None:
        super().__init__()
        del device, sentence_model
        self.slot_count = slot_count
        self.projection = nn.Linear(3, 4)
        self.slot_queries = nn.Parameter(torch.randn(slot_count, 4))
        self.attn_log_temp = nn.Parameter(torch.tensor(0.0))


class FakeLatentLoop(nn.Module):
    def __init__(self, num_steps: int, device: str, backbone: object) -> None:
        super().__init__()
        del device
        self.num_steps = num_steps
        self.projection = nn.Linear(4, 4)
        self.proj_norm = nn.LayerNorm(4)
        self.layer_norm = nn.LayerNorm(4)
        self.model = backbone.model


class FakeTokenizer:
    pass


def _patch_training_dependencies(monkeypatch) -> None:
    monkeypatch.setattr("train.training_state.Workspace", FakeWorkspace)
    monkeypatch.setattr("train.training_state.SlotEncoder", FakeEncoder)
    monkeypatch.setattr("train.training_state.LatentLoop", FakeLatentLoop)
    monkeypatch.setattr(
        "train.training_state.load_frozen_qwen_backbone",
        lambda **_: SimpleNamespace(model=nn.Identity(), tokenizer=FakeTokenizer()),
    )


def _optimizer_step(optimizer, params, gradient: float) -> None:
    optimizer.zero_grad()
    for parameter in params:
        parameter.grad = torch.full_like(parameter, gradient)
    optimizer.step()


# covers: train/stage0-training :: Training methods own separate optimizer parameter scopes :: Construct method-specific optimizers
def test_shared_state_uses_explicit_method_parameter_registries(monkeypatch) -> None:
    _patch_training_dependencies(monkeypatch)

    state = TrainingState(slot_count=3, num_steps=1)
    gradient_trainer = LatentCoreTrainer(state=state)
    eggroll_trainer = EggrollTrainer(pop_size=2, state=state)

    assert tuple(state.trainable_params) == GRADIENT_PARAMETER_PATHS
    assert len(state.trainable_params) == 10
    assert tuple(EGGROLL_PARAMETER_PATHS) == (
        "encoder.projection.weight",
        "encoder.slot_queries",
        "latent_loop.projection.weight",
    )
    assert gradient_trainer.state is state
    assert eggroll_trainer.state is state
    assert gradient_trainer.encoder is eggroll_trainer.encoder
    assert gradient_trainer.latent_loop is eggroll_trainer.latent_loop
    assert gradient_trainer.optimizer is not eggroll_trainer.optimizer
    assert gradient_trainer.trainable_params == list(state.parameters())
    assert eggroll_trainer.trainable_params == list(state.eggroll_parameters())
    gradient_optimizer_ids = {
        id(parameter) for parameter in gradient_trainer.optimizer.param_groups[0]["params"]
    }
    eggroll_optimizer_ids = {
        id(parameter) for parameter in eggroll_trainer.optimizer.param_groups[0]["params"]
    }
    gradient_optimizer_paths = tuple(
        name
        for name, parameter in state.trainable_params.items()
        if id(parameter) in gradient_optimizer_ids
    )
    eggroll_optimizer_paths = tuple(
        name
        for name, parameter in state.trainable_params.items()
        if id(parameter) in eggroll_optimizer_ids
    )
    assert gradient_optimizer_paths == GRADIENT_PARAMETER_PATHS
    assert eggroll_optimizer_paths == EGGROLL_PARAMETER_PATHS
    assert type(gradient_trainer.optimizer) is torch.optim.Adam
    assert type(eggroll_trainer.optimizer) is torch.optim.SGD
    assert eggroll_trainer.optimizer.defaults["momentum"] == 0.0
    assert set(eggroll_trainer.optimizer.state) == set()


def test_trainers_create_independent_state_when_none_is_supplied(monkeypatch) -> None:
    _patch_training_dependencies(monkeypatch)

    gradient_trainer = LatentCoreTrainer()
    eggroll_trainer = EggrollTrainer(pop_size=2)

    assert gradient_trainer.state is not eggroll_trainer.state
    assert len(gradient_trainer.trainable_params) == 10
    assert len(eggroll_trainer.trainable_params) == 3


# covers: train/eggroll-execution :: Matrix perturbations remain factorized during candidate evaluation :: Vector perturbation remains compatible
def test_eggroll_keeps_biases_vectors_and_temperature_at_base_values(monkeypatch) -> None:
    _patch_training_dependencies(monkeypatch)

    state = TrainingState(slot_count=3, num_steps=1)
    trainer = EggrollTrainer(pop_size=2, lr=0.1, state=state)
    excluded_paths = tuple(
        path for path in GRADIENT_PARAMETER_PATHS if path not in EGGROLL_PARAMETER_PATHS
    )
    base_values = {
        path: state.trainable_params[path].detach().clone() for path in excluded_paths
    }

    _optimizer_step(trainer.optimizer, trainer.trainable_params, gradient=1.0)
    _optimizer_step(trainer.optimizer, trainer.trainable_params, gradient=2.0)

    assert excluded_paths == (
        "encoder.projection.bias",
        "encoder.attn_log_temp",
        "latent_loop.projection.bias",
        "latent_loop.proj_norm.weight",
        "latent_loop.proj_norm.bias",
        "latent_loop.layer_norm.weight",
        "latent_loop.layer_norm.bias",
    )
    assert all(
        torch.equal(state.trainable_params[path], base_values[path])
        for path in excluded_paths
    )
    assert set(trainer.optimizer.state) == set()


# covers: train/stage0-training :: Training methods own separate optimizer parameter scopes :: Eggroll leaves non-matrix trainables unchanged
def test_eggroll_updates_leave_non_matrix_trainables_bitwise_unchanged(monkeypatch) -> None:
    _patch_training_dependencies(monkeypatch)

    state = TrainingState(slot_count=3, num_steps=1)
    gradient_trainer = LatentCoreTrainer(lr=0.1, state=state)
    eggroll_trainer = EggrollTrainer(pop_size=2, lr=0.1, state=state)
    gradient_params = list(state.parameters())
    eggroll_params = list(state.eggroll_parameters())
    excluded = [
        parameter
        for name, parameter in state.trainable_params.items()
        if name not in EGGROLL_PARAMETER_PATHS
    ]
    _optimizer_step(gradient_trainer.optimizer, gradient_params, gradient=1.0)
    gradient_state_before_eggroll = {
        parameter: gradient_trainer.optimizer.state[parameter]["exp_avg"].clone()
        for parameter in gradient_params
    }
    before = [parameter.detach().clone() for parameter in excluded]

    _optimizer_step(eggroll_trainer.optimizer, eggroll_params, gradient=2.0)
    _optimizer_step(eggroll_trainer.optimizer, eggroll_params, gradient=3.0)

    assert all(
        torch.equal(gradient_trainer.optimizer.state[parameter]["exp_avg"], saved_state)
        for parameter, saved_state in gradient_state_before_eggroll.items()
    )
    assert all(
        torch.equal(parameter, original)
        for parameter, original in zip(excluded, before, strict=True)
    )
    assert set(eggroll_trainer.optimizer.state) == set()


def _parameter_ids(optimizer: torch.optim.Optimizer) -> tuple[int, ...]:
    return tuple(id(parameter) for parameter in optimizer.param_groups[0]["params"])


# covers: train/alternating-cycle :: Continuous model and optimizer state :: Switch from Eggroll to gradient training
def test_switch_from_eggroll_preserves_shared_parameters_and_prior_adam_state(monkeypatch) -> None:
    _patch_training_dependencies(monkeypatch)

    state = TrainingState(slot_count=3, num_steps=1)
    gradient_trainer = LatentCoreTrainer(lr=0.1, state=state)
    eggroll_trainer = EggrollTrainer(pop_size=2, lr=0.1, state=state)
    assert eggroll_trainer.state is gradient_trainer.state is state
    assert eggroll_trainer.latent_loop is gradient_trainer.latent_loop
    gradient_params = list(state.parameters())
    eggroll_params = list(state.eggroll_parameters())
    gradient_optimizer_id = id(gradient_trainer.optimizer)
    eggroll_optimizer_id = id(eggroll_trainer.optimizer)
    gradient_group_ids = _parameter_ids(gradient_trainer.optimizer)
    eggroll_group_ids = _parameter_ids(eggroll_trainer.optimizer)

    _optimizer_step(gradient_trainer.optimizer, gradient_params, gradient=1.0)
    adam_before_eggroll = {
        parameter: gradient_trainer.optimizer.state[parameter]["exp_avg"].clone()
        for parameter in gradient_params
    }
    _optimizer_step(eggroll_trainer.optimizer, eggroll_params, gradient=2.0)
    eggroll_values = [parameter.detach().clone() for parameter in eggroll_params]

    assert all(
        torch.equal(
            gradient_trainer.optimizer.state[parameter]["exp_avg"], saved_state
        )
        for parameter, saved_state in adam_before_eggroll.items()
    )
    assert id(gradient_trainer.optimizer) == gradient_optimizer_id
    assert id(eggroll_trainer.optimizer) == eggroll_optimizer_id
    assert _parameter_ids(gradient_trainer.optimizer) == gradient_group_ids
    assert _parameter_ids(eggroll_trainer.optimizer) == eggroll_group_ids
    assert gradient_group_ids == tuple(id(parameter) for parameter in gradient_params)
    assert eggroll_group_ids == tuple(id(parameter) for parameter in eggroll_params)

    _optimizer_step(gradient_trainer.optimizer, gradient_params, gradient=3.0)

    assert id(gradient_trainer.optimizer) == gradient_optimizer_id
    assert _parameter_ids(gradient_trainer.optimizer) == gradient_group_ids
    assert any(
        not torch.equal(parameter, expected)
        for parameter, expected in zip(eggroll_params, eggroll_values, strict=True)
    )
    assert all(
        torch.allclose(
            gradient_trainer.optimizer.state[parameter]["exp_avg"],
            torch.full_like(parameter, 0.39),
        )
        for parameter in gradient_params
    )
    assert id(eggroll_trainer.optimizer) == eggroll_optimizer_id
    assert _parameter_ids(eggroll_trainer.optimizer) == eggroll_group_ids
    assert set(eggroll_trainer.optimizer.state) == set()


# covers: train/alternating-cycle :: Continuous model and optimizer state :: Return to Eggroll
def test_return_to_eggroll_preserves_shared_matrices_and_inactive_adam_state(monkeypatch) -> None:
    _patch_training_dependencies(monkeypatch)

    state = TrainingState(slot_count=3, num_steps=1)
    gradient_trainer = LatentCoreTrainer(lr=0.1, state=state)
    eggroll_trainer = EggrollTrainer(pop_size=2, lr=0.1, state=state)
    gradient_params = list(state.parameters())
    eggroll_params = list(state.eggroll_parameters())
    gradient_optimizer_id = id(gradient_trainer.optimizer)
    eggroll_optimizer_id = id(eggroll_trainer.optimizer)
    gradient_group_ids = _parameter_ids(gradient_trainer.optimizer)
    eggroll_group_ids = _parameter_ids(eggroll_trainer.optimizer)

    _optimizer_step(eggroll_trainer.optimizer, eggroll_params, gradient=2.0)
    _optimizer_step(gradient_trainer.optimizer, gradient_params, gradient=3.0)

    adam_before_return = {
        parameter: gradient_trainer.optimizer.state[parameter]["exp_avg"].clone()
        for parameter in gradient_params
    }
    gradient_values = [parameter.detach().clone() for parameter in eggroll_params]
    _optimizer_step(eggroll_trainer.optimizer, eggroll_params, gradient=4.0)

    assert id(eggroll_trainer.optimizer) == eggroll_optimizer_id
    assert _parameter_ids(eggroll_trainer.optimizer) == eggroll_group_ids
    assert eggroll_group_ids == tuple(id(parameter) for parameter in eggroll_params)
    assert set(eggroll_trainer.optimizer.state) == set()
    assert id(gradient_trainer.optimizer) == gradient_optimizer_id
    assert _parameter_ids(gradient_trainer.optimizer) == gradient_group_ids
    assert all(
        torch.equal(
            gradient_trainer.optimizer.state[parameter]["exp_avg"], saved_state
        )
        for parameter, saved_state in adam_before_return.items()
    )
    assert all(
        torch.allclose(parameter, expected - 0.4)
        for parameter, expected in zip(eggroll_params, gradient_values, strict=True)
    )
