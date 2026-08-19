from __future__ import annotations

import sys
from types import ModuleType

import torch
from torch import nn


class StubAutoTokenizer:
    @staticmethod
    def from_pretrained(name: str) -> None:
        return None


transformers = ModuleType("transformers")
transformers.AutoTokenizer = StubAutoTokenizer
transformers.AutoModelForCausalLM = object
sys.modules.setdefault("transformers", transformers)

sentence_transformers = ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", sentence_transformers)

from train.eggroll_trainer import EggrollTrainer
from train.trainer import LatentCoreTrainer
from train.training_state import TrainingState


class FakeWorkspace:
    def __init__(self, slot_count: int) -> None:
        self.slot_count = slot_count


class FakeEncoder(nn.Module):
    def __init__(self, slot_count: int, device: str) -> None:
        super().__init__()
        self.slot_count = slot_count
        self.projection = nn.Linear(3, 4)
        self.slot_queries = nn.Parameter(torch.randn(slot_count, 4))
        self.attn_log_temp = nn.Parameter(torch.tensor(0.0))


class FakeLatentLoop(nn.Module):
    def __init__(self, num_steps: int, device: str) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.projection = nn.Linear(4, 4)
        self.proj_norm = nn.LayerNorm(4)
        self.layer_norm = nn.LayerNorm(4)
        self.model = nn.Identity()


class FakeTokenizer:
    pass


def _patch_training_dependencies(monkeypatch) -> None:
    monkeypatch.setattr("train.training_state.Workspace", FakeWorkspace)
    monkeypatch.setattr("train.training_state.SlotEncoder", FakeEncoder)
    monkeypatch.setattr("train.training_state.LatentLoop", FakeLatentLoop)
    monkeypatch.setattr(
        "train.trainer.AutoTokenizer.from_pretrained", lambda _: FakeTokenizer()
    )
    monkeypatch.setattr(
        "train.eggroll_trainer.AutoTokenizer.from_pretrained", lambda _: FakeTokenizer()
    )


def _optimizer_step(optimizer, params, gradient: float) -> None:
    optimizer.zero_grad()
    for parameter in params:
        parameter.grad = torch.full_like(parameter, gradient)
    optimizer.step()


def test_shared_state_registry_is_used_by_both_trainers(monkeypatch) -> None:
    _patch_training_dependencies(monkeypatch)

    state = TrainingState(slot_count=3, num_steps=1)
    gradient_trainer = LatentCoreTrainer(state=state)
    eggroll_trainer = EggrollTrainer(pop_size=2, state=state)

    expected_names = (
        "encoder.projection.weight",
        "encoder.projection.bias",
        "encoder.slot_queries",
        "encoder.attn_log_temp",
        "latent_loop.projection.weight",
        "latent_loop.projection.bias",
        "latent_loop.proj_norm.weight",
        "latent_loop.proj_norm.bias",
        "latent_loop.layer_norm.weight",
        "latent_loop.layer_norm.bias",
    )
    assert tuple(state.trainable_params) == expected_names
    assert len(state.trainable_params) == 10
    assert gradient_trainer.state is state
    assert eggroll_trainer.state is state
    assert gradient_trainer.encoder is eggroll_trainer.encoder
    assert gradient_trainer.latent_loop is eggroll_trainer.latent_loop
    assert gradient_trainer.optimizer is not eggroll_trainer.optimizer
    assert all(
        gradient_param is eggroll_param is state_param
        for gradient_param, eggroll_param, state_param in zip(
            gradient_trainer.trainable_params,
            eggroll_trainer.trainable_params,
            state.parameters(),
            strict=True,
        )
    )


def test_trainers_create_independent_state_when_none_is_supplied(monkeypatch) -> None:
    _patch_training_dependencies(monkeypatch)

    gradient_trainer = LatentCoreTrainer()
    eggroll_trainer = EggrollTrainer(pop_size=2)

    assert gradient_trainer.state is not eggroll_trainer.state
    assert len(gradient_trainer.trainable_params) == 10
    assert len(eggroll_trainer.trainable_params) == 10


def test_optimizer_state_survives_both_shared_model_phase_transitions(monkeypatch) -> None:
    _patch_training_dependencies(monkeypatch)

    state = TrainingState(slot_count=3, num_steps=1)
    gradient_trainer = LatentCoreTrainer(lr=0.1, state=state)
    eggroll_trainer = EggrollTrainer(pop_size=2, lr=0.1, state=state)
    params = list(state.parameters())
    param_ids = tuple(id(parameter) for parameter in params)

    _optimizer_step(gradient_trainer.optimizer, params, gradient=1.0)
    gradient_state_before_eggroll = {
        parameter: gradient_trainer.optimizer.state[parameter]["exp_avg"].clone()
        for parameter in params
    }

    _optimizer_step(eggroll_trainer.optimizer, params, gradient=2.0)
    eggroll_output = tuple(parameter.detach().clone() for parameter in params)

    assert tuple(id(parameter) for parameter in state.parameters()) == param_ids
    assert all(
        torch.equal(gradient_trainer.optimizer.state[parameter]["exp_avg"], saved_state)
        for parameter, saved_state in gradient_state_before_eggroll.items()
    )
    assert all(
        torch.equal(parameter, produced_value)
        for parameter, produced_value in zip(params, eggroll_output, strict=True)
    )

    eggroll_state_before_gradient = {
        parameter: eggroll_trainer.optimizer.state[parameter]["exp_avg"].clone()
        for parameter in params
    }
    _optimizer_step(gradient_trainer.optimizer, params, gradient=3.0)
    gradient_output = tuple(parameter.detach().clone() for parameter in params)

    assert tuple(id(parameter) for parameter in state.parameters()) == param_ids
    assert all(
        torch.equal(eggroll_trainer.optimizer.state[parameter]["exp_avg"], saved_state)
        for parameter, saved_state in eggroll_state_before_gradient.items()
    )
    assert all(
        torch.equal(parameter, produced_value)
        for parameter, produced_value in zip(params, gradient_output, strict=True)
    )
