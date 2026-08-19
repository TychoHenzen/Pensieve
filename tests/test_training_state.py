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


def test_shared_state_registry_is_used_by_both_trainers(monkeypatch) -> None:
    monkeypatch.setattr("train.training_state.Workspace", FakeWorkspace)
    monkeypatch.setattr("train.training_state.SlotEncoder", FakeEncoder)
    monkeypatch.setattr("train.training_state.LatentLoop", FakeLatentLoop)
    monkeypatch.setattr(
        "train.trainer.AutoTokenizer.from_pretrained", lambda _: FakeTokenizer()
    )
    monkeypatch.setattr(
        "train.eggroll_trainer.AutoTokenizer.from_pretrained", lambda _: FakeTokenizer()
    )

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
    monkeypatch.setattr("train.training_state.Workspace", FakeWorkspace)
    monkeypatch.setattr("train.training_state.SlotEncoder", FakeEncoder)
    monkeypatch.setattr("train.training_state.LatentLoop", FakeLatentLoop)
    monkeypatch.setattr(
        "train.trainer.AutoTokenizer.from_pretrained", lambda _: FakeTokenizer()
    )
    monkeypatch.setattr(
        "train.eggroll_trainer.AutoTokenizer.from_pretrained", lambda _: FakeTokenizer()
    )

    gradient_trainer = LatentCoreTrainer()
    eggroll_trainer = EggrollTrainer(pop_size=2)

    assert gradient_trainer.state is not eggroll_trainer.state
    assert len(gradient_trainer.trainable_params) == 10
    assert len(eggroll_trainer.trainable_params) == 10
