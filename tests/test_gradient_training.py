from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
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

from train.trainer import LatentCoreTrainer
from train.training_results import ExperimentPosition, StepResult


class FakeTokenizer:
    def __call__(self, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
        token_id = 1 if text == "question" else 2
        return {"input_ids": torch.tensor([[token_id]])}


class FakeWorkspace:
    def __init__(self) -> None:
        self.slots = torch.zeros(2, 3)

    def write_slots(self, slots: torch.Tensor) -> None:
        self.slots = slots

    def read_slots(self) -> torch.Tensor:
        return self.slots

    def collapse_stats(self) -> dict[str, float]:
        return {"covariance": 0.0}


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.slot_count = 2
        self.projection = nn.Linear(3, 3)
        self.slot_queries = nn.Parameter(torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]))
        self.attn_log_temp = nn.Parameter(torch.tensor(0.0))

    def encode(self, question: str) -> torch.Tensor:
        return self.slot_queries


class FakeLatentLoop(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3)
        self.model = FakeLanguageModel()

    def embed_tokens(self, question_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, 3)

    def run(self, workspace: FakeWorkspace, context_embeds: torch.Tensor) -> torch.Tensor:
        workspace.write_slots(workspace.read_slots() * 2)
        return workspace.read_slots()


class FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Linear(3, 4, bias=False)

    def forward(self, *, inputs_embeds: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.output(inputs_embeds))


class SharedState:
    def __init__(self) -> None:
        self.workspace = FakeWorkspace()
        self.encoder = FakeEncoder()
        self.latent_loop = FakeLatentLoop()

    def parameters(self):
        return iter(
            [
                self.encoder.projection.weight,
                self.encoder.projection.bias,
                self.encoder.slot_queries,
                self.encoder.attn_log_temp,
                self.latent_loop.projection.weight,
                self.latent_loop.projection.bias,
            ]
        )

    def create_optimizer(self, lr: float) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=lr)


def test_gradient_step_uses_shared_state_and_emits_common_objective_metrics(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "train.trainer.AutoTokenizer.from_pretrained", lambda _: FakeTokenizer()
    )
    state = SharedState()
    trainer = LatentCoreTrainer(state=state)
    position = ExperimentPosition("gradient", 3, 12, 2, 4, 1)

    result = trainer.train_step("question", "answer", position)

    assert trainer.state is state
    assert trainer.encoder is state.encoder
    assert trainer.latent_loop is state.latent_loop
    assert isinstance(result, StepResult)
    assert result.position is position
    assert result.shared_variance == pytest.approx(4.0)
    assert result.regularizer_loss > 0.0
    assert result.total_objective == pytest.approx(
        result.language_model_loss + result.regularizer_loss
    )
