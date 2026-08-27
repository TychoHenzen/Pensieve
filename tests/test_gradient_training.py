from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

sentence_transformers = ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", sentence_transformers)

from train.trainer import LatentCoreTrainer
from train.training_results import ExperimentPosition, StepResult


class FakeTokenizer:
    eos_token_id = 3

    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        token_id = 2 if text == "2" else 1
        return {"input_ids": torch.tensor([[token_id]])}

    def apply_chat_template(
        self, messages: object, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        del messages, kwargs
        return {"input_ids": torch.tensor([[1]])}


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
        self.slot_queries = nn.Parameter(
            torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
        )
        self.attn_log_temp = nn.Parameter(torch.tensor(0.0))

    def encode(self, question: str) -> torch.Tensor:
        return self.slot_queries


class FakeLatentLoop(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3)
        self.model = FakeLanguageModel()
        self.run_calls = 0

    def embed_tokens(self, question_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, 3)

    def run(self, workspace: FakeWorkspace, context_embeds: torch.Tensor) -> torch.Tensor:
        self.run_calls += 1
        workspace.write_slots(workspace.read_slots() * 2)
        return workspace.read_slots()


class FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(4, 3)
        self.output = nn.Linear(3, 4, bias=False)
        self.model = FakeLanguageModelBody(self.embedding)
        self.lm_head = self.output

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(self, *, inputs_embeds: torch.Tensor, **_: object) -> SimpleNamespace:
        return SimpleNamespace(logits=self.output(inputs_embeds))


class FakeLanguageModelBody(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ) -> SimpleNamespace:
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(last_hidden_state=hidden)


class SharedState:
    def __init__(self) -> None:
        self.workspace = FakeWorkspace()
        self.encoder = FakeEncoder()
        self.latent_loop = FakeLatentLoop()
        self.tokenizer = FakeTokenizer()

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

    def create_gradient_optimizer(self, lr: float) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=lr)


def test_gradient_step_uses_shared_state_and_emits_common_objective_metrics(
    monkeypatch,
) -> None:
    state = SharedState()
    trainer = LatentCoreTrainer(state=state, variance_weight=0.5)
    position = ExperimentPosition("gradient", 3, 12, 2, 4, 1)

    result = trainer.train_step("question", "2", position)

    assert trainer.state is state
    assert trainer.encoder is state.encoder
    assert trainer.latent_loop is state.latent_loop
    assert trainer.variance_weight == 0.5
    assert isinstance(result, StepResult)
    assert result.position is position
    assert state.latent_loop.run_calls == 2
    assert result.shared_variance == pytest.approx(0.04)
    assert result.regularizer_loss > 0.0
    assert result.total_objective == pytest.approx(
        result.language_model_loss + result.regularizer_loss
    )
