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

from train.eggroll_trainer import EggrollTrainer
from train.training_results import ExperimentPosition, StepResult
from train.vicreg import post_loop_slot_variance


def test_fitness_uses_canonical_post_loop_slot_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = EggrollTrainer.__new__(EggrollTrainer)
    trainer.variance_weight = 2.0
    trainer.trainable_params = [nn.Parameter(torch.zeros(1)) for _ in range(10)]
    trainer.encoder = SimpleNamespace(
        projection=nn.Linear(2, 2),
        slot_queries=nn.Parameter(torch.zeros(2, 2)),
        attn_log_temp=nn.Parameter(torch.tensor(0.0)),
    )
    model = FakeLanguageModel()
    trainer.latent_loop = SimpleNamespace(
        projection=nn.Linear(2, 2),
        proj_norm=nn.LayerNorm(2),
        layer_norm=nn.LayerNorm(2),
        residual_weight=0.0,
        num_steps=0,
        model=model,
    )

    observed_slots: list[torch.Tensor] = []

    def observe_variance(slots: torch.Tensor) -> torch.Tensor:
        observed_slots.append(slots)
        return post_loop_slot_variance(slots)

    monkeypatch.setattr("train.eggroll_trainer.post_loop_slot_variance", observe_variance)

    fitness, losses, variance = trainer._forward_fitness_batch(
        token_embeddings=torch.tensor([[1.0, 0.0]]),
        context_embeds=torch.zeros(1, 2),
        answer_ids=torch.tensor([[0]]),
        perturbations=[[torch.zeros_like(parameter) for parameter in trainer.trainable_params]],
    )

    assert variance.tolist() == pytest.approx(
        [post_loop_slot_variance(slots).item() for slots in observed_slots]
    )
    assert fitness.tolist() == pytest.approx(
        (-losses + trainer.variance_weight * variance.clamp_min(1e-10).log()).tolist()
    )


def test_constructor_accepts_supplied_shared_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "train.eggroll_trainer.AutoTokenizer.from_pretrained", lambda _: object()
    )
    state = SimpleNamespace(
        workspace=object(),
        encoder=SimpleNamespace(slot_count=2),
        latent_loop=SimpleNamespace(model=nn.Identity()),
        parameters=lambda: iter(()),
        create_optimizer=lambda _: object(),
    )

    trainer = EggrollTrainer(pop_size=2, state=state)

    assert trainer.state is state
    assert trainer.workspace is state.workspace
    assert trainer.encoder is state.encoder
    assert trainer.latent_loop is state.latent_loop


def test_train_step_accepts_shared_state_and_reports_common_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = EggrollTrainer.__new__(EggrollTrainer)
    trainer.state = object()
    trainer.encoder = SimpleNamespace(_token_embeddings=lambda _: torch.zeros(1, 2))
    trainer.latent_loop = SimpleNamespace(embed_tokens=lambda _: torch.zeros(1, 2))
    trainer.tokenizer = lambda text, return_tensors: {"input_ids": torch.tensor([[0]])}
    trainer.device = "cpu"
    trainer.use_amp = False
    trainer.pop_size = 2
    trainer.eval_batch_size = 2
    trainer.trainable_params = [nn.Parameter(torch.zeros(1))]
    trainer.optimizer = torch.optim.SGD(trainer.trainable_params, lr=0.1)
    trainer._generate_perturbation = lambda seed, sign: [torch.zeros(1)]
    fitness_input: list[torch.Tensor] = []

    def forward(*args, **kwargs):
        variance = torch.tensor([0.25, 0.75])
        fitness_input.append(variance)
        return torch.tensor([-3.0, -1.0]), torch.tensor([2.0, 1.0]), variance

    monkeypatch.setattr(trainer, "_forward_fitness_batch", forward)
    position = ExperimentPosition("eggroll", 1, 2, 3, 4, 5)

    result = trainer.train_step("question", "answer", position)

    assert trainer.state is not None
    assert isinstance(result, StepResult)
    assert result.position is position
    assert result.shared_variance == pytest.approx(fitness_input[0].mean().item())
    assert result.language_model_loss == pytest.approx(1.5)
    assert result.total_objective == pytest.approx(2.0)
    assert result.regularizer_loss == pytest.approx(0.5)


class FakeLanguageModel(nn.Module):
    def forward(self, *, inputs_embeds: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        logits = torch.zeros(*inputs_embeds.shape[:-1], 2)
        return SimpleNamespace(logits=logits, hidden_states=[inputs_embeds])
