from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

sentence_transformers = ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", sentence_transformers)

from train.eggroll_trainer import EggrollTrainer
from train.eggroll_perturbations import MatrixFactors, sample_antithetic_pair
from train.training_results import ExperimentPosition, StepResult
from train.vicreg import post_loop_slot_variance


def test_fitness_uses_canonical_post_loop_slot_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = EggrollTrainer.__new__(EggrollTrainer)
    trainer.variance_weight = 2.0
    trainer.encoder = SimpleNamespace(
        projection=nn.Linear(2, 2),
        slot_queries=nn.Parameter(torch.zeros(2, 2)),
        attn_log_temp=nn.Parameter(torch.tensor(0.0)),
    )
    model = FakeLanguageModel()
    tap_adapter = FakeTapAdapter()
    trainer.latent_loop = SimpleNamespace(
        projection=nn.Linear(2, 2),
        proj_norm=nn.LayerNorm(2),
        layer_norm=nn.LayerNorm(2),
        residual_weight=0.0,
        num_steps=1,
        tap_layer=0,
        model=model,
        tap_adapter=tap_adapter,
    )
    trainer.trainable_params = [
        trainer.encoder.projection.weight,
        trainer.encoder.projection.bias,
        trainer.encoder.slot_queries,
        trainer.encoder.attn_log_temp,
        trainer.latent_loop.projection.weight,
        trainer.latent_loop.projection.bias,
        trainer.latent_loop.proj_norm.weight,
        trainer.latent_loop.proj_norm.bias,
        trainer.latent_loop.layer_norm.weight,
        trainer.latent_loop.layer_norm.bias,
    ]
    perturbations = list(
        sample_antithetic_pair(
            trainer.trainable_params,
            seed=41,
            sigma=0.02,
            rank=1,
        )
    )
    factorized_calls: list[tuple[object, ...]] = []
    from train import eggroll_trainer as trainer_module

    original_factorized_linear = trainer_module.factorized_linear

    def observe_factorized(*args, **kwargs):
        factorized_calls.append(args)
        return original_factorized_linear(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "factorized_linear", observe_factorized)

    observed_slots: list[torch.Tensor] = []

    def observe_variance(slots: torch.Tensor) -> torch.Tensor:
        observed_slots.append(slots)
        return post_loop_slot_variance(slots)

    monkeypatch.setattr("train.eggroll_trainer.post_loop_slot_variance", observe_variance)

    fitness, losses, variance = trainer._forward_fitness_batch(
        token_embeddings=torch.tensor([[[1.0, 0.0]]]),
        context_embeds=torch.zeros(1, 2),
        answer_ids=torch.tensor([[0]]),
        perturbations=perturbations,
        eos_token_id=1,
    )

    assert variance.tolist() == pytest.approx(
        [post_loop_slot_variance(slots).item() for slots in observed_slots]
    )
    assert fitness.tolist() == pytest.approx(
        (-losses + trainer.variance_weight * variance.clamp_min(1e-10).log()).tolist()
    )
    assert model.forward_calls == 1
    assert len(tap_adapter.prepare_calls) == 1
    assert len(tap_adapter.cached_calls) == 2
    assert len(factorized_calls) == 4
    assert factorized_calls[0][0].shape == (1, 2)
    assert all(
        isinstance(direction, MatrixFactors)
        for call in factorized_calls
        for direction in call[2]
    )

    with pytest.raises(
        ValueError,
        match="expected encoder token embeddings shape",
    ):
        trainer._forward_fitness_batch(
            token_embeddings=torch.ones(2, 1, 2),
            context_embeds=torch.zeros(1, 2),
            answer_ids=torch.tensor([[0]]),
            perturbations=perturbations,
            eos_token_id=1,
        )


def test_constructor_accepts_supplied_shared_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SimpleNamespace(
        workspace=object(),
        encoder=SimpleNamespace(slot_count=2),
        latent_loop=SimpleNamespace(model=nn.Identity()),
        tokenizer=object(),
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
    tap_adapter = FakeTapAdapter()
    trainer.latent_loop = SimpleNamespace(
        embed_tokens=lambda _: torch.zeros(1, 2),
        tap_adapter=tap_adapter,
    )
    trainer.tokenizer = FakeTokenizer()
    trainer.device = "cpu"
    trainer.use_amp = False
    trainer.pop_size = 2
    trainer.eval_batch_size = 2
    trainer.trainable_params = [nn.Parameter(torch.zeros(1))]
    trainer.optimizer = torch.optim.SGD(trainer.trainable_params, lr=0.1)
    trainer.sigma = 0.02
    trainer.rank = 1
    fitness_input: list[torch.Tensor] = []
    prefixes: list[object] = []
    candidate_batches: list[list[object]] = []
    inference_flags: list[bool] = []
    update_calls: list[dict[str, object]] = []

    def forward(*args, **kwargs):
        variance = torch.tensor([0.25, 0.75])
        fitness_input.append(variance)
        prefixes.append(kwargs["context_prefix"])
        candidate_batches.append(args[3])
        inference_flags.append(torch.is_inference_mode_enabled())
        return torch.tensor([-3.0, -1.0]), torch.tensor([2.0, 1.0]), variance

    monkeypatch.setattr(trainer, "_forward_fitness_batch", forward)
    monkeypatch.setattr(
        "train.eggroll_trainer.apply_factorized_update",
        lambda parameters, optimizer, **kwargs: update_calls.append(kwargs),
    )
    position = ExperimentPosition("eggroll", 1, 2, 3, 4, 5)

    result = trainer.train_step("question", "0", position)

    assert trainer.state is not None
    assert isinstance(result, StepResult)
    assert result.position is position
    assert result.shared_variance == pytest.approx(fitness_input[0].mean().item())
    assert result.language_model_loss == pytest.approx(1.5)
    assert result.total_objective == pytest.approx(2.0)
    assert result.regularizer_loss == pytest.approx(0.5)
    assert len(tap_adapter.prepare_calls) == 1
    assert prefixes == [tap_adapter.prefix]
    assert inference_flags == [True]
    positive, negative = candidate_batches[0]
    assert positive.sign == 1.0
    assert negative.sign == -1.0
    assert positive.directions is negative.directions
    assert len(update_calls) == 1
    assert update_calls[0]["fitnesses"] == [-3.0, -1.0]
    assert update_calls[0]["base_seed"] >= 0


class FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.embedding = nn.Embedding(2, 2)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(self, *, inputs_embeds: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        self.forward_calls += 1
        logits = torch.zeros(*inputs_embeds.shape[:-1], 2)
        return SimpleNamespace(logits=logits, hidden_states=[inputs_embeds])


class FakeTapAdapter:
    def __init__(self) -> None:
        self.prefix = object()
        self.prepare_calls: list[torch.Tensor] = []
        self.cached_calls: list[tuple[object, torch.Tensor]] = []

    def prepare_prefix(self, context: torch.Tensor) -> object:
        self.prepare_calls.append(context.clone())
        return self.prefix

    def cached_partial(
        self,
        prefix: object,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        self.cached_calls.append((prefix, slots.clone()))
        return slots


class FakeTokenizer:
    eos_token_id = 1

    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        del text, kwargs
        return {"input_ids": torch.tensor([[0]])}

    def apply_chat_template(
        self, messages: object, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        del messages, kwargs
        return {"input_ids": torch.tensor([[0]])}
