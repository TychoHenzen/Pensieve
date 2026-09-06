from __future__ import annotations

import random
from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from eval.stream.generators.asdiv_a import AsdivRecord
from tests.eggroll_reference import (
    EggrollStepSnapshot,
    apply_reference_pair_loop_update,
    capture_eggroll_step_snapshot,
    materialized_linear,
    restore_eggroll_step_snapshot,
)
from train import eggroll_trainer as trainer_module
from train.eggroll_trainer import EggrollTrainer
from train.stage0_data import FitnessBatch


class _TinyWorkspace:
    def __init__(self) -> None:
        self.slots = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10

    def snapshot(self) -> torch.Tensor:
        return self.slots.clone()

    def restore(self, state: torch.Tensor) -> None:
        self.slots = state.clone()


class _TinyTokenizer:
    eos_token_id = 5

    def apply_chat_template(self, *_: Any, **__: Any) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[0, 1, 2]])}

    def __call__(self, *_: Any, **__: Any) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[2, 3]])}


class _TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(6, 4)
        self.register_buffer(
            "output_weight",
            torch.arange(24, dtype=torch.float32).reshape(6, 4) / 17 - 0.4,
        )
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(6, 4) / 19 - 0.5)
        self.model = _TinyLanguageModelBody(self)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(self, *, inputs_embeds: torch.Tensor, **_: Any) -> Any:
        return SimpleNamespace(logits=F.linear(inputs_embeds, self.output_weight))

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.output_weight)


class _TinyLanguageModelBody(nn.Module):
    def __init__(self, parent: _TinyLanguageModel) -> None:
        super().__init__()
        object.__setattr__(self, "parent", parent)

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> Any:
        parent = self.parent
        hidden = parent.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(last_hidden_state=hidden)


class _TinyTapAdapter:
    def prepare_prefix(self, context: torch.Tensor) -> torch.Tensor:
        return context.detach().clone()

    def cached_partial(
        self,
        prefix: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        context_term = prefix.mean().to(slots.dtype)
        return slots * 0.73 + torch.tanh(slots) * 0.11 + context_term * 0.07


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4)
        self.slot_queries = nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(2, 4) / 13 - 0.2)
        self.attn_log_temp = nn.Parameter(torch.tensor(-0.17))
        with torch.no_grad():
            self.projection.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 23 - 0.3)
            self.projection.bias.copy_(torch.linspace(-0.2, 0.25, 4))

    def _token_embeddings(self, _: str) -> torch.Tensor:
        return torch.tensor(
            [[[-0.4, 0.2, 0.7], [0.3, -0.5, 0.1], [0.8, 0.4, -0.2]]],
            dtype=torch.float32,
        )


class _TinyLatentLoop(nn.Module):
    def __init__(self, model: _TinyLanguageModel) -> None:
        super().__init__()
        self.model = model
        self.projection = nn.Linear(4, 4)
        self.proj_norm = nn.LayerNorm(4, eps=1e-5)
        self.layer_norm = nn.LayerNorm(4, eps=1e-5)
        self.residual_weight = 0.5
        self.num_steps = 1
        self.tap_adapter = _TinyTapAdapter()
        with torch.no_grad():
            self.projection.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 29 - 0.25)
            self.projection.bias.copy_(torch.linspace(0.15, -0.1, 4))
            self.proj_norm.weight.copy_(torch.tensor([0.8, 0.9, 1.1, 1.2]))
            self.proj_norm.bias.copy_(torch.tensor([-0.1, 0.05, 0.02, 0.08]))
            self.layer_norm.weight.copy_(torch.tensor([1.2, 0.7, 1.0, 0.85]))
            self.layer_norm.bias.copy_(torch.tensor([0.04, -0.06, 0.09, -0.02]))

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(token_ids).squeeze(0)


def _make_trainer() -> EggrollTrainer:
    trainer = EggrollTrainer.__new__(EggrollTrainer)
    model = _TinyLanguageModel()
    trainer.encoder = _TinyEncoder()
    trainer.latent_loop = _TinyLatentLoop(model)
    trainer.workspace = _TinyWorkspace()
    trainer.tokenizer = _TinyTokenizer()
    trainer.device = "cpu"
    trainer.pop_size = 8
    trainer.eval_batch_size = 4
    trainer.sigma = 0.2
    trainer.rank = 2
    trainer.variance_weight = 0.35
    trainer.prompt_alignment_weight = 1.0
    trainer.use_amp = False
    all_trainable_params = [
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
    trainer.trainable_params = [
        trainer.encoder.projection.weight,
        trainer.encoder.slot_queries,
        trainer.latent_loop.projection.weight,
    ]
    trainer.non_matrix_params = [
        parameter
        for parameter in all_trainable_params
        if all(parameter is not matrix_parameter for matrix_parameter in trainer.trainable_params)
    ]
    trainer.optimizer = torch.optim.SGD(trainer.trainable_params, lr=0.009, momentum=0.0)
    return trainer


def _all_trainable_params(trainer: EggrollTrainer) -> list[nn.Parameter]:
    """Return the fixture's full Stage 0 registry in canonical order."""
    return [
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


@dataclass
class _StepTrace:
    questions: list[str]
    fitnesses: list[torch.Tensor]
    losses: list[torch.Tensor]
    variances: list[torch.Tensor]
    normalized: torch.Tensor | None = None


def _normalized(fitnesses: list[float] | torch.Tensor) -> torch.Tensor:
    values = (
        fitnesses.detach().clone()
        if isinstance(fitnesses, torch.Tensor)
        else torch.tensor(fitnesses, dtype=torch.float32)
    )
    return (values - values.mean()) / (values.std(unbiased=False) + 1e-5)


def _deterministic_fitness_batch(start_position: int = 5) -> FitnessBatch:
    return FitnessBatch(
        records=(
            AsdivRecord(
                id="first",
                split="train",
                question="first deterministic question",
                target="1",
            ),
            AsdivRecord(
                id="second",
                split="train",
                question="second deterministic question",
                target="2",
            ),
            AsdivRecord(
                id="third",
                split="train",
                question="third deterministic question",
                target="3",
            ),
        ),
        start_position=start_position,
        next_position=start_position + 3,
    )


def _assert_nested_close(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_close(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_close(actual_item, expected_item)
    else:
        assert actual == expected


def _assert_final_snapshots_close(
    actual: EggrollStepSnapshot,
    expected: EggrollStepSnapshot,
) -> None:
    for actual_parameter, expected_parameter in zip(
        actual.parameter_values,
        expected.parameter_values,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_parameter,
            expected_parameter,
            rtol=1e-4,
            atol=1e-4,
        )
    _assert_nested_close(actual.optimizer_state, expected.optimizer_state)
    torch.testing.assert_close(
        actual.workspace_slots,
        expected.workspace_slots,
        rtol=1e-4,
        atol=1e-4,
    )
    assert actual.python_rng_state == expected.python_rng_state
    _assert_nested_close(actual.numpy_rng_state, expected.numpy_rng_state)
    assert torch.equal(actual.torch_cpu_rng_state, expected.torch_cpu_rng_state)
    assert len(actual.torch_cuda_rng_states) == len(expected.torch_cuda_rng_states)
    for actual_cuda, expected_cuda in zip(
        actual.torch_cuda_rng_states,
        expected.torch_cuda_rng_states,
        strict=True,
    ):
        assert torch.equal(actual_cuda, expected_cuda)


def _assert_non_matrix_tensors_unchanged(
    initial: EggrollStepSnapshot,
    final: EggrollStepSnapshot,
) -> None:
    matrix_indices = {0, 2, 4}
    for index, (initial_parameter, final_parameter) in enumerate(
        zip(initial.parameter_values, final.parameter_values, strict=True)
    ):
        if index not in matrix_indices:
            torch.testing.assert_close(initial_parameter, final_parameter, rtol=0, atol=0)


# covers: train/eggroll-execution::Optimized execution preserves EGGROLL training semantics::One optimized step matches the reference step
def test_complete_optimized_step_matches_materialized_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(404)

    trainer = _make_trainer()
    initial = capture_eggroll_step_snapshot(
        _all_trainable_params(trainer),
        trainer.optimizer,
        trainer.workspace,
    )
    fitness_batch = _deterministic_fitness_batch()
    optimized_trace = _StepTrace([], [], [], [])
    reference_trace = _StepTrace([], [], [], [])
    active_trace = [optimized_trace]
    original_forward = trainer._forward_fitness_batch
    original_evaluate = trainer._evaluate_fitness_record
    original_update = trainer_module.apply_factorized_update

    def record_forward(*args: Any, **kwargs: Any) -> Any:
        fitness, loss, variance = original_forward(*args, **kwargs)
        active_trace[0].fitnesses.append(fitness.detach().clone())
        active_trace[0].losses.append(loss.detach().clone())
        active_trace[0].variances.append(variance.detach().clone())
        return fitness, loss, variance

    trainer._forward_fitness_batch = record_forward  # type: ignore[method-assign]

    def record_evaluate(question: str, *args: Any, **kwargs: Any) -> Any:
        active_trace[0].questions.append(question)
        return original_evaluate(question, *args, **kwargs)

    trainer._evaluate_fitness_record = record_evaluate  # type: ignore[method-assign]

    def optimized_update(*args: Any, **kwargs: Any) -> Any:
        optimized_trace.normalized = _normalized(kwargs["fitnesses"])
        return original_update(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "apply_factorized_update", optimized_update)
    optimized_result = trainer.train_fitness_batch(fitness_batch)
    optimized_final = capture_eggroll_step_snapshot(
        _all_trainable_params(trainer),
        trainer.optimizer,
        trainer.workspace,
    )

    restore_eggroll_step_snapshot(
        initial,
        _all_trainable_params(trainer),
        trainer.optimizer,
        trainer.workspace,
    )
    active_trace[0] = reference_trace
    monkeypatch.setattr(trainer_module, "factorized_linear", materialized_linear)

    def reference_update(*args: Any, **kwargs: Any) -> Any:
        reference_trace.normalized = _normalized(kwargs["fitnesses"])
        return apply_reference_pair_loop_update(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "apply_factorized_update", reference_update)
    reference_result = trainer.train_fitness_batch(fitness_batch)
    reference_final = capture_eggroll_step_snapshot(
        _all_trainable_params(trainer),
        trainer.optimizer,
        trainer.workspace,
    )

    for optimized_batches, reference_batches in (
        (optimized_trace.fitnesses, reference_trace.fitnesses),
        (optimized_trace.losses, reference_trace.losses),
        (optimized_trace.variances, reference_trace.variances),
    ):
        assert len(optimized_batches) == len(reference_batches) == 6
        torch.testing.assert_close(
            torch.cat(optimized_batches),
            torch.cat(reference_batches),
            rtol=1e-4,
            atol=1e-4,
        )
    assert optimized_trace.normalized is not None
    assert reference_trace.normalized is not None
    optimized_per_problem_fitnesses = [
        torch.cat(optimized_trace.fitnesses[index : index + 2]) for index in range(0, len(optimized_trace.fitnesses), 2)
    ]
    reference_per_problem_fitnesses = [
        torch.cat(reference_trace.fitnesses[index : index + 2]) for index in range(0, len(reference_trace.fitnesses), 2)
    ]
    assert len(optimized_per_problem_fitnesses) == len(fitness_batch.records)
    for optimized_fitnesses, reference_fitnesses in zip(
        optimized_per_problem_fitnesses,
        reference_per_problem_fitnesses,
        strict=True,
    ):
        torch.testing.assert_close(
            optimized_fitnesses,
            reference_fitnesses,
            rtol=1e-4,
            atol=1e-4,
        )
    optimized_mean_fitnesses = torch.stack(optimized_per_problem_fitnesses).mean(dim=0)
    reference_mean_fitnesses = torch.stack(reference_per_problem_fitnesses).mean(dim=0)
    torch.testing.assert_close(
        _normalized(optimized_mean_fitnesses),
        optimized_trace.normalized,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        optimized_trace.normalized,
        reference_trace.normalized,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        optimized_mean_fitnesses,
        reference_mean_fitnesses,
        rtol=1e-4,
        atol=1e-4,
    )
    assert torch.unique(torch.cat(optimized_trace.fitnesses)).numel() > 2
    expected_questions = [record.question for record in fitness_batch.records]
    assert optimized_trace.questions == expected_questions
    assert reference_trace.questions == expected_questions

    for field in fields(optimized_result):
        optimized_value = getattr(optimized_result, field.name)
        reference_value = getattr(reference_result, field.name)
        if isinstance(optimized_value, float):
            assert optimized_value == pytest.approx(
                reference_value,
                rel=1e-4,
                abs=1e-4,
            )
        else:
            assert optimized_value == reference_value
    _assert_final_snapshots_close(optimized_final, reference_final)
    assert isinstance(trainer.optimizer, torch.optim.SGD)
    assert trainer.optimizer.state_dict()["state"] == {}
    _assert_non_matrix_tensors_unchanged(initial, optimized_final)
    _assert_non_matrix_tensors_unchanged(initial, reference_final)
