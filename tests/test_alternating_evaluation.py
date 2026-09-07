from __future__ import annotations

import random
from dataclasses import dataclass
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eval.stream.generators.asdiv_a import AsdivRecord
from train.alternating_evaluation import (
    HeldOutProblem,
    evaluate_unperturbed,
    load_held_out_problems,
)
from train.alternating_scheduler import (
    EPOCH_BOUNDARY,
    PARTIAL_PHASE_BOUNDARY,
    PHASE_BOUNDARY,
    EvaluationRecord,
    VarianceHysteresisScheduler,
)
from train.training_results import ExperimentPosition


def test_held_out_loader_uses_only_canonical_validation_records() -> None:
    records = [
        AsdivRecord(id="one", split="validation", question="first", target="1"),
        AsdivRecord(id="two", split="validation", question="second", target="2"),
        AsdivRecord(id="three", split="validation", question="third", target="3"),
    ]

    assert load_held_out_problems(records, 2) == [
        HeldOutProblem("first", "1"),
        HeldOutProblem("second", "2"),
    ]


def test_unperturbed_evaluation_averages_metrics_without_changing_model() -> None:
    model = FakeSharedModel()
    position = ExperimentPosition("eggroll", 2, 4, 1, 3, 4)
    before_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    before_workspace = model.workspace.snapshot()

    result = evaluate_unperturbed(
        model,
        [HeldOutProblem("one", "1"), HeldOutProblem("two", "2")],
        position,
        answer_decoder=lambda _: model.answers.pop(0),
    )

    assert result.position is position
    assert result.language_model_loss == pytest.approx(torch.log(torch.tensor(3.0)).item())
    assert result.shared_variance == pytest.approx(1.0)
    assert result.answer_exact_match == pytest.approx(0.5)
    assert all(torch.equal(before, after) for before, after in zip(before_parameters, model.parameters(), strict=True))
    assert torch.equal(model.workspace.snapshot(), before_workspace)
    assert model.encoder.training
    assert model.latent_loop.training
    assert model.latent_loop.model.training


@pytest.mark.parametrize(
    "evaluation_position",
    [
        ExperimentPosition("eggroll", 2, 500, 1, 500, 500),
        ExperimentPosition("eggroll", 2, 1_000, 1, 1_000, 500),
    ],
    ids=["phase", "epoch"],
)
def test_evaluation_isolates_phase_and_epoch_training_state(
    evaluation_position: ExperimentPosition,
) -> None:
    run = FakeAlternatingRun()
    before = run.snapshot()

    result = run.evaluate(evaluation_position)

    assert result.position is evaluation_position
    assert run.snapshot() == before


def test_scheduler_evaluates_once_when_a_variance_window_completes() -> None:
    eggroll = FakeEngine("eggroll")
    evaluator = FakePhaseEvaluator()
    scheduler = VarianceHysteresisScheduler(
        phase_steps=500,
        eggroll_engine=eggroll,
        gradient_engine=FakeEngine("gradient"),
        evaluator=evaluator,
    )

    for example_position in range(1, 501):
        scheduler.train_step("example", epoch=3, example_position=example_position)

    expected_position = ExperimentPosition("eggroll", 1, 500, 3, 500, 500, optimizer_call_count=500)
    assert evaluator.positions == [expected_position]
    assert scheduler.evaluation_results == (
        EvaluationRecord(expected_position, expected_position, frozenset({PHASE_BOUNDARY})),
    )


def test_scheduler_deduplicates_a_window_and_epoch_evaluation_at_the_same_position() -> None:
    evaluator = FakePhaseEvaluator()
    scheduler = VarianceHysteresisScheduler(
        phase_steps=500,
        eggroll_engine=FakeEngine("eggroll"),
        gradient_engine=FakeEngine("gradient"),
        evaluator=evaluator,
    )

    for example_position in range(1, 501):
        scheduler.train_step("example", epoch=1, example_position=example_position)

    position = ExperimentPosition("eggroll", 1, 500, 1, 500, 500, optimizer_call_count=500)
    scheduler.evaluate_epoch_boundary(position)

    assert evaluator.positions == [position]
    assert scheduler.evaluation_results == (
        EvaluationRecord(position, position, frozenset({PHASE_BOUNDARY, EPOCH_BOUNDARY})),
    )


def test_scheduler_evaluates_an_incomplete_final_window_with_a_partial_label() -> None:
    evaluator = FakePhaseEvaluator()
    scheduler = VarianceHysteresisScheduler(
        phase_steps=500,
        eggroll_engine=FakeEngine("eggroll"),
        gradient_engine=FakeEngine("gradient"),
        evaluator=evaluator,
    )

    for example_position in range(1, 301):
        scheduler.train_step("example", epoch=1, example_position=example_position)

    position = ExperimentPosition("eggroll", 1, 300, 1, 300, 300)
    scheduler.evaluate_final_partial_phase(position)

    assert evaluator.positions == [position]
    assert scheduler.evaluation_results == (EvaluationRecord(position, position, frozenset({PARTIAL_PHASE_BOUNDARY})),)


@dataclass
class FakeEngine:
    update_method: str

    def train_step(self, example: object, position: ExperimentPosition) -> object:
        del example
        assert position.update_method == self.update_method
        return SimpleNamespace(shared_variance=0.015)


@dataclass
class FakePhaseEvaluator:
    positions: list[ExperimentPosition]

    def __init__(self) -> None:
        self.positions = []

    def evaluate(self, position: ExperimentPosition) -> ExperimentPosition:
        self.positions.append(position)
        return position


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        return {"input_ids": torch.tensor([[1 if text == "one" else 2]])}

    def apply_chat_template(self, messages: object, **kwargs: object) -> dict[str, torch.Tensor]:
        del messages, kwargs
        return {"input_ids": torch.tensor([[1]])}


class FakeWorkspace:
    def __init__(self) -> None:
        self.slots = torch.tensor([[9.0, 9.0], [9.0, 9.0]])

    def snapshot(self) -> torch.Tensor:
        return self.slots.clone()

    def restore(self, state: torch.Tensor) -> None:
        self.slots = state.clone()

    def write_slots(self, slots: torch.Tensor) -> None:
        self.slots = slots

    def read_slots(self) -> torch.Tensor:
        return self.slots


class FakeEncoder(nn.Module):
    def encode(self, question: str) -> torch.Tensor:
        del question
        return torch.tensor([[0.0, 0.0], [2.0, 2.0]])


class FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.embedding = nn.Embedding(3, 2)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(self, *, inputs_embeds: torch.Tensor, **_: object) -> SimpleNamespace:
        logits = torch.zeros(*inputs_embeds.shape[:-1], 3) * self.scale
        return SimpleNamespace(logits=logits)


class FakeLatentLoop(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeLanguageModel()

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        del ids
        return torch.zeros(1, 2)

    def run(self, workspace: FakeWorkspace, context_embeds: torch.Tensor) -> None:
        del context_embeds
        workspace.write_slots(workspace.read_slots())


class FakeSharedModel:
    def __init__(self) -> None:
        self.workspace = FakeWorkspace()
        self.encoder = FakeEncoder()
        self.latent_loop = FakeLatentLoop()
        self.tokenizer = FakeTokenizer()
        self.answers = ["The answer is 1", "wrong"]

    def parameters(self):
        return self.latent_loop.parameters()


@dataclass
class FakeAlternatingRun:
    """The mutable state held by an alternating training run."""

    model: FakeSharedModel
    eggroll_optimizer: torch.optim.Adam
    gradient_optimizer: torch.optim.Adam
    dataset_position: int
    phase_position: ExperimentPosition

    def __init__(self) -> None:
        self.model = FakeSharedModel()
        parameters = list(self.model.parameters())
        self.eggroll_optimizer = torch.optim.Adam(parameters, lr=0.1)
        self.gradient_optimizer = torch.optim.Adam(parameters, lr=0.01)
        _initialize_optimizer_state(self.eggroll_optimizer, parameters, gradient=1.0)
        _initialize_optimizer_state(self.gradient_optimizer, parameters, gradient=2.0)
        self.dataset_position = 499
        self.phase_position = ExperimentPosition("eggroll", 2, 499, 1, 499, 499)

    def evaluate(self, position: ExperimentPosition):
        return evaluate_unperturbed(
            self.model,
            [HeldOutProblem("one", "1")],
            position,
            answer_decoder=lambda _: "1",
        )

    def snapshot(self) -> tuple[bytes, bytes, bytes, object, bytes, int, ExperimentPosition]:
        return (
            _serialized_state([parameter.detach() for parameter in self.model.parameters()]),
            _serialized_state(self.eggroll_optimizer.state_dict()),
            _serialized_state(self.gradient_optimizer.state_dict()),
            random.getstate(),
            _serialized_state(torch.get_rng_state()),
            self.dataset_position,
            self.phase_position,
        )


def _initialize_optimizer_state(
    optimizer: torch.optim.Adam,
    parameters: list[nn.Parameter],
    *,
    gradient: float,
) -> None:
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, gradient)
    optimizer.step()
    optimizer.zero_grad()


def _serialized_state(state: object) -> bytes:
    buffer = BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()
