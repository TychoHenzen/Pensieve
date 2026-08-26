from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eval.stage0_identity import qwen_messages
from train import answer_objective
from train.alternating_evaluation import HeldOutProblem, evaluate_unperturbed
from train.training_results import ExperimentPosition


class SequenceTokenizer:
    eos_token_id = 6

    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        ids = [9] if text == "question" else [4, 5]
        return {"input_ids": torch.tensor([ids])}

    def apply_chat_template(
        self, messages: object, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        del messages, kwargs
        return {"input_ids": torch.tensor([[9]])}


class RecordingLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.embedding = nn.Embedding(10, 2)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor([[float(i), -float(i)] for i in range(10)])
            )
        self.seen_inputs: list[torch.Tensor] = []

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, *, inputs_embeds: torch.Tensor, **_: object) -> SimpleNamespace:
        self.seen_inputs.append(inputs_embeds.detach().clone())
        logits = torch.full((*inputs_embeds.shape[:-1], 10), -100.0)
        if inputs_embeds.shape[1] >= 4:
            logits[:, 1, 4] = 100.0
            logits[:, 2, 5] = 100.0
            logits[:, 3, 6] = 100.0
        return SimpleNamespace(logits=logits + self.anchor * 0.0)


class SequenceWorkspace:
    def __init__(self) -> None:
        self.slots = torch.tensor([[0.25, 0.75], [0.5, 1.5]])

    def snapshot(self) -> torch.Tensor:
        return self.slots.clone()

    def restore(self, state: torch.Tensor) -> None:
        self.slots = state.clone()

    def write_slots(self, slots: torch.Tensor) -> None:
        self.slots = slots

    def read_slots(self) -> torch.Tensor:
        return self.slots


class SequenceEncoder(nn.Module):
    def encode(self, question: str) -> torch.Tensor:
        del question
        return torch.tensor([[0.25, 0.75], [0.5, 1.5]])


class SequenceLatentLoop(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = RecordingLanguageModel()
        self.run_calls = 0

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        del ids
        return torch.zeros(1, 2)

    def run(self, workspace: SequenceWorkspace, context_embeds: torch.Tensor) -> None:
        self.run_calls += 1
        del workspace, context_embeds


class SequenceSharedModel:
    def __init__(self) -> None:
        self.workspace = SequenceWorkspace()
        self.encoder = SequenceEncoder()
        self.latent_loop = SequenceLatentLoop()
        self.tokenizer = SequenceTokenizer()


def test_held_out_loss_teacher_forces_decoder_sequence_through_eos() -> None:
    shared_model = SequenceSharedModel()
    slots = shared_model.workspace.read_slots().clone()

    result = evaluate_unperturbed(
        shared_model,
        [HeldOutProblem("question", "45")],
        ExperimentPosition("gradient", 0, 1, 1, 1, 1),
        answer_decoder=lambda _: "45",
    )

    expected_prefixes = shared_model.latent_loop.model.embedding(
        torch.tensor([4, 5])
    )
    seen_input = shared_model.latent_loop.model.seen_inputs[-1].squeeze(0)
    assert torch.equal(seen_input[:2], slots)
    assert torch.equal(seen_input[2:], expected_prefixes)
    assert shared_model.latent_loop.run_calls == 2
    assert result.language_model_loss == pytest.approx(0.0)


class Stage0ContractTokenizer:
    eos_token_id = 63

    def __init__(self) -> None:
        self.target_calls: list[tuple[str, dict[str, object]]] = []
        self.chat_calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        self.target_calls.append((text, kwargs))
        ids_by_target = {
            "1": [10],
            "6.5": [20, 21],
            "1/2": [30, 31],
            "0": [40],
        }
        return {"input_ids": torch.tensor([ids_by_target[text]])}

    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> dict[str, torch.Tensor]:
        self.chat_calls.append((messages, kwargs))
        return {"input_ids": torch.tensor([[50, 51, 52]])}


class Stage0EmbeddingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, 2)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor([[float(index), -float(index)] for index in range(64)])
            )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


class PromptAlignmentBody(nn.Module):
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


class PromptAlignmentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(4, 2)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [-1.0, 0.0],
                        [0.0, -1.0],
                    ]
                )
            )
        self.model = PromptAlignmentBody(self.embedding)
        self.lm_head = nn.Linear(2, 4, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


# covers: train/stage0-training :: Decoder-aligned numerical objective :: Question-conditioned output alignment
def test_prompt_alignment_penalizes_cross_problem_output_collapse() -> None:
    model = PromptAlignmentModel()
    answer_ids = torch.tensor([2])
    slots = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    first_teacher = answer_objective.prompt_teacher_state(
        model, torch.tensor([[0]])
    )
    second_teacher = answer_objective.prompt_teacher_state(
        model, torch.tensor([[1]])
    )
    first = answer_objective.prompt_aligned_answer_objective(
        model,
        slots,
        answer_ids,
        eos_token_id=3,
        teacher_state=first_teacher,
    )
    second = answer_objective.prompt_aligned_answer_objective(
        model,
        slots,
        answer_ids,
        eos_token_id=3,
        teacher_state=second_teacher,
    )

    assert not first_teacher.requires_grad
    assert not second_teacher.requires_grad
    assert not torch.equal(first_teacher, second_teacher)
    assert first.prompt_alignment_loss.item() == pytest.approx(0.0)
    assert second.prompt_alignment_loss.item() == pytest.approx(1.0)


@pytest.mark.parametrize("weight", [-0.1, float("inf"), float("nan")])
def test_prompt_alignment_weight_must_be_finite_and_non_negative(
    weight: float,
) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        answer_objective.validate_prompt_alignment_weight(weight)


@pytest.mark.parametrize(
    ("source_target", "canonical_target", "answer_ids"),
    [
        ("+001", "1", [10]),
        ("6.500", "6.5", [20, 21]),
        ("2/4", "1/2", [30, 31]),
        ("-0.000", "0", [40]),
    ],
)
# covers: train/stage0-training :: Decoder-aligned numerical objective :: Multi-token numerical target
def test_numerical_target_layout_contains_every_answer_token_then_eos(
    source_target: str,
    canonical_target: str,
    answer_ids: list[int],
) -> None:
    tokenizer = Stage0ContractTokenizer()
    model = Stage0EmbeddingModel()
    slots = torch.tensor([[0.25, 0.75], [0.5, 1.5], [1.0, 2.0]])

    prepared = answer_objective.prepare_training_example(
        tokenizer, "How many remain?", source_target
    )
    input_embeds, labels = answer_objective.decoder_aligned_inputs_and_labels(
        model,
        slots,
        prepared.answer_ids,
        tokenizer.eos_token_id,
    )

    expected_answer_ids = torch.tensor(answer_ids)
    expected_answer_embeds = model.embedding(expected_answer_ids)
    assert prepared.canonical_target == canonical_target
    assert torch.equal(prepared.answer_ids, expected_answer_ids)
    assert tokenizer.target_calls == [
        (canonical_target, {"add_special_tokens": False, "return_tensors": "pt"})
    ]
    assert torch.equal(input_embeds.squeeze(0)[: len(slots)], slots)
    assert torch.equal(input_embeds.squeeze(0)[len(slots) :], expected_answer_embeds)
    assert labels.shape == input_embeds.shape[:2]
    assert labels.squeeze(0).tolist() == [
        *([-100] * (len(slots) - 1)),
        *answer_ids,
        tokenizer.eos_token_id,
    ]


# covers: train/stage0-training :: Decoder-aligned numerical objective :: Training prompt contract
def test_training_example_uses_shared_qwen_prompt_and_only_canonical_target_text() -> None:
    tokenizer = Stage0ContractTokenizer()
    question = "A box has 7 balls and loses 6. How many remain?"

    prepared = answer_objective.prepare_training_example(tokenizer, question, "+001")

    assert prepared.context_input_ids.tolist() == [[50, 51, 52]]
    assert prepared.canonical_target == "1"
    assert tokenizer.target_calls == [
        ("1", {"add_special_tokens": False, "return_tensors": "pt"})
    ]
    assert tokenizer.chat_calls == [
        (
            qwen_messages(question),
            {
                "tokenize": True,
                "continue_final_message": True,
                "return_dict": True,
                "return_tensors": "pt",
                "padding": False,
                "truncation": False,
            },
        )
    ]
    assert all(question not in text for text, _ in tokenizer.target_calls)
    assert all("reasoning" not in text.lower() for text, _ in tokenizer.target_calls)
