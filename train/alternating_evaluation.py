"""Held-out GSM8K evaluation for the alternating training experiment.

The evaluator is deliberately separate from either update engine.  It uses
the shared model objects in read-only mode and returns the common evaluation
record without advancing an optimizer or a training schedule.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from eval.stream.generators.gsm8k import _extract_answer, _load_split
from train.training_results import EvaluationResult, ExperimentPosition
from train.vicreg import post_loop_slot_variance

DEFAULT_EVAL_PROBLEM_COUNT = 128


@dataclass(frozen=True)
class HeldOutProblem:
    """One fixed GSM8K question and its normalized numerical answer."""

    question: str
    answer: str


class SharedModel(Protocol):
    """The shared components needed for an unperturbed evaluation pass."""

    workspace: object
    encoder: object
    latent_loop: object
    tokenizer: object


def load_held_out_problems(
    problem_count: int = DEFAULT_EVAL_PROBLEM_COUNT,
) -> list[HeldOutProblem]:
    """Load the deterministic prefix of the cached GSM8K test split.

    Call this once when a run starts, then retain its returned list for every
    phase and epoch evaluation in that run.
    """
    if problem_count < 1:
        raise ValueError(f"problem_count must be at least 1, got {problem_count}")

    return [
        HeldOutProblem(item["question"], _extract_answer(item["answer"]))
        for item in _load_split("test")[:problem_count]
    ]


def evaluate_unperturbed(
    shared_model: SharedModel,
    problems: Sequence[HeldOutProblem],
    position: ExperimentPosition,
    answer_decoder: Callable[[object], str] | None = None,
) -> EvaluationResult:
    """Measure a shared model without updating its parameters or optimizers.

    Loss uses the same answer-token rule as gradient training.  Variance uses
    the canonical post-loop metric.  Generated answers are compared directly
    with GSM8K's normalized numerical targets.
    """
    if not problems:
        return EvaluationResult(position, 0.0, 0.0, 0.0)

    workspace = shared_model.workspace
    encoder = shared_model.encoder
    latent_loop = shared_model.latent_loop
    tokenizer = shared_model.tokenizer
    language_model = latent_loop.model  # type: ignore[attr-defined]
    decoder = answer_decoder or _slot_decoder(language_model, tokenizer)
    modules = _evaluation_modules(encoder, latent_loop, language_model)
    workspace_state = workspace.snapshot()  # type: ignore[attr-defined]
    previous_modes = [module.training for module in modules]

    try:
        for module in modules:
            module.eval()

        total_loss = 0.0
        total_variance = 0.0
        correct = 0
        with torch.no_grad():
            for problem in problems:
                question_ids = tokenizer(problem.question, return_tensors="pt")["input_ids"]
                context_embeds = latent_loop.embed_tokens(question_ids)  # type: ignore[attr-defined]
                workspace.write_slots(encoder.encode(problem.question))  # type: ignore[attr-defined]
                latent_loop.run(workspace, context_embeds=context_embeds)  # type: ignore[attr-defined]
                loop_slots = workspace.read_slots()  # type: ignore[attr-defined]

                answer_ids = tokenizer(problem.answer, return_tensors="pt")["input_ids"]
                answer_ids = answer_ids.to(loop_slots.device).squeeze(0)
                logits = language_model(inputs_embeds=loop_slots.unsqueeze(0)).logits.squeeze(0)
                total_loss += _answer_loss(logits, answer_ids).item()
                total_variance += post_loop_slot_variance(loop_slots).item()
                correct += int(decoder(workspace) == problem.answer)
    finally:
        workspace.restore(workspace_state)  # type: ignore[attr-defined]
        for module, was_training in zip(modules, previous_modes):
            module.train(was_training)

    total = len(problems)
    return EvaluationResult(
        position=position,
        language_model_loss=total_loss / total,
        shared_variance=total_variance / total,
        answer_exact_match=correct / total,
    )


def _answer_loss(logits: torch.Tensor, answer_ids: torch.Tensor) -> torch.Tensor:
    """Return teacher-forced answer loss using the trainer's token rule."""
    if answer_ids.numel() == 1:
        return nn.functional.cross_entropy(logits, answer_ids.expand(logits.shape[0]))

    first_position = max(logits.shape[0] - answer_ids.numel(), 0)
    answer_logits = logits[first_position : first_position + answer_ids.numel()]
    return nn.functional.cross_entropy(answer_logits, answer_ids)


def _evaluation_modules(*candidates: object) -> list[nn.Module]:
    """Return unique modules whose train/eval mode evaluation temporarily changes."""
    modules: list[nn.Module] = []
    for candidate in candidates:
        if isinstance(candidate, nn.Module) and all(candidate is not item for item in modules):
            modules.append(candidate)
    return modules


def _slot_decoder(language_model: object, tokenizer: object) -> Callable[[object], str]:
    """Build the real decoder only when generated answers are required."""
    from codecs_module.decoder import SlotDecoder

    return SlotDecoder(language_model, tokenizer).decode
