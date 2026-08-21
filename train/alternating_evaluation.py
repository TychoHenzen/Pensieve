"""Held-out Calc-MAWPS evaluation for the alternating training experiment.

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

from eval.gate.answer_scoring import score_numerical_answer
from eval.stream.generators.calc_mawps import CalcMawpsRecord
from train.answer_objective import (
    SUBJECT_LATENT_RUNS_PER_ANSWER,
    decoder_aligned_answer_loss,
    prepare_training_example,
)
from train.training_results import EvaluationResult, ExperimentPosition
from train.vicreg import post_loop_slot_variance

DEFAULT_EVAL_PROBLEM_COUNT = 128


@dataclass(frozen=True)
class HeldOutProblem:
    """One fixed Calc-MAWPS question and its normalized numerical answer."""

    question: str
    answer: str


class SharedModel(Protocol):
    """The shared components needed for an unperturbed evaluation pass."""

    workspace: object
    encoder: object
    latent_loop: object
    tokenizer: object


def load_held_out_problems(
    records: Sequence[CalcMawpsRecord],
    problem_count: int = DEFAULT_EVAL_PROBLEM_COUNT,
) -> list[HeldOutProblem]:
    """Convert the persisted seed-0 validation prefix into evaluator inputs.

    The caller supplies canonical records from ``Stage0Dataset``. This keeps
    evaluation isolated from the test split and from dataset loading details.
    """
    if problem_count < 1:
        raise ValueError(f"problem_count must be at least 1, got {problem_count}")
    if problem_count > len(records):
        raise ValueError(
            f"problem_count must not exceed the {len(records)} persisted "
            "validation records"
        )
    if any(record.split != "validation" for record in records):
        raise ValueError("held-out evaluation accepts only validation records")

    return [
        HeldOutProblem(record.question, record.target)
        for record in records[:problem_count]
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
    with Calc-MAWPS's normalized numerical targets.
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
                prepared = prepare_training_example(
                    tokenizer, problem.question, problem.answer
                )
                context_embeds = latent_loop.embed_tokens(  # type: ignore[attr-defined]
                    prepared.context_input_ids
                )
                workspace.write_slots(encoder.encode(problem.question))  # type: ignore[attr-defined]
                for _ in range(SUBJECT_LATENT_RUNS_PER_ANSWER):
                    latent_loop.run(workspace, context_embeds=context_embeds)  # type: ignore[attr-defined]
                loop_slots = workspace.read_slots()  # type: ignore[attr-defined]

                loss = decoder_aligned_answer_loss(
                    language_model,
                    loop_slots,
                    prepared.answer_ids.to(loop_slots.device),
                    getattr(tokenizer, "eos_token_id", None),
                )
                total_loss += loss.mean().item()
                total_variance += post_loop_slot_variance(loop_slots).item()
                correct += int(score_numerical_answer(decoder(workspace), problem.answer))
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
