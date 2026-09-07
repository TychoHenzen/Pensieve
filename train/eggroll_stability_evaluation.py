"""Real-model metrics for the bounded EGGROLL stability gate."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from codecs_module.decoder import SlotDecoder
from eval.gate.answer_scoring import extract_predicted_number, score_numerical_answer
from train.answer_objective import (
    SUBJECT_LATENT_RUNS_PER_ANSWER,
    _decoder_hidden_and_logits,
    decoder_aligned_inputs_and_labels,
    prepare_training_example,
    prompt_teacher_state,
)
from train.eggroll_stability import (
    BASELINE_PROBLEM_COUNT,
    ParameterRms,
    StabilityMetrics,
    StabilityTrainer,
)
from train.vicreg import post_loop_slot_variance


def evaluate_stability_metrics(
    trainer: StabilityTrainer,
    problems: Sequence[tuple[str, str]],
    *,
    device: str,
    max_decode_tokens: int,
) -> StabilityMetrics:
    """Measure task quality and cross-problem separation on the real model."""
    if len(problems) != BASELINE_PROBLEM_COUNT:
        raise ValueError(
            f"stability evaluation requires {BASELINE_PROBLEM_COUNT} problems"
        )
    workspace = trainer.state.workspace
    encoder = trainer.state.encoder
    latent_loop = trainer.state.latent_loop
    tokenizer = trainer.state.tokenizer
    language_model = latent_loop.model
    decoder = SlotDecoder(
        language_model,
        tokenizer,
        max_tokens=max_decode_tokens,
        device=device,
    )
    modules = [encoder, latent_loop, language_model]
    previous_modes = [module.training for module in modules]

    decoded_answers: list[str] = []
    student_states: list[torch.Tensor] = []
    teacher_states: list[torch.Tensor] = []
    first_token_correct = 0
    exact = 0
    valid = 0
    total_loss = 0.0
    total_variance = 0.0
    try:
        for module in modules:
            module.eval()
        with torch.no_grad():
            for question, answer in problems:
                prepared = prepare_training_example(tokenizer, question, answer)
                context_ids = prepared.context_input_ids.to(device)
                context_embeds = latent_loop.embed_tokens(context_ids)
                workspace.write_slots(encoder.encode(question))
                for _ in range(SUBJECT_LATENT_RUNS_PER_ANSWER):
                    latent_loop.run(workspace, context_embeds=context_embeds)
                slots = workspace.read_slots()
                input_embeds, labels = decoder_aligned_inputs_and_labels(
                    language_model,
                    slots,
                    prepared.answer_ids.to(device),
                    getattr(tokenizer, "eos_token_id", None),
                )
                sequence_length = labels.shape[1]
                attention_mask = torch.ones_like(labels, dtype=torch.long)
                position_ids = torch.arange(
                    sequence_length,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                hidden, logits = _decoder_hidden_and_logits(
                    language_model,
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                supervised = labels.ne(-100)
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).reshape_as(labels)
                total_loss += float(
                    ((losses * supervised).sum() / supervised.sum()).item()
                )
                slot_count = slots.shape[-2]
                student_state = hidden[0, slot_count - 1]
                teacher_state = prompt_teacher_state(language_model, context_ids)[0]
                prediction = int(logits[0, slot_count - 1].argmax().item())
                target = int(prepared.answer_ids[0].item())
                generated = decoder.decode(workspace)

                student_states.append(student_state.detach().cpu())
                teacher_states.append(teacher_state.detach().cpu())
                decoded_answers.append(generated)
                first_token_correct += int(prediction == target)
                exact += int(score_numerical_answer(generated, answer))
                valid += int(extract_predicted_number(generated) is not None)
                total_variance += float(post_loop_slot_variance(slots).item())
    finally:
        for module, previous_mode in zip(modules, previous_modes, strict=True):
            module.train(previous_mode)

    count = len(problems)
    answer_counts = Counter(decoded_answers)
    student_cross = _cross_problem_cosine(student_states)
    teacher_cross = _cross_problem_cosine(teacher_states)
    teacher_separation = max(1.0 - teacher_cross, 1e-8)
    student_separation = max(1.0 - student_cross, 0.0)
    return StabilityMetrics(
        problem_count=count,
        parameter_rms=tuple(
            ParameterRms(
                path,
                float(torch.sqrt(torch.mean(parameter.detach().float().square())).item()),
            )
            for path, parameter in trainer.state.trainable_params.items()
        ),
        language_model_loss=total_loss / count,
        exact_accuracy=exact / count,
        first_token_accuracy=first_token_correct / count,
        valid_answer_rate=valid / count,
        output_diversity=len(answer_counts) / count,
        output_dominance=answer_counts.most_common(1)[0][1] / count,
        shared_slot_variance=total_variance / count,
        student_teacher_mse=float(
            F.mse_loss(
                torch.stack(student_states),
                torch.stack(teacher_states),
            ).item()
        ),
        student_cross_problem_cosine=student_cross,
        teacher_cross_problem_cosine=teacher_cross,
        separation_retention=min(student_separation / teacher_separation, 1.0),
    )


def _cross_problem_cosine(states: Sequence[torch.Tensor]) -> float:
    normalized = F.normalize(torch.stack(list(states)), dim=-1)
    similarities = normalized @ normalized.T
    mask = ~torch.eye(len(states), dtype=torch.bool)
    return float(similarities[mask].mean().item())
