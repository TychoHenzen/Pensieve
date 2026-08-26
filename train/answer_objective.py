"""Decoder-aligned teacher forcing for latent-core answer loss."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from eval.stage0_identity import apply_qwen_chat_template
from eval.stream.generators.asdiv_a import canonicalize_numerical_target


SUBJECT_LATENT_RUNS_PER_ANSWER = 2
QWEN_PROMPT_TOKEN_LIMIT = 512
DEFAULT_PROMPT_ALIGNMENT_WEIGHT = 0.1


@dataclass(frozen=True)
class PreparedTrainingExample:
    """The Qwen context and canonical answer tokens for one training row."""

    context_input_ids: torch.Tensor
    answer_ids: torch.Tensor
    canonical_target: str


@dataclass(frozen=True)
class PromptAlignedAnswerObjective:
    """Per-example answer loss and question-conditioned alignment loss."""

    language_model_loss: torch.Tensor
    prompt_alignment_loss: torch.Tensor


def validate_prompt_alignment_weight(weight: float) -> float:
    """Return a finite non-negative question-conditioning loss weight."""
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("prompt_alignment_weight must be finite and non-negative")
    return weight


def _decoder_hidden_states(
    model: object,
    *,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Run the pinned decoder body while retaining its final hidden state."""
    body = getattr(model, "model", None)
    if not callable(body):
        raise ValueError("language model must expose a callable model body")
    outputs = body(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    hidden_states = getattr(outputs, "last_hidden_state", None)
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
        raise ValueError("decoder body must return three-dimensional last_hidden_state")
    return hidden_states


def _decoder_hidden_and_logits(
    model: object,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the pinned decoder body and frozen vocabulary head."""
    hidden_states = _decoder_hidden_states(
        model,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    lm_head = getattr(model, "lm_head", None)
    if not callable(lm_head):
        raise ValueError("language model must expose callable model and lm_head modules")
    logits = lm_head(hidden_states)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("language-model head must return three-dimensional logits")
    return hidden_states, logits


def prompt_teacher_state(
    model: object,
    context_input_ids: torch.Tensor,
) -> torch.Tensor:
    """Return the detached Qwen prompt state that predicts the first answer token."""
    if context_input_ids.ndim != 2 or context_input_ids.shape[0] != 1:
        raise ValueError("teacher context input_ids must have shape (1, tokens)")
    if context_input_ids.shape[1] < 1:
        raise ValueError("teacher context must contain at least one token")
    attention_mask = torch.ones_like(context_input_ids, dtype=torch.long)
    position_ids = torch.arange(
        context_input_ids.shape[1],
        dtype=torch.long,
        device=context_input_ids.device,
    ).unsqueeze(0)
    with torch.no_grad():
        hidden_states = _decoder_hidden_states(
            model,
            input_ids=context_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    return hidden_states[:, -1, :].detach()


def prepare_training_example(
    tokenizer: object,
    question: str,
    target: str,
) -> PreparedTrainingExample:
    """Apply the shared Stage 0 prompt and exact numerical target contract."""
    context = apply_qwen_chat_template(tokenizer, question)
    if not isinstance(context, Mapping) or "input_ids" not in context:
        raise ValueError("Qwen chat template must return input_ids")
    context_input_ids = context["input_ids"]
    if not isinstance(context_input_ids, torch.Tensor) or context_input_ids.ndim != 2:
        raise ValueError("Qwen chat input_ids must have shape (1, tokens)")
    if context_input_ids.shape[0] != 1:
        raise ValueError("Qwen chat input_ids must contain exactly one example")
    if context_input_ids.shape[1] > QWEN_PROMPT_TOKEN_LIMIT:
        raise ValueError(
            "Qwen chat prompt exceeds the 512-token Stage 0 limit: "
            f"actual {context_input_ids.shape[1]}"
        )

    canonical_target = canonicalize_numerical_target(target)
    encoded_target = tokenizer(
        canonical_target,
        add_special_tokens=False,
        return_tensors="pt",
    )
    if not isinstance(encoded_target, Mapping) or "input_ids" not in encoded_target:
        raise ValueError("Qwen target tokenizer must return input_ids")
    answer_ids = encoded_target["input_ids"]
    if not isinstance(answer_ids, torch.Tensor) or answer_ids.ndim != 2:
        raise ValueError("Qwen target input_ids must have shape (1, tokens)")
    if answer_ids.shape[0] != 1 or answer_ids.shape[1] < 1:
        raise ValueError("Qwen target must tokenize to at least one token")
    return PreparedTrainingExample(
        context_input_ids=context_input_ids,
        answer_ids=answer_ids.squeeze(0),
        canonical_target=canonical_target,
    )


def decoder_aligned_inputs_and_labels(
    model: object,
    slots: torch.Tensor,
    answer_ids: torch.Tensor,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``[slots, answer]`` inputs and shifted decoder labels."""
    if slots.ndim == 2:
        slots = slots.unsqueeze(0)
    if slots.ndim != 3:
        raise ValueError(f"slots must have shape (N, H) or (B, N, H), got {slots.shape}")
    if slots.shape[1] < 1:
        raise ValueError("slots must contain at least one slot")
    if eos_token_id is None:
        raise ValueError("Qwen tokenizer must define eos_token_id")

    targets = answer_ids.reshape(-1).to(device=slots.device, dtype=torch.long)
    if targets.numel() < 1:
        raise ValueError("answer must contain at least one token")
    embedding_layer = model.get_input_embeddings()  # type: ignore[attr-defined]
    answer_embeds = embedding_layer(targets.unsqueeze(0))
    answer_embeds = answer_embeds.expand(slots.shape[0], -1, -1)
    input_embeds = torch.cat([slots, answer_embeds], dim=1)

    ignore = targets.new_full((slots.shape[1] - 1,), -100)
    labels = torch.cat([ignore, targets, targets.new_tensor([eos_token_id])])
    labels = labels.unsqueeze(0).expand(slots.shape[0], -1)
    return input_embeds, labels


def decoder_aligned_answer_loss(
    model: object,
    slots: torch.Tensor,
    answer_ids: torch.Tensor,
    eos_token_id: int | None,
) -> torch.Tensor:
    """Return per-example loss for the sequence consumed by ``SlotDecoder``.

    The last slot predicts the first answer token. Each answer-token prefix
    predicts the next token, and EOS terminates the target sequence.
    """
    input_embeds, labels = decoder_aligned_inputs_and_labels(
        model, slots, answer_ids, eos_token_id
    )
    batch_size, sequence_length = labels.shape
    attention_mask = torch.ones(
        (batch_size, sequence_length), dtype=torch.long, device=input_embeds.device
    )
    position_ids = torch.arange(
        sequence_length, dtype=torch.long, device=input_embeds.device
    ).unsqueeze(0).expand(batch_size, -1)
    logits = model(  # type: ignore[operator]
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
    ).logits
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    token_losses = token_losses.reshape(batch_size, sequence_length)
    supervised = labels.ne(-100)
    return (token_losses * supervised).sum(dim=1) / supervised.sum(dim=1)


def prompt_aligned_answer_objective(
    model: object,
    slots: torch.Tensor,
    answer_ids: torch.Tensor,
    eos_token_id: int | None,
    *,
    teacher_state: torch.Tensor,
) -> PromptAlignedAnswerObjective:
    """Train answer decoding while preserving Qwen's question-conditioned state."""
    input_embeds, labels = decoder_aligned_inputs_and_labels(
        model, slots, answer_ids, eos_token_id
    )
    batch_size, sequence_length = labels.shape
    attention_mask = torch.ones(
        (batch_size, sequence_length), dtype=torch.long, device=input_embeds.device
    )
    position_ids = torch.arange(
        sequence_length, dtype=torch.long, device=input_embeds.device
    ).unsqueeze(0).expand(batch_size, -1)
    hidden_states, logits = _decoder_hidden_and_logits(
        model,
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(batch_size, sequence_length)
    supervised = labels.ne(-100)
    language_model_loss = (token_losses * supervised).sum(dim=1) / supervised.sum(
        dim=1
    )

    slot_count = input_embeds.shape[1] - answer_ids.numel()
    student_state = hidden_states[:, slot_count - 1, :]
    detached_teacher = teacher_state.detach().to(
        device=student_state.device,
        dtype=student_state.dtype,
    )
    if detached_teacher.ndim == 1:
        detached_teacher = detached_teacher.unsqueeze(0)
    if detached_teacher.ndim != 2 or detached_teacher.shape[1] != student_state.shape[1]:
        raise ValueError("teacher state must have shape (1, hidden) or (batch, hidden)")
    if detached_teacher.shape[0] == 1:
        detached_teacher = detached_teacher.expand(batch_size, -1)
    elif detached_teacher.shape[0] != batch_size:
        raise ValueError("teacher-state batch size must match the slot batch size")
    prompt_alignment_loss = F.mse_loss(
        student_state,
        detached_teacher,
        reduction="none",
    ).mean(dim=-1)
    return PromptAlignedAnswerObjective(
        language_model_loss=language_model_loss,
        prompt_alignment_loss=prompt_alignment_loss,
    )
