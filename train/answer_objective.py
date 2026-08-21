"""Decoder-aligned teacher forcing for latent-core answer loss."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from eval.stage0_identity import apply_qwen_chat_template
from eval.stream.generators.calc_mawps import canonicalize_numerical_target


SUBJECT_LATENT_RUNS_PER_ANSWER = 2
QWEN_PROMPT_TOKEN_LIMIT = 512


@dataclass(frozen=True)
class PreparedTrainingExample:
    """The Qwen context and canonical answer tokens for one training row."""

    context_input_ids: torch.Tensor
    answer_ids: torch.Tensor
    canonical_target: str


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
