"""Narrow access to the pinned Qwen layer-12 hidden state."""

from __future__ import annotations

from typing import Any

import torch

from eval.stage0_identity import LATENT_TAP_LAYER, WORKSPACE_DIMENSION


class QwenTapAdapter:
    """Validate and select the latent slot state from the frozen Qwen model."""

    def __init__(self, model: Any) -> None:
        self.model = model
        actual_width = getattr(model.config, "hidden_size", None)
        actual_layers = getattr(model.config, "num_hidden_layers", None)
        if (
            actual_width != WORKSPACE_DIMENSION
            or not isinstance(actual_layers, int)
            or actual_layers < LATENT_TAP_LAYER
        ):
            raise ValueError(
                "expected Qwen shape with hidden width "
                f"{WORKSPACE_DIMENSION} and tap layer {LATENT_TAP_LAYER}, actual "
                f"hidden width {actual_width} and {actual_layers} hidden layers"
            )

        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def full_reference(
        self,
        context_embeddings: torch.Tensor,
        slot_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Run the declared full sequence and return only layer-12 slot state."""
        batched_context, batched_slots, remove_batch = self._batch_inputs(
            context_embeddings,
            slot_embeddings,
        )
        batch_size = batched_slots.shape[0]
        slot_count = batched_slots.shape[1]
        combined = torch.cat((batched_context, batched_slots), dim=1)
        sequence_length = combined.shape[1]
        attention_mask = torch.ones(
            (batch_size, sequence_length),
            dtype=torch.long,
            device=combined.device,
        )
        position_ids = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=combined.device,
        ).unsqueeze(0).expand(batch_size, -1)

        outputs = self.model(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError(
                "expected hidden-state tuple with index "
                f"{LATENT_TAP_LAYER}, actual output has no hidden states"
            )
        if len(hidden_states) <= LATENT_TAP_LAYER:
            raise ValueError(
                "expected hidden-state tuple with index "
                f"{LATENT_TAP_LAYER}, actual length {len(hidden_states)}"
            )

        tapped_hidden = hidden_states[LATENT_TAP_LAYER]
        expected_shape = (
            batch_size,
            sequence_length,
            WORKSPACE_DIMENSION,
        )
        if tuple(tapped_hidden.shape) != expected_shape:
            raise ValueError(
                f"expected hidden state shape {expected_shape}, actual "
                f"{tuple(tapped_hidden.shape)}"
            )

        selected = tapped_hidden[:, -slot_count:, :]
        return selected.squeeze(0) if remove_batch else selected

    @staticmethod
    def _batch_inputs(
        context_embeddings: torch.Tensor,
        slot_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        if slot_embeddings.ndim == 2:
            if context_embeddings.ndim != 2:
                raise ValueError(
                    "unbatched slots require unbatched context embeddings"
                )
            batched_slots = slot_embeddings.unsqueeze(0)
            batched_context = context_embeddings.unsqueeze(0)
            remove_batch = True
        elif slot_embeddings.ndim == 3:
            batched_slots = slot_embeddings
            remove_batch = False
            if context_embeddings.ndim == 2:
                batched_context = context_embeddings.unsqueeze(0).expand(
                    slot_embeddings.shape[0], -1, -1
                )
            elif context_embeddings.ndim == 3:
                if context_embeddings.shape[0] != slot_embeddings.shape[0]:
                    raise ValueError(
                        "context candidate count does not match slot candidate count"
                    )
                batched_context = context_embeddings
            else:
                raise ValueError(
                    "candidate slots require shared or candidate-batched context"
                )
        else:
            raise ValueError("slot embeddings must be unbatched or candidate-batched")

        if batched_slots.shape[1] < 1:
            raise ValueError("slot embeddings must contain at least one slot")
        if (
            batched_context.shape[-1] != WORKSPACE_DIMENSION
            or batched_slots.shape[-1] != WORKSPACE_DIMENSION
        ):
            raise ValueError(
                f"expected embedding width {WORKSPACE_DIMENSION}, actual context "
                f"width {batched_context.shape[-1]} and slot width "
                f"{batched_slots.shape[-1]}"
            )
        if batched_context.device != batched_slots.device:
            raise ValueError("context and slot embeddings must use the same device")
        if batched_context.dtype != batched_slots.dtype:
            raise ValueError("context and slot embeddings must use the same dtype")
        return batched_context, batched_slots, remove_batch
