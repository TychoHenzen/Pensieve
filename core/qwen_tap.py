"""Narrow access to the pinned Qwen layer-12 hidden state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from eval.stage0_identity import LATENT_TAP_LAYER, WORKSPACE_DIMENSION


@dataclass(frozen=True)
class PreparedQwenPrefix:
    """Detached layer caches for one shared context."""

    context_length: int
    layer_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]


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

    def prepare_prefix(
        self,
        context_embeddings: torch.Tensor,
    ) -> PreparedQwenPrefix:
        """Prepare detached context keys and values through the tapped layer."""
        batched_context = self._single_context_batch(context_embeddings)
        context_length = batched_context.shape[1]
        attention_mask = torch.ones(
            (1, context_length),
            dtype=torch.long,
            device=batched_context.device,
        )
        position_ids = torch.arange(
            context_length,
            dtype=torch.long,
            device=batched_context.device,
        ).unsqueeze(0)
        cache = DynamicCache()
        model_body = getattr(self.model, "model", self.model)

        with torch.no_grad():
            outputs = model_body(
                inputs_embeds=batched_context,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )

        output_cache = getattr(outputs, "past_key_values", None)
        if not isinstance(output_cache, DynamicCache):
            raise ValueError(
                "expected a DynamicCache from context-prefix preparation, actual "
                f"{type(output_cache).__name__}"
            )
        if len(output_cache.layers) < LATENT_TAP_LAYER:
            raise ValueError(
                f"expected cache for {LATENT_TAP_LAYER} layers, actual "
                f"{len(output_cache.layers)}"
            )

        layer_key_values = tuple(
            self._clone_prefix_layer(
                output_cache.layers[layer_index],
                layer_index=layer_index,
                context_length=context_length,
            )
            for layer_index in range(LATENT_TAP_LAYER)
        )
        return PreparedQwenPrefix(
            context_length=context_length,
            layer_key_values=layer_key_values,
        )

    @staticmethod
    def fresh_candidate_cache(
        prefix: PreparedQwenPrefix,
        candidate_batch_size: int,
    ) -> DynamicCache:
        """Create an independent mutable cache for one candidate batch."""
        if candidate_batch_size < 1:
            raise ValueError("candidate batch size must be positive")

        cache = DynamicCache()
        for key, value in prefix.layer_key_values:
            expanded_key = key.expand(candidate_batch_size, -1, -1, -1)
            expanded_value = value.expand(candidate_batch_size, -1, -1, -1)
            layer = DynamicLayer()
            layer.lazy_initialization(expanded_key, expanded_value)
            layer.keys = expanded_key
            layer.values = expanded_value
            cache.layers.append(layer)
        return cache

    @staticmethod
    def candidate_layout(
        prefix: PreparedQwenPrefix,
        *,
        candidate_batch_size: int,
        slot_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the visible attention mask and slot position identifiers."""
        if candidate_batch_size < 1:
            raise ValueError("candidate batch size must be positive")
        if slot_count < 1:
            raise ValueError("slot count must be positive")
        device = prefix.layer_key_values[0][0].device
        attention_mask = torch.ones(
            (candidate_batch_size, prefix.context_length + slot_count),
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.arange(
            prefix.context_length,
            prefix.context_length + slot_count,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).expand(candidate_batch_size, -1)
        return attention_mask, position_ids

    def cached_partial(
        self,
        prefix: PreparedQwenPrefix,
        slot_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Run slots through decoder blocks 0 through 11 using a cached prefix."""
        batched_slots, remove_batch = self._batch_slots(prefix, slot_embeddings)
        body, layers, layer_types = self._partial_body()
        batch_size, slot_count, _ = batched_slots.shape
        cache = self.fresh_candidate_cache(prefix, batch_size)
        attention_mask, position_ids = self.candidate_layout(
            prefix,
            candidate_batch_size=batch_size,
            slot_count=slot_count,
        )
        mask_arguments = {
            "config": body.config,
            "inputs_embeds": batched_slots,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "position_ids": position_ids,
        }
        mask_builders = {
            "full_attention": create_causal_mask,
            "sliding_attention": create_sliding_window_causal_mask,
        }
        causal_masks = {
            layer_type: mask_builders[layer_type](**mask_arguments)
            for layer_type in set(layer_types)
        }
        position_embeddings = body.rotary_emb(batched_slots, position_ids)

        hidden_states = batched_slots
        for layer_index in range(LATENT_TAP_LAYER):
            hidden_states = layers[layer_index](
                hidden_states,
                attention_mask=causal_masks[layer_types[layer_index]],
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=cache,
                use_cache=True,
            )

        if tuple(hidden_states.shape) != tuple(batched_slots.shape):
            raise ValueError(
                f"expected partial hidden state shape {tuple(batched_slots.shape)}, "
                f"actual {tuple(hidden_states.shape)}"
            )
        return hidden_states.squeeze(0) if remove_batch else hidden_states

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
    def _single_context_batch(context_embeddings: torch.Tensor) -> torch.Tensor:
        if context_embeddings.ndim == 2:
            batched_context = context_embeddings.unsqueeze(0)
        elif context_embeddings.ndim == 3 and context_embeddings.shape[0] == 1:
            batched_context = context_embeddings
        else:
            raise ValueError(
                "context embeddings must be unbatched or have batch size one"
            )
        if batched_context.shape[1] < 1:
            raise ValueError("context embeddings must contain at least one token")
        if batched_context.shape[-1] != WORKSPACE_DIMENSION:
            raise ValueError(
                f"expected context embedding width {WORKSPACE_DIMENSION}, actual "
                f"{batched_context.shape[-1]}"
            )
        return batched_context

    @staticmethod
    def _batch_slots(
        prefix: PreparedQwenPrefix,
        slot_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        if len(prefix.layer_key_values) != LATENT_TAP_LAYER:
            raise ValueError(
                f"expected prepared prefix for {LATENT_TAP_LAYER} layers, actual "
                f"{len(prefix.layer_key_values)}"
            )
        if slot_embeddings.ndim == 2:
            batched_slots = slot_embeddings.unsqueeze(0)
            remove_batch = True
        elif slot_embeddings.ndim == 3:
            batched_slots = slot_embeddings
            remove_batch = False
        else:
            raise ValueError("slot embeddings must be unbatched or candidate-batched")
        if batched_slots.shape[0] < 1 or batched_slots.shape[1] < 1:
            raise ValueError("slot embeddings must contain a candidate and a slot")
        if batched_slots.shape[-1] != WORKSPACE_DIMENSION:
            raise ValueError(
                f"expected slot embedding width {WORKSPACE_DIMENSION}, actual "
                f"{batched_slots.shape[-1]}"
            )
        prefix_key = prefix.layer_key_values[0][0]
        if batched_slots.device != prefix_key.device:
            raise ValueError("prefix and slot embeddings must use the same device")
        if batched_slots.dtype != prefix_key.dtype:
            raise ValueError("prefix and slot embeddings must use the same dtype")
        return batched_slots, remove_batch

    def _partial_body(self) -> tuple[Any, Any, tuple[str, ...]]:
        body = getattr(self.model, "model", self.model)
        layers = getattr(body, "layers", None)
        rotary_emb = getattr(body, "rotary_emb", None)
        config = getattr(body, "config", None)
        layer_types = getattr(config, "layer_types", None)
        actual_layer_count = len(layers) if layers is not None else None
        if actual_layer_count is None or actual_layer_count < LATENT_TAP_LAYER:
            raise ValueError(
                f"expected model body with {LATENT_TAP_LAYER} decoder layers, "
                f"actual {actual_layer_count}"
            )
        if not callable(rotary_emb):
            raise ValueError("expected model body with callable rotary embeddings")
        if layer_types is None or len(layer_types) < LATENT_TAP_LAYER:
            actual_type_count = len(layer_types) if layer_types is not None else None
            raise ValueError(
                f"expected layer types for {LATENT_TAP_LAYER} decoder layers, "
                f"actual {actual_type_count}"
            )
        selected_types = tuple(layer_types[:LATENT_TAP_LAYER])
        supported_types = {"full_attention", "sliding_attention"}
        unsupported_types = set(selected_types) - supported_types
        if unsupported_types:
            raise ValueError(
                "expected full or sliding attention through tap layer, actual "
                f"{sorted(unsupported_types)}"
            )
        return body, layers, selected_types

    @staticmethod
    def _clone_prefix_layer(
        layer: Any,
        *,
        layer_index: int,
        context_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)
        expected_prefix = (1, context_length)
        if (
            not isinstance(key, torch.Tensor)
            or not isinstance(value, torch.Tensor)
            or key.ndim != 4
            or value.ndim != 4
            or key.shape[0] != expected_prefix[0]
            or value.shape[0] != expected_prefix[0]
            or key.shape[-2] != expected_prefix[1]
            or value.shape[-2] != expected_prefix[1]
            or key.shape != value.shape
        ):
            key_shape = tuple(key.shape) if isinstance(key, torch.Tensor) else None
            value_shape = (
                tuple(value.shape) if isinstance(value, torch.Tensor) else None
            )
            raise ValueError(
                f"expected layer {layer_index} cache shapes "
                f"(1, heads, {context_length}, head_dim), actual key "
                f"{key_shape} and value {value_shape}"
            )
        return key.detach().clone(), value.detach().clone()

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
