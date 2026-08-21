"""Shared trainable model state for the gradient and EGGROLL trainers."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from codecs_module.encoder import SlotEncoder
from core.latent_loop import LatentLoop
from eval.stage0_identity import load_frozen_qwen_backbone
from workspace.concept_slots import DEFAULT_SLOT_COUNT, Workspace


class TrainingState:
    """Own the model objects and their named trainable-parameter registry."""

    def __init__(
        self,
        slot_count: int = DEFAULT_SLOT_COUNT,
        num_steps: int = 2,
        device: str = "cpu",
        backbone: object | None = None,
        sentence_model: object | None = None,
    ) -> None:
        self.backbone = (
            load_frozen_qwen_backbone(device=device) if backbone is None else backbone
        )
        self.workspace = Workspace(slot_count=slot_count)
        self.encoder = SlotEncoder(
            slot_count=slot_count,
            device=device,
            sentence_model=sentence_model,
        )
        self.latent_loop = LatentLoop(
            num_steps=num_steps,
            device=device,
            backbone=self.backbone,
        )
        self.tokenizer = self.backbone.tokenizer
        self.trainable_params: dict[str, nn.Parameter] = {
            "encoder.projection.weight": self.encoder.projection.weight,
            "encoder.projection.bias": self.encoder.projection.bias,
            "encoder.slot_queries": self.encoder.slot_queries,
            "encoder.attn_log_temp": self.encoder.attn_log_temp,
            "latent_loop.projection.weight": self.latent_loop.projection.weight,
            "latent_loop.projection.bias": self.latent_loop.projection.bias,
            "latent_loop.proj_norm.weight": self.latent_loop.proj_norm.weight,
            "latent_loop.proj_norm.bias": self.latent_loop.proj_norm.bias,
            "latent_loop.layer_norm.weight": self.latent_loop.layer_norm.weight,
            "latent_loop.layer_norm.bias": self.latent_loop.layer_norm.bias,
        }
        self._validate_trainable_registry()

    def _validate_trainable_registry(self) -> None:
        allowed_ids = {id(parameter) for parameter in self.trainable_params.values()}
        actual = {
            f"encoder.{name}": parameter
            for name, parameter in self.encoder.named_parameters()
            if parameter.requires_grad
        }
        actual.update(
            {
                f"latent_loop.{name}": parameter
                for name, parameter in self.latent_loop.named_parameters()
                if parameter.requires_grad
            }
        )
        extras = sorted(
            name for name, parameter in actual.items() if id(parameter) not in allowed_ids
        )
        missing = sorted(
            name
            for name, parameter in self.trainable_params.items()
            if not parameter.requires_grad
        )
        if extras or missing:
            details = []
            if extras:
                details.append(f"extra trainable parameters: {', '.join(extras)}")
            if missing:
                details.append(f"frozen allowed parameters: {', '.join(missing)}")
            raise ValueError("invalid Stage 0 trainable registry: " + "; ".join(details))

    def parameters(self) -> Iterator[nn.Parameter]:
        """Yield trainable parameters in the registry's stable order."""
        return iter(self.trainable_params.values())

    def create_optimizer(self, lr: float) -> torch.optim.Adam:
        """Create an independent Adam optimizer over the shared registry."""
        return torch.optim.Adam(self.parameters(), lr=lr)
