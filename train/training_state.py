"""Shared trainable model state for the gradient and EGGROLL trainers."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from codecs_module.encoder import SlotEncoder
from core.latent_loop import LatentLoop
from workspace.concept_slots import DEFAULT_SLOT_COUNT, Workspace


class TrainingState:
    """Own the model objects and their named trainable-parameter registry."""

    def __init__(
        self,
        slot_count: int = DEFAULT_SLOT_COUNT,
        num_steps: int = 2,
        device: str = "cpu",
    ) -> None:
        self.workspace = Workspace(slot_count=slot_count)
        self.encoder = SlotEncoder(slot_count=slot_count, device=device)
        self.latent_loop = LatentLoop(num_steps=num_steps, device=device)
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

    def parameters(self) -> Iterator[nn.Parameter]:
        """Yield trainable parameters in the registry's stable order."""
        return iter(self.trainable_params.values())

    def create_optimizer(self, lr: float) -> torch.optim.Adam:
        """Create an independent Adam optimizer over the shared registry."""
        return torch.optim.Adam(self.parameters(), lr=lr)
