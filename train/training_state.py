"""Shared trainable model state for the gradient and EGGROLL trainers."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from codecs_module.encoder import SlotEncoder
from core.latent_loop import LatentLoop
from eval.stage0_identity import load_frozen_qwen_backbone
from workspace.concept_slots import DEFAULT_SLOT_COUNT, Workspace

GRADIENT_PARAMETER_PATHS = (
    "encoder.projection.weight",
    "encoder.projection.bias",
    "encoder.slot_queries",
    "encoder.attn_log_temp",
    "latent_loop.projection.weight",
    "latent_loop.projection.bias",
    "latent_loop.proj_norm.weight",
    "latent_loop.proj_norm.bias",
    "latent_loop.layer_norm.weight",
    "latent_loop.layer_norm.bias",
)

EGGROLL_PARAMETER_PATHS = (
    "encoder.projection.weight",
    "encoder.slot_queries",
    "latent_loop.projection.weight",
)


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
        if tuple(self.trainable_params) != GRADIENT_PARAMETER_PATHS:
            raise ValueError("invalid Stage 0 gradient parameter registry order")
        if any(
            self.trainable_params[path].ndim != 2
            for path in EGGROLL_PARAMETER_PATHS
        ):
            raise ValueError("invalid Stage 0 EGGROLL matrix parameter registry")
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
        """Yield the complete gradient-training registry in stable order."""
        return iter(self.trainable_params.values())

    def eggroll_parameters(self) -> Iterator[nn.Parameter]:
        """Yield the declared matrix-only EGGROLL registry in stable order."""
        return (self.trainable_params[path] for path in EGGROLL_PARAMETER_PATHS)

    def create_gradient_optimizer(self, lr: float) -> torch.optim.Adam:
        """Create Adam over every Stage 0 gradient-training parameter."""
        return torch.optim.Adam(self.parameters(), lr=lr)

    def create_eggroll_optimizer(self, lr: float) -> torch.optim.SGD:
        """Create momentum-free SGD over the matrix-only EGGROLL registry."""
        return torch.optim.SGD(self.eggroll_parameters(), lr=lr, momentum=0.0)

    def create_optimizer(self, lr: float) -> torch.optim.Adam:
        """Backward-compatible name for the complete gradient optimizer."""
        return self.create_gradient_optimizer(lr)
