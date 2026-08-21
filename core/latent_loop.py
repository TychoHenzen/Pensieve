from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from eval.subject import CostCounters
from eval.stage0_identity import LATENT_TAP_LAYER, QWEN_MODEL, WORKSPACE_DIMENSION
from workspace.concept_slots import SLOT_DIM, Workspace

DEFAULT_MODEL_NAME = QWEN_MODEL
DEFAULT_NUM_STEPS = 2
DEFAULT_RESIDUAL_WEIGHT = 0.5
DEFAULT_TAP_LAYER = LATENT_TAP_LAYER


class LatentLoop(nn.Module):
    """Runs slot vectors through a frozen LM's hidden-state space, no decoding.

    Each step reads the workspace slots as input embeddings, runs one forward
    pass through the (frozen) language model, projects the resulting hidden
    states back into slot space with a learned linear layer, and writes the
    result back to the workspace. The projected hidden state becomes the
    input embedding for the next step; no token is ever sampled or decoded.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_steps: int = DEFAULT_NUM_STEPS,
        device: str = "cpu",
        residual_weight: float = DEFAULT_RESIDUAL_WEIGHT,
        tap_layer: int | None = None,
        backbone: object | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_steps = num_steps
        self.device = device
        self.residual_weight = residual_weight
        self.backbone = backbone
        self.tap_layer = (
            LATENT_TAP_LAYER if backbone is not None else tap_layer or DEFAULT_TAP_LAYER
        )

        self.model = (
            backbone.model
            if backbone is not None
            else AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        )
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.hidden_dim = self.model.config.hidden_size
        if self.hidden_dim != SLOT_DIM:
            raise ValueError(
                f"expected workspace width {SLOT_DIM}, actual model hidden width {self.hidden_dim}"
            )
        if backbone is not None:
            self._validate_qwen_shape()
        self.projection = nn.Linear(self.hidden_dim, SLOT_DIM)
        self.projection.to(device)

        gamma_init = 0.7 / (SLOT_DIM ** 0.5)
        self.proj_norm = nn.LayerNorm(SLOT_DIM, eps=1e-5)
        self.layer_norm = nn.LayerNorm(SLOT_DIM, eps=1e-5)
        with torch.no_grad():
            self.proj_norm.weight.fill_(gamma_init)
            self.layer_norm.weight.fill_(gamma_init)
        self.proj_norm.to(device)
        self.layer_norm.to(device)

        self._steps = 0
        self._flops = 0

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up token IDs through the backbone's public embedding interface."""
        with torch.no_grad():
            return self.model.get_input_embeddings()(token_ids.to(self.device)).squeeze(0)

    def _validate_qwen_shape(self) -> None:
        actual_width = getattr(self.model.config, "hidden_size", None)
        if actual_width != WORKSPACE_DIMENSION:
            raise ValueError(
                "expected Qwen hidden width "
                f"{WORKSPACE_DIMENSION}, actual hidden width {actual_width}"
            )
        actual_layers = getattr(self.model.config, "num_hidden_layers", None)
        if not isinstance(actual_layers, int) or actual_layers < LATENT_TAP_LAYER:
            raise ValueError(
                "expected Qwen tap layer "
                f"{LATENT_TAP_LAYER}, actual model exposes {actual_layers} hidden layers"
            )

    def step(
        self, workspace: Workspace, context_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run one latent step: project, add residual, normalize.

        When context_embeds is provided (shape (C, hidden_dim)), those
        vectors are prepended to the slot sequence so the slots attend
        to question context as well as to each other.
        """
        slots = workspace.read_slots().to(self.device)
        context = (
            context_embeds.to(self.device)
            if context_embeds is not None
            else slots.new_empty((0, self.hidden_dim))
        )
        input_embeds = torch.cat([context, slots], dim=0).unsqueeze(0)
        sequence_length = input_embeds.shape[1]
        attention_mask = torch.ones(
            (1, sequence_length), dtype=torch.long, device=self.device
        )
        position_ids = torch.arange(
            sequence_length, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        if len(hidden_states) <= self.tap_layer:
            raise ValueError(
                "expected hidden-state tuple with index "
                f"{self.tap_layer}, actual length {len(hidden_states)}"
            )
        expected_shape = (1, sequence_length, WORKSPACE_DIMENSION)
        tapped_hidden = hidden_states[self.tap_layer]
        if tuple(tapped_hidden.shape) != expected_shape:
            expected_shape_text = ", ".join(str(size) for size in expected_shape)
            actual_shape_text = ", ".join(str(size) for size in tapped_hidden.shape)
            raise ValueError(
                f"expected hidden state shape {expected_shape_text}, actual "
                f"{actual_shape_text}"
            )
        slot_hidden = tapped_hidden[0, -workspace.slot_count :, :]

        projected = self.proj_norm(self.projection(slot_hidden))
        updated = self.layer_norm(projected + self.residual_weight * slots)
        workspace.write_slots(updated)

        self._steps += 1
        self._flops += self.flops_per_step(sequence_length)

        return updated

    def run(
        self, workspace: Workspace, context_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run num_steps latent steps in sequence, mutating the workspace in place.

        When context_embeds is provided, each step prepends those vectors
        to the slot sequence so slots attend to question context rather
        than only to each other.
        """
        result = workspace.read_slots().to(self.device)
        for _ in range(self.num_steps):
            result = self.step(workspace, context_embeds)
        return result

    def flops_per_step(self, slot_count: int) -> int:
        """Approximate FLOPs for one forward pass over slot_count tokens.

        Uses the standard transformer approximation of 2 * params * tokens
        (a forward pass costs roughly 2 FLOPs per parameter per token).
        """
        num_params = sum(p.numel() for p in self.model.parameters())
        return 2 * num_params * slot_count

    def get_cost(self) -> CostCounters:
        """Return cumulative cost counters accrued by calls to step()."""
        return CostCounters(steps=self._steps, flops=self._flops, wall_seconds=0.0)

    def reset_cost(self) -> None:
        """Zero out the cumulative step and flop counters."""
        self._steps = 0
        self._flops = 0
