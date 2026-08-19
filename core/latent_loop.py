from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from eval.subject import CostCounters
from workspace.concept_slots import SLOT_DIM, Workspace

DEFAULT_MODEL_NAME = "EleutherAI/pythia-160m"
DEFAULT_NUM_STEPS = 2
DEFAULT_RESIDUAL_WEIGHT = 0.5
DEFAULT_TAP_LAYER = 6


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
        tap_layer: int = DEFAULT_TAP_LAYER,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_steps = num_steps
        self.device = device
        self.residual_weight = residual_weight
        self.tap_layer = tap_layer

        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_dim = self.model.config.hidden_size
        self.projection = nn.Linear(self.hidden_dim, SLOT_DIM)
        self.projection.to(device)

        gamma_init = 0.7 / (SLOT_DIM ** 0.5)
        self.proj_norm = nn.LayerNorm(SLOT_DIM)
        self.layer_norm = nn.LayerNorm(SLOT_DIM)
        with torch.no_grad():
            self.proj_norm.weight.fill_(gamma_init)
            self.layer_norm.weight.fill_(gamma_init)
        self.proj_norm.to(device)
        self.layer_norm.to(device)

        self._steps = 0
        self._flops = 0

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up token IDs in Pythia's embedding table. Returns (seq_len, hidden_dim)."""
        with torch.no_grad():
            return self.model.gpt_neox.embed_in(token_ids.to(self.device)).squeeze(0)

    def step(
        self, workspace: Workspace, context_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run one latent step: project, add residual, normalize.

        When context_embeds is provided (shape (C, hidden_dim)), those
        vectors are prepended to the slot sequence so the slots attend
        to question context as well as to each other.
        """
        slots = workspace.read_slots().to(self.device)

        if context_embeds is not None:
            combined = torch.cat([context_embeds, slots], dim=0)
            input_embeds = combined.unsqueeze(0)
        else:
            input_embeds = slots.unsqueeze(0)

        outputs = self.model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
        )
        all_hidden = outputs.hidden_states[self.tap_layer].squeeze(0)

        if context_embeds is not None:
            slot_hidden = all_hidden[context_embeds.shape[0] :]
        else:
            slot_hidden = all_hidden

        seq_len = input_embeds.shape[1]

        projected = self.proj_norm(self.projection(slot_hidden))
        updated = self.layer_norm(projected + self.residual_weight * slots)
        workspace.write_slots(updated)

        self._steps += 1
        self._flops += self.flops_per_step(seq_len)

        return updated

    def run(
        self, workspace: Workspace, context_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run num_steps latent steps in sequence, mutating the workspace in place.

        When context_embeds is provided, each step prepends those vectors
        to the slot sequence so slots attend to question context rather
        than only to each other.
        """
        initial = self.layer_norm(workspace.read_slots().to(self.device))
        workspace.write_slots(initial)
        result = initial
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
