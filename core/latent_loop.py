from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from workspace.concept_slots import SLOT_DIM, Workspace

DEFAULT_MODEL_NAME = "EleutherAI/pythia-160m"
DEFAULT_NUM_STEPS = 8


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
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_steps = num_steps
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_dim = self.model.config.hidden_size
        self.projection = nn.Linear(self.hidden_dim, SLOT_DIM)
        self.projection.to(device)

    def step(self, workspace: Workspace) -> torch.Tensor:
        """Run one latent step: slots -> model forward -> projection -> slots."""
        slots = workspace.read_slots().to(self.device)
        input_embeds = slots.unsqueeze(0)  # (1, N, hidden_dim)

        outputs = self.model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
        )
        last_hidden_state = outputs.hidden_states[-1].squeeze(0)  # (N, hidden_dim)

        projected = self.projection(last_hidden_state)  # (N, SLOT_DIM)
        workspace.write_slots(projected)
        return projected

    def run(self, workspace: Workspace) -> torch.Tensor:
        """Run num_steps latent steps in sequence, mutating the workspace in place."""
        result = workspace.read_slots()
        for _ in range(self.num_steps):
            result = self.step(workspace)
        return result

    def flops_per_step(self, slot_count: int) -> int:
        """Approximate FLOPs for one forward pass over slot_count tokens.

        Uses the standard transformer approximation of 2 * params * tokens
        (a forward pass costs roughly 2 FLOPs per parameter per token).
        """
        num_params = sum(p.numel() for p in self.model.parameters())
        return 2 * num_params * slot_count
