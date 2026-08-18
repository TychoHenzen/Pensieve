"""Trains the latent-core subject's learned components on GSM8K.

The frozen Pythia-160M backbone and the frozen MiniLM sentence encoder are
never updated. Only the encoder's projection, cross-attention, and slot
queries, plus the latent loop's hidden-state projection, receive gradients.

The forward pass does not go through `LatentCoreSubject`: it builds the
sequence directly, so the loss can be computed over the answer token
positions only. `LatentCoreSubject.observe`/`answer` are read-path
conveniences that mutate the workspace in place and decode autoregressively;
neither shape fits a teacher-forced training step.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoTokenizer

from codecs_module.encoder import SlotEncoder
from core.latent_loop import LatentLoop
from workspace.concept_slots import DEFAULT_SLOT_COUNT, Workspace

DEFAULT_NUM_STEPS = 8
DEFAULT_LR = 1e-4
TOKENIZER_NAME = "EleutherAI/pythia-160m"


class LatentCoreTrainer:
    """Trains the encoder and latent loop's learned parameters on GSM8K."""

    def __init__(
        self,
        slot_count: int = DEFAULT_SLOT_COUNT,
        num_steps: int = DEFAULT_NUM_STEPS,
        lr: float = DEFAULT_LR,
        device: str = "cpu",
    ) -> None:
        self.slot_count = slot_count
        self.device = device

        self.workspace = Workspace(slot_count=slot_count)
        self.encoder = SlotEncoder(slot_count=slot_count, device=device)
        self.latent_loop = LatentLoop(num_steps=num_steps, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

        self.trainable_params = [
            *self.encoder.projection.parameters(),
            *self.encoder.cross_attention.parameters(),
            self.encoder.slot_queries,
            *self.latent_loop.projection.parameters(),
        ]
        self.optimizer = torch.optim.Adam(self.trainable_params, lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def trainable_param_count(self) -> int:
        """Number of trainable scalar parameters across all trainable modules."""
        return sum(p.numel() for p in self.trainable_params)

    def train_step(self, question: str, answer: str) -> float:
        """One teacher-forced training step; returns the scalar loss value."""
        self.optimizer.zero_grad()

        slots = self.encoder.encode(question)
        self.workspace.write_slots(slots)
        self.latent_loop.run(self.workspace)
        loop_slots = self.workspace.read_slots()

        answer_ids = self.tokenizer(answer, return_tensors="pt")["input_ids"].to(
            self.device
        )
        embedding_layer = self.latent_loop.model.get_input_embeddings()
        answer_embeds = embedding_layer(answer_ids).squeeze(0)  # (T, hidden_dim)

        input_embeds = torch.cat([loop_slots, answer_embeds], dim=0).unsqueeze(0)

        outputs = self.latent_loop.model(inputs_embeds=input_embeds)
        logits = outputs.logits.squeeze(0)  # (N + T, vocab)

        num_slots = loop_slots.shape[0]
        num_answer_tokens = answer_ids.shape[1]
        # Position (num_slots - 1) predicts the first answer token, and so on.
        answer_logits = logits[num_slots - 1 : num_slots - 1 + num_answer_tokens]

        loss = self.loss_fn(answer_logits, answer_ids.squeeze(0))
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataset: list[tuple[str, str]], batch_size: int = 1) -> float:
        """Runs one pass over `dataset`, returning the average loss.

        `batch_size` is accepted for interface compatibility; examples are
        processed one at a time since each question produces a different
        number of workspace slot writes and answer token counts.
        """
        del batch_size
        if not dataset:
            return 0.0

        total_loss = 0.0
        for question, answer in dataset:
            total_loss += self.train_step(question, answer)

        return total_loss / len(dataset)
