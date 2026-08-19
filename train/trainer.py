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

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoTokenizer

from codecs_module.encoder import SlotEncoder
from core.latent_loop import LatentLoop
from train.vicreg import DEFAULT_EMA_DECAY, EMAProjection, VICRegLoss
from workspace.concept_slots import DEFAULT_SLOT_COUNT, Workspace


@dataclass
class StepResult:
    loss: float
    variance: float
    covariance: float


@dataclass
class EpochStats:
    avg_loss: float
    avg_variance: float
    avg_covariance: float
    min_variance: float
    max_variance: float
    steps: int

DEFAULT_NUM_STEPS = 2
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
        ema_decay: float = DEFAULT_EMA_DECAY,
    ) -> None:
        self.slot_count = slot_count
        self.device = device

        self.workspace = Workspace(slot_count=slot_count)
        self.encoder = SlotEncoder(slot_count=slot_count, device=device)
        self.latent_loop = LatentLoop(num_steps=num_steps, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.vicreg = VICRegLoss()
        self.ema_projection = EMAProjection(self.latent_loop.projection, decay=ema_decay)

        self.trainable_params = [
            *self.encoder.projection.parameters(),
            self.encoder.slot_queries,
            *self.latent_loop.projection.parameters(),
            *self.latent_loop.layer_norm.parameters(),
        ]
        self.optimizer = torch.optim.Adam(self.trainable_params, lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def trainable_param_count(self) -> int:
        """Number of trainable scalar parameters across all trainable modules."""
        return sum(p.numel() for p in self.trainable_params)

    def train_step(self, question: str, answer: str) -> StepResult:
        """One training step. The model predicts answer tokens from slots alone.

        The forward pass feeds only the slot vectors as inputs_embeds.
        Logits at positions [N-T, N-1] predict the T answer tokens.
        No answer token embeddings are concatenated, so the model
        cannot shortcut by attending to previous answer tokens.
        """
        self.optimizer.zero_grad()

        question_ids = self.tokenizer(question, return_tensors="pt")["input_ids"]
        context_embeds = self.latent_loop.embed_tokens(question_ids)

        slots = self.encoder.encode(question)
        self.workspace.write_slots(slots)
        self.latent_loop.run(self.workspace, context_embeds=context_embeds)
        loop_slots = self.workspace.read_slots()

        collapse = self.workspace.collapse_stats()

        answer_ids = self.tokenizer(answer, return_tensors="pt")["input_ids"].to(
            self.device
        )
        num_answer_tokens = answer_ids.shape[1]
        num_slots = loop_slots.shape[0]

        outputs = self.latent_loop.model(inputs_embeds=loop_slots.unsqueeze(0))
        logits = outputs.logits.squeeze(0)  # (N, vocab)

        answer_id = answer_ids.squeeze(0)
        if num_answer_tokens == 1:
            target = answer_id.expand(num_slots)
            lm_loss = self.loss_fn(logits, target)
        else:
            predict_positions = num_slots - num_answer_tokens
            if predict_positions < 0:
                predict_positions = 0
            answer_logits = logits[
                predict_positions : predict_positions + num_answer_tokens
            ]
            lm_loss = self.loss_fn(answer_logits, answer_id)
        vicreg_loss = self.vicreg(slots)
        loss = lm_loss + vicreg_loss
        loss.backward()
        self.optimizer.step()
        self.ema_projection.update()

        return StepResult(
            loss=loss.item(),
            variance=collapse["variance"],
            covariance=collapse["covariance"],
        )

    def train_epoch(
        self,
        dataset: list[tuple[str, str]],
        on_step: None | object = None,
    ) -> EpochStats:
        """Runs one pass over `dataset`, returning aggregated stats.

        `on_step`, when provided, is called after each example with
        (step_index, total_steps, StepResult).
        """
        if not dataset:
            return EpochStats(0.0, 0.0, 0.0, 0.0, 0.0, 0)

        total_loss = 0.0
        total_var = 0.0
        total_cov = 0.0
        min_var = float("inf")
        max_var = float("-inf")

        for i, (question, answer) in enumerate(dataset):
            result = self.train_step(question, answer)
            total_loss += result.loss
            total_var += result.variance
            total_cov += result.covariance
            min_var = min(min_var, result.variance)
            max_var = max(max_var, result.variance)
            if on_step is not None:
                on_step(i, len(dataset), result)  # type: ignore[operator]

        n = len(dataset)
        return EpochStats(
            avg_loss=total_loss / n,
            avg_variance=total_var / n,
            avg_covariance=total_cov / n,
            min_variance=min_var,
            max_variance=max_var,
            steps=n,
        )
