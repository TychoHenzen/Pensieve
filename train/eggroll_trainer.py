"""EGGROLL-style evolution strategies trainer for the latent core.

Uses low-rank perturbations with antithetic sampling to estimate
parameter gradients without backpropagation. This bypasses gradient
attenuation through the frozen Pythia backbone, which kills standard
gradient descent for the latent loop's trainable wrappers.

Based on: https://eshyperscale.github.io/
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoTokenizer

from codecs_module.encoder import SlotEncoder
from core.latent_loop import LatentLoop
from train.training_state import TrainingState
from workspace.concept_slots import DEFAULT_SLOT_COUNT, Workspace

TOKENIZER_NAME = "EleutherAI/pythia-160m"

DEFAULT_POP_SIZE = 128
DEFAULT_SIGMA = 0.02
DEFAULT_LR = 1e-3
DEFAULT_RANK = 4
DEFAULT_NUM_STEPS = 2
DEFAULT_VARIANCE_WEIGHT = 1.0


@dataclass
class StepResult:
    loss: float
    variance: float
    best_fitness: float
    mean_fitness: float


@dataclass
class EpochStats:
    avg_loss: float
    avg_variance: float
    min_variance: float
    max_variance: float
    steps: int


class EggrollTrainer:
    """Trains encoder and loop wrappers using evolution strategies."""

    def __init__(
        self,
        slot_count: int = DEFAULT_SLOT_COUNT,
        num_steps: int = DEFAULT_NUM_STEPS,
        pop_size: int = DEFAULT_POP_SIZE,
        sigma: float = DEFAULT_SIGMA,
        lr: float = DEFAULT_LR,
        rank: int = DEFAULT_RANK,
        variance_weight: float = DEFAULT_VARIANCE_WEIGHT,
        device: str = "cpu",
        state: TrainingState | None = None,
    ) -> None:
        if pop_size % 2 != 0:
            raise ValueError("pop_size must be even for antithetic sampling")

        self.state = state or TrainingState(
            slot_count=slot_count, num_steps=num_steps, device=device
        )
        self.slot_count = self.state.encoder.slot_count
        self.device = device
        self.pop_size = pop_size
        self.sigma = sigma
        self.rank = rank
        self.variance_weight = variance_weight

        self.workspace = self.state.workspace
        self.encoder = self.state.encoder
        self.latent_loop = self.state.latent_loop
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.loss_fn = nn.CrossEntropyLoss()

        self.trainable_params = list(self.state.parameters())

        self.optimizer = self.state.create_optimizer(lr)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_params)

    def _save_params(self) -> list[torch.Tensor]:
        return [p.data.clone() for p in self.trainable_params]

    def _restore_params(self, saved: list[torch.Tensor]) -> None:
        for p, s in zip(self.trainable_params, saved):
            p.data.copy_(s)

    def _generate_perturbation(
        self, seed: int, sign: float
    ) -> list[torch.Tensor]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        deltas = []
        for p in self.trainable_params:
            if p.ndim == 2:
                a, b = p.shape
                noise = torch.randn(
                    a + b, self.rank, generator=gen, device="cpu"
                )
                big = noise[:a]
                small = noise[a:]
                delta = (big @ small.T) * (sign * self.sigma / math.sqrt(self.rank))
            else:
                noise = torch.randn(p.shape, generator=gen, device="cpu")
                delta = noise * (sign * self.sigma)
            deltas.append(delta.to(self.device))
        return deltas

    def _apply_perturbation(self, deltas: list[torch.Tensor]) -> None:
        for p, d in zip(self.trainable_params, deltas):
            p.data.add_(d)

    def _forward_fitness(
        self,
        token_embeddings: torch.Tensor,
        context_embeds: torch.Tensor,
        answer_ids: torch.Tensor,
    ) -> tuple[float, float, float]:
        projected = self.encoder.projection(token_embeddings).squeeze(0)
        scale = self.encoder.attn_log_temp.exp()
        attn_logits = self.encoder.slot_queries @ projected.T / scale
        attn_weights = torch.softmax(attn_logits, dim=-1)
        slots = attn_weights @ projected

        self.workspace.write_slots(slots)
        self.latent_loop.run(self.workspace, context_embeds=context_embeds)
        loop_slots = self.workspace.read_slots()

        outputs = self.latent_loop.model(inputs_embeds=loop_slots.unsqueeze(0))
        logits = outputs.logits.squeeze(0)

        num_answer_tokens = answer_ids.shape[1]
        num_slots = loop_slots.shape[0]
        answer_id = answer_ids.squeeze(0)

        if num_answer_tokens == 1:
            target = answer_id.expand(num_slots)
            lm_loss = self.loss_fn(logits, target)
        else:
            pos = max(0, num_slots - num_answer_tokens)
            lm_loss = self.loss_fn(logits[pos : pos + num_answer_tokens], answer_id)

        variance = loop_slots.var(dim=0).mean()
        log_var = math.log(max(variance.item(), 1e-10))
        fitness = -lm_loss.item() + self.variance_weight * log_var
        return fitness, lm_loss.item(), variance.item()

    def train_step(self, question: str, answer: str) -> StepResult:
        with torch.no_grad():
            token_embeddings = self.encoder._token_embeddings(question)
            question_ids = self.tokenizer(question, return_tensors="pt")[
                "input_ids"
            ]
            context_embeds = self.latent_loop.embed_tokens(question_ids)
            answer_ids = self.tokenizer(answer, return_tensors="pt")[
                "input_ids"
            ].to(self.device)

        saved = self._save_params()
        base_seed = int(torch.randint(0, 2**31, (1,)).item())

        fitnesses = []
        lm_losses = []
        variances = []

        with torch.no_grad():
            for i in range(self.pop_size):
                pair_seed = base_seed + (i // 2)
                sign = 1.0 if i % 2 == 0 else -1.0

                deltas = self._generate_perturbation(pair_seed, sign)
                self._apply_perturbation(deltas)

                fitness, lm_loss, variance = self._forward_fitness(
                    token_embeddings, context_embeds, answer_ids
                )
                fitnesses.append(fitness)
                lm_losses.append(lm_loss)
                variances.append(variance)

                self._restore_params(saved)

        fitnesses_t = torch.tensor(fitnesses, device=self.device)
        normalized = (fitnesses_t - fitnesses_t.mean()) / (
            fitnesses_t.std() + 1e-5
        )

        self.optimizer.zero_grad()
        for i in range(self.pop_size):
            pair_seed = base_seed + (i // 2)
            sign = 1.0 if i % 2 == 0 else -1.0
            deltas = self._generate_perturbation(pair_seed, sign)
            score = -normalized[i].item()
            for p, d in zip(self.trainable_params, deltas):
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.add_(d, alpha=score)

        scale = math.sqrt(self.pop_size) / self.pop_size
        for p in self.trainable_params:
            if p.grad is not None:
                p.grad.mul_(scale)

        self.optimizer.step()

        return StepResult(
            loss=sum(lm_losses) / len(lm_losses),
            variance=sum(variances) / len(variances),
            best_fitness=max(fitnesses),
            mean_fitness=sum(fitnesses) / len(fitnesses),
        )

    def train_epoch(
        self,
        dataset: list[tuple[str, str]],
        on_step: Callable[[int, int, StepResult], None] | None = None,
    ) -> EpochStats:
        if not dataset:
            return EpochStats(0.0, 0.0, 0.0, 0.0, 0)

        total_loss = 0.0
        total_var = 0.0
        min_var = float("inf")
        max_var = float("-inf")

        for i, (question, answer) in enumerate(dataset):
            result = self.train_step(question, answer)
            total_loss += result.loss
            total_var += result.variance
            min_var = min(min_var, result.variance)
            max_var = max(max_var, result.variance)
            if on_step is not None:
                on_step(i, len(dataset), result)

        n = len(dataset)
        return EpochStats(
            avg_loss=total_loss / n,
            avg_variance=total_var / n,
            min_variance=min_var,
            max_variance=max_var,
            steps=n,
        )
