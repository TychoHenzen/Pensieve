"""EGGROLL-style evolution strategies trainer for the latent core.

Uses low-rank perturbations with antithetic sampling to estimate
parameter gradients without backpropagation. This bypasses gradient
attenuation through the frozen Qwen backbone, which weakens standard
gradient descent for the latent loop's trainable wrappers.

Based on: https://eshyperscale.github.io/
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from codecs_module.encoder import SlotEncoder
from core.latent_loop import LatentLoop
from core.qwen_tap import PreparedQwenPrefix
from train.answer_objective import (
    SUBJECT_LATENT_RUNS_PER_ANSWER,
    decoder_aligned_answer_loss,
    prepare_training_example,
)
from train.training_results import ExperimentPosition, StepResult
from train.training_state import TrainingState
from train.vicreg import post_loop_slot_variance
from workspace.concept_slots import DEFAULT_SLOT_COUNT, Workspace

DEFAULT_POP_SIZE = 128
DEFAULT_SIGMA = 0.02
DEFAULT_LR = 1e-3
DEFAULT_RANK = 4
DEFAULT_NUM_STEPS = 2
DEFAULT_VARIANCE_WEIGHT = 1.0
DEFAULT_EVAL_BATCH_SIZE = 8


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
        eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
        use_amp: bool = False,
        device: str = "cpu",
        state: TrainingState | None = None,
    ) -> None:
        if pop_size % 2 != 0:
            raise ValueError("pop_size must be even for antithetic sampling")
        if eval_batch_size < 1:
            raise ValueError("eval_batch_size must be at least 1")

        self.state = state or TrainingState(
            slot_count=slot_count, num_steps=num_steps, device=device
        )
        self.slot_count = self.state.encoder.slot_count
        self.device = device
        self.pop_size = pop_size
        self.sigma = sigma
        self.rank = rank
        self.variance_weight = variance_weight
        self.eval_batch_size = eval_batch_size
        self.use_amp = use_amp and torch.device(device).type == "cuda"

        self.workspace = self.state.workspace
        self.encoder = self.state.encoder
        self.latent_loop = self.state.latent_loop
        self.latent_loop.model.eval()
        self.tokenizer = self.state.tokenizer

        self.trainable_params = list(self.state.parameters())
        self.optimizer = self.state.create_optimizer(lr)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_params)

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

    @staticmethod
    def _batched_layer_norm(
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = inputs.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (inputs - mean) * torch.rsqrt(variance + eps)
        return normalized * weight.unsqueeze(1) + bias.unsqueeze(1)

    def _forward_fitness_batch(
        self,
        token_embeddings: torch.Tensor,
        context_embeds: torch.Tensor,
        answer_ids: torch.Tensor,
        perturbations: list[list[torch.Tensor]],
        eos_token_id: int | None = None,
        context_prefix: PreparedQwenPrefix | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate several perturbed candidates in each frozen-model pass."""
        params = [
            torch.stack([d[param_index] for d in perturbations])
            for param_index in range(len(self.trainable_params))
        ]
        batch_size = len(perturbations)

        projection_weight = self.encoder.projection.weight.unsqueeze(0) + params[0]
        projection_bias = self.encoder.projection.bias.unsqueeze(0) + params[1]
        slot_queries = self.encoder.slot_queries.unsqueeze(0) + params[2]
        attn_log_temp = self.encoder.attn_log_temp.unsqueeze(0) + params[3]

        tokens = token_embeddings.expand(batch_size, -1, -1)
        projected = torch.bmm(tokens, projection_weight.transpose(1, 2))
        projected = projected + projection_bias.unsqueeze(1)
        scale = attn_log_temp.exp().view(batch_size, 1, 1)
        attn_logits = torch.bmm(slot_queries, projected.transpose(1, 2)) / scale
        attn_weights = torch.softmax(attn_logits, dim=-1)
        slots = torch.bmm(attn_weights, projected)

        loop_projection_weight = self.latent_loop.projection.weight.unsqueeze(0) + params[4]
        loop_projection_bias = self.latent_loop.projection.bias.unsqueeze(0) + params[5]
        proj_norm_weight = self.latent_loop.proj_norm.weight.unsqueeze(0) + params[6]
        proj_norm_bias = self.latent_loop.proj_norm.bias.unsqueeze(0) + params[7]
        layer_norm_weight = self.latent_loop.layer_norm.weight.unsqueeze(0) + params[8]
        layer_norm_bias = self.latent_loop.layer_norm.bias.unsqueeze(0) + params[9]

        loop_slots = slots
        tap_adapter = self.latent_loop.tap_adapter
        if context_prefix is None:
            context_prefix = tap_adapter.prepare_prefix(context_embeds)
        for _ in range(SUBJECT_LATENT_RUNS_PER_ANSWER):
            loop_slots = self._batched_layer_norm(
                loop_slots,
                layer_norm_weight,
                layer_norm_bias,
                self.latent_loop.layer_norm.eps,
            )
            for _ in range(self.latent_loop.num_steps):
                slot_hidden = tap_adapter.cached_partial(
                    context_prefix,
                    loop_slots,
                )
                if tuple(slot_hidden.shape) != tuple(loop_slots.shape):
                    raise ValueError(
                        f"expected slot hidden shape {tuple(loop_slots.shape)}, "
                        f"actual {tuple(slot_hidden.shape)}"
                    )
                transformed = torch.bmm(
                    slot_hidden, loop_projection_weight.transpose(1, 2)
                ) + loop_projection_bias.unsqueeze(1)
                transformed = self._batched_layer_norm(
                    transformed,
                    proj_norm_weight,
                    proj_norm_bias,
                    self.latent_loop.proj_norm.eps,
                )
                loop_slots = self._batched_layer_norm(
                    transformed + self.latent_loop.residual_weight * loop_slots,
                    layer_norm_weight,
                    layer_norm_bias,
                    self.latent_loop.layer_norm.eps,
                )

        lm_loss = decoder_aligned_answer_loss(
            self.latent_loop.model,
            loop_slots,
            answer_ids,
            eos_token_id,
        )

        variance = torch.stack(
            [post_loop_slot_variance(candidate_slots) for candidate_slots in loop_slots]
        )
        log_var = variance.clamp_min(1e-10).log()
        fitness = -lm_loss + self.variance_weight * log_var
        return fitness, lm_loss, variance

    def train_step(
        self,
        question: str,
        answer: str,
        position: ExperimentPosition | None = None,
    ) -> StepResult:
        with torch.no_grad():
            token_embeddings = self.encoder._token_embeddings(question)
            prepared = prepare_training_example(self.tokenizer, question, answer)
            context_embeds = self.latent_loop.embed_tokens(prepared.context_input_ids)
            context_prefix = self.latent_loop.tap_adapter.prepare_prefix(
                context_embeds
            )
            answer_ids = prepared.answer_ids.to(self.device)

        base_seed = int(torch.randint(0, 2**31, (1,)).item())

        fitnesses = []
        lm_losses = []
        variances = []

        with torch.no_grad(), torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            for start in range(0, self.pop_size, self.eval_batch_size):
                stop = min(start + self.eval_batch_size, self.pop_size)
                perturbations = []
                for i in range(start, stop):
                    pair_seed = base_seed + (i // 2)
                    sign = 1.0 if i % 2 == 0 else -1.0
                    perturbations.append(
                        self._generate_perturbation(pair_seed, sign)
                    )

                batch_fitness, batch_losses, batch_variances = (
                    self._forward_fitness_batch(
                        token_embeddings,
                        context_embeds,
                        answer_ids,
                        perturbations,
                        getattr(self.tokenizer, "eos_token_id", None),
                        context_prefix=context_prefix,
                    )
                )
                fitnesses.extend(batch_fitness.tolist())
                lm_losses.extend(batch_losses.tolist())
                variances.extend(batch_variances.tolist())

        fitnesses_t = torch.tensor(fitnesses, device=self.device)
        normalized = (fitnesses_t - fitnesses_t.mean()) / (
            fitnesses_t.std() + 1e-5
        )

        self.optimizer.zero_grad()
        for pair_index in range(self.pop_size // 2):
            deltas = self._generate_perturbation(
                base_seed + pair_index, sign=1.0
            )
            positive_score = -normalized[2 * pair_index].item()
            negative_score = -normalized[2 * pair_index + 1].item()
            score = positive_score - negative_score
            for p, d in zip(self.trainable_params, deltas):
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.add_(d, alpha=score)

        scale = math.sqrt(self.pop_size) / self.pop_size
        for p in self.trainable_params:
            if p.grad is not None:
                p.grad.mul_(scale)

        self.optimizer.step()

        language_model_loss = sum(lm_losses) / len(lm_losses)
        total_objective = -sum(fitnesses) / len(fitnesses)
        if position is None:
            position = ExperimentPosition(
                update_method="eggroll",
                cycle=0,
                global_step=0,
                epoch=0,
                example_position=0,
                phase_step=0,
            )

        return StepResult(
            position=position,
            language_model_loss=language_model_loss,
            total_objective=total_objective,
            regularizer_loss=total_objective - language_model_loss,
            shared_variance=sum(variances) / len(variances),
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
            total_loss += result.total_objective
            total_var += result.shared_variance
            min_var = min(min_var, result.shared_variance)
            max_var = max(max_var, result.shared_variance)
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
