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
from train.eggroll_factorized import factorized_linear
from train.eggroll_perturbations import (
    MatrixFactors,
    SignedPerturbationSet,
    sample_antithetic_pair,
)
from train.answer_objective import (
    DEFAULT_PROMPT_ALIGNMENT_WEIGHT,
    SUBJECT_LATENT_RUNS_PER_ANSWER,
    prepare_training_example,
    prompt_aligned_answer_objective,
    prompt_teacher_state,
    validate_prompt_alignment_weight,
)
from train.training_results import ExperimentPosition, StepResult
from train.training_state import TrainingState
from train.eggroll_updates import apply_factorized_update
from train.vicreg import post_loop_slot_variance, slot_variance_penalty
from workspace.concept_slots import DEFAULT_SLOT_COUNT, Workspace

DEFAULT_POP_SIZE = 128
DEFAULT_SIGMA = 0.001
DEFAULT_LR = 0.1
DEFAULT_RANK = 4
DEFAULT_NUM_STEPS = 2
DEFAULT_VARIANCE_WEIGHT = 1.0
DEFAULT_EVAL_BATCH_SIZE = 8


def validate_eggroll_config(
    pop_size: int,
    sigma: float,
    lr: float,
    rank: int,
    eval_batch_size: int,
) -> None:
    """Reject invalid EGGROLL values before production dependencies load."""
    if pop_size < 2 or pop_size % 2 != 0:
        raise ValueError("pop_size must be an even integer of at least 2")
    if rank < 1:
        raise ValueError("rank must be at least 1")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if eval_batch_size < 2:
        raise ValueError("eval_batch_size must be at least 2")


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
        prompt_alignment_weight: float = DEFAULT_PROMPT_ALIGNMENT_WEIGHT,
        eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
        use_amp: bool = False,
        device: str = "cpu",
        state: TrainingState | None = None,
    ) -> None:
        validate_eggroll_config(pop_size, sigma, lr, rank, eval_batch_size)

        self.state = state or TrainingState(
            slot_count=slot_count, num_steps=num_steps, device=device
        )
        self.slot_count = self.state.encoder.slot_count
        self.device = device
        self.pop_size = pop_size
        self.sigma = sigma
        self.rank = rank
        self.variance_weight = variance_weight
        self.prompt_alignment_weight = validate_prompt_alignment_weight(
            prompt_alignment_weight
        )
        self.eval_batch_size = eval_batch_size
        self.use_amp = use_amp and torch.device(device).type == "cuda"

        self.workspace = self.state.workspace
        self.encoder = self.state.encoder
        self.latent_loop = self.state.latent_loop
        self.latent_loop.model.eval()
        self.tokenizer = self.state.tokenizer

        self.trainable_params = list(self.state.eggroll_parameters())
        self.optimizer = self.state.create_eggroll_optimizer(lr)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_params)

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
        perturbations: list[SignedPerturbationSet],
        teacher_state: torch.Tensor,
        eos_token_id: int | None = None,
        context_prefix: PreparedQwenPrefix | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate several perturbed candidates in each frozen-model pass."""
        batch_size = len(perturbations)
        signs = [candidate.sign for candidate in perturbations]
        matrix_directions = lambda index: self._matrix_directions(
            perturbations, index
        )
        expected_token_width = self.encoder.projection.weight.shape[1]
        if (
            token_embeddings.ndim != 3
            or token_embeddings.shape[0] != 1
            or token_embeddings.shape[1] < 1
            or token_embeddings.shape[2] != expected_token_width
        ):
            raise ValueError(
                "expected encoder token embeddings shape "
                f"(1, tokens, {expected_token_width}) with nonempty tokens, "
                f"actual {tuple(token_embeddings.shape)}"
            )
        shared_token_embeddings = token_embeddings.squeeze(0)

        projected = factorized_linear(
            shared_token_embeddings,
            self.encoder.projection.weight,
            matrix_directions(0),
            signs,
            bias=self.encoder.projection.bias,
        )
        attention_projection = factorized_linear(
            projected,
            self.encoder.slot_queries,
            matrix_directions(1),
            signs,
        )
        attn_log_temp = self.encoder.attn_log_temp
        scale = attn_log_temp.exp().expand(batch_size).view(batch_size, 1, 1)
        attn_logits = attention_projection.transpose(1, 2) / scale
        attn_weights = torch.softmax(attn_logits, dim=-1)
        slots = torch.bmm(attn_weights, projected)

        proj_norm_weight = self.latent_loop.proj_norm.weight.expand(batch_size, -1)
        proj_norm_bias = self.latent_loop.proj_norm.bias.expand(batch_size, -1)
        layer_norm_weight = self.latent_loop.layer_norm.weight.expand(batch_size, -1)
        layer_norm_bias = self.latent_loop.layer_norm.bias.expand(batch_size, -1)

        loop_slots = slots
        tap_adapter = self.latent_loop.tap_adapter
        if context_prefix is None:
            context_prefix = tap_adapter.prepare_prefix(context_embeds)
        for _ in range(SUBJECT_LATENT_RUNS_PER_ANSWER):
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
                transformed = factorized_linear(
                    slot_hidden,
                    self.latent_loop.projection.weight,
                    matrix_directions(2),
                    signs,
                    bias=self.latent_loop.projection.bias,
                )
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

        answer_objective = prompt_aligned_answer_objective(
            self.latent_loop.model,
            loop_slots,
            answer_ids,
            eos_token_id,
            teacher_state=teacher_state,
        )
        lm_loss = answer_objective.language_model_loss
        prompt_alignment_penalty = (
            self.prompt_alignment_weight * answer_objective.prompt_alignment_loss
        )

        variance = torch.stack(
            [post_loop_slot_variance(candidate_slots) for candidate_slots in loop_slots]
        )
        collapse_penalty = slot_variance_penalty(loop_slots)
        fitness = (
            -lm_loss
            - prompt_alignment_penalty
            - self.variance_weight * collapse_penalty
        )
        return fitness, lm_loss, variance

    @staticmethod
    def _matrix_directions(
        candidates: list[SignedPerturbationSet],
        parameter_index: int,
    ) -> list[MatrixFactors]:
        directions = [candidate.directions[parameter_index] for candidate in candidates]
        if not all(isinstance(direction, MatrixFactors) for direction in directions):
            raise TypeError("matrix parameter received non-matrix direction")
        return directions  # type: ignore[return-value]


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
            teacher_state = prompt_teacher_state(
                self.latent_loop.model,
                prepared.context_input_ids.to(self.device),
            )

        base_seed = int(torch.randint(0, 2**31, (1,)).item())

        fitnesses = []
        lm_losses = []
        variances = []

        candidates = []
        for pair_index in range(self.pop_size // 2):
            candidates.extend(
                sample_antithetic_pair(
                    self.trainable_params,
                    seed=base_seed + pair_index,
                    sigma=self.sigma,
                    rank=self.rank,
                )
            )

        with torch.inference_mode(), torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            for start in range(0, self.pop_size, self.eval_batch_size):
                stop = min(start + self.eval_batch_size, self.pop_size)
                perturbations = candidates[start:stop]

                batch_fitness, batch_losses, batch_variances = (
                    self._forward_fitness_batch(
                        token_embeddings,
                        context_embeds,
                        answer_ids,
                        perturbations,
                        teacher_state,
                        eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                        context_prefix=context_prefix,
                    )
                )
                fitnesses.extend(batch_fitness.tolist())
                lm_losses.extend(batch_losses.tolist())
                variances.extend(batch_variances.tolist())

        apply_factorized_update(
            self.trainable_params,
            self.optimizer,
            base_seed=base_seed,
            fitnesses=fitnesses,
            sigma=self.sigma,
            rank=self.rank,
        )

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
