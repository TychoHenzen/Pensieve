"""Train the latent-core subject's learned components on Calc-ASDiv_A.

The frozen Qwen backbone and the frozen MiniLM sentence encoder are
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
from train.vicreg import (
    DEFAULT_EMA_DECAY,
    EMAProjection,
    post_loop_slot_variance,
    slot_variance_penalty,
)
from workspace.concept_slots import DEFAULT_SLOT_COUNT


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


class LatentCoreTrainer:
    """Train the encoder and latent loop wrappers on Stage 0 examples."""

    def __init__(
        self,
        slot_count: int = DEFAULT_SLOT_COUNT,
        num_steps: int = DEFAULT_NUM_STEPS,
        lr: float = DEFAULT_LR,
        variance_weight: float = 1.0,
        prompt_alignment_weight: float = DEFAULT_PROMPT_ALIGNMENT_WEIGHT,
        device: str = "cpu",
        ema_decay: float = DEFAULT_EMA_DECAY,
        state: TrainingState | None = None,
    ) -> None:
        self.state = state or TrainingState(
            slot_count=slot_count, num_steps=num_steps, device=device
        )
        self.slot_count = self.state.encoder.slot_count
        self.device = device
        self.variance_weight = variance_weight
        self.prompt_alignment_weight = validate_prompt_alignment_weight(
            prompt_alignment_weight
        )
        self.workspace = self.state.workspace
        self.encoder = self.state.encoder
        self.latent_loop = self.state.latent_loop
        self.tokenizer = self.state.tokenizer
        self.ema_projection = EMAProjection(self.latent_loop.projection, decay=ema_decay)

        self.trainable_params = list(self.state.parameters())
        self.optimizer = self.state.create_gradient_optimizer(lr)

    def trainable_param_count(self) -> int:
        """Number of trainable scalar parameters across all trainable modules."""
        return sum(p.numel() for p in self.trainable_params)

    def train_step(
        self,
        question: str,
        answer: str,
        position: ExperimentPosition | None = None,
    ) -> StepResult:
        """Run one decoder-aligned, teacher-forced training step.

        Slots predict the first answer token. Answer prefixes predict the
        remaining tokens and EOS, matching ``SlotDecoder.decode``.
        """
        self.optimizer.zero_grad()

        prepared = prepare_training_example(self.tokenizer, question, answer)
        teacher_state = prompt_teacher_state(
            self.latent_loop.model,
            prepared.context_input_ids.to(self.device),
        )
        context_embeds = self.latent_loop.embed_tokens(prepared.context_input_ids)

        slots = self.encoder.encode(question)
        self.workspace.write_slots(slots)
        for _ in range(SUBJECT_LATENT_RUNS_PER_ANSWER):
            self.latent_loop.run(self.workspace, context_embeds=context_embeds)
        loop_slots = self.workspace.read_slots()

        answer_objective = prompt_aligned_answer_objective(
            self.latent_loop.model,
            loop_slots,
            prepared.answer_ids.to(self.device),
            getattr(self.tokenizer, "eos_token_id", None),
            teacher_state=teacher_state,
        )
        lm_loss = answer_objective.language_model_loss.mean()
        prompt_alignment_penalty = (
            self.prompt_alignment_weight
            * answer_objective.prompt_alignment_loss.mean()
        )
        collapse_penalty = self.variance_weight * slot_variance_penalty(
            loop_slots
        ).mean()
        total_objective = lm_loss + prompt_alignment_penalty + collapse_penalty
        total_objective.backward()
        self.optimizer.step()
        self.ema_projection.update()

        if position is None:
            position = ExperimentPosition(
                update_method="gradient",
                cycle=0,
                global_step=0,
                epoch=0,
                example_position=0,
                phase_step=0,
            )

        return StepResult(
            position=position,
            language_model_loss=lm_loss.item(),
            total_objective=total_objective.item(),
            regularizer_loss=(prompt_alignment_penalty + collapse_penalty).item(),
            shared_variance=post_loop_slot_variance(loop_slots).item(),
            optimizer_call_count=position.optimizer_call_count,
        )

    def train_epoch(
        self,
        dataset: list[tuple[str, str]],
        on_step: object | None = None,
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
            total_loss += result.total_objective
            total_var += result.shared_variance
            covariance = self.workspace.collapse_stats()["covariance"]
            total_cov += covariance
            min_var = min(min_var, result.shared_variance)
            max_var = max(max_var, result.shared_variance)
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
