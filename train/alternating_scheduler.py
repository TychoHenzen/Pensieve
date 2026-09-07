"""Average-variance hysteresis scheduler for alternating optimizer engines."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from train.alternating_config import (
    DEFAULT_VARIANCE_LOWER_THRESHOLD,
    DEFAULT_VARIANCE_UPPER_THRESHOLD,
    validate_scheduler_config,
)
from train.training_results import ExperimentPosition

PHASE_BOUNDARY = "phase"
EPOCH_BOUNDARY = "epoch"
PARTIAL_PHASE_BOUNDARY = "partial_phase"


@dataclass(frozen=True)
class EvaluationRecord[EvaluationT]:
    """One evaluation result and every boundary reached at its position."""

    result: EvaluationT
    position: ExperimentPosition
    boundaries: frozenset[str]


class TrainingEngine[ResultT](Protocol):
    """Minimal update interface used by the scheduler."""

    @property
    def max_consumed_records(self) -> int:
        """Largest contiguous record batch this engine can consume per call."""
        ...

    def train_step(self, example: object, position: ExperimentPosition) -> ResultT:
        """Apply one update and return its method-specific result."""
        ...


class PhaseEvaluator[EvaluationT](Protocol):
    """Evaluate the shared model at a completed phase boundary."""

    def evaluate(self, position: ExperimentPosition) -> EvaluationT:
        """Return the evaluation labeled with the completed phase position."""
        ...


class VarianceHysteresisScheduler[ResultT, EvaluationT]:
    """Select an optimizer from average slot variance over fixed windows."""

    def __init__(
        self,
        *,
        phase_steps: int,
        variance_lower_threshold: float = DEFAULT_VARIANCE_LOWER_THRESHOLD,
        variance_upper_threshold: float = DEFAULT_VARIANCE_UPPER_THRESHOLD,
        eggroll_engine: TrainingEngine[ResultT],
        gradient_engine: TrainingEngine[ResultT],
        evaluator: PhaseEvaluator[EvaluationT] | None = None,
    ) -> None:
        validate_scheduler_config(
            phase_steps=phase_steps,
            epochs=1,
            variance_lower_threshold=variance_lower_threshold,
            variance_upper_threshold=variance_upper_threshold,
        )
        self._phase_steps = phase_steps
        self._variance_lower_threshold = variance_lower_threshold
        self._variance_upper_threshold = variance_upper_threshold
        self._eggroll_engine = eggroll_engine
        self._gradient_engine = gradient_engine
        self._evaluator = evaluator
        self._completed_steps = 0
        self._completed_phase_steps = 0
        self._phase_variance_sum = 0.0
        self._active_phase = "eggroll"
        self._cycle = 1
        self._optimizer_call_counts = {"eggroll": 0, "gradient": 0}
        self._evaluation_results: list[EvaluationRecord[EvaluationT]] = []

    @property
    def active_phase(self) -> str:
        return self._active_phase

    @property
    def completed_phase_steps(self) -> int:
        return self._completed_phase_steps

    @property
    def completed_steps(self) -> int:
        """Return the total number of consumed examples."""
        return self._completed_steps

    @property
    def phase_variance_sum(self) -> float:
        return self._phase_variance_sum

    @property
    def records_until_observation_boundary(self) -> int:
        """Return the positive record capacity left in the current window."""
        return self._phase_steps - self._completed_phase_steps

    def restore(
        self,
        *,
        active_phase: str,
        completed_steps: int,
        completed_phase_steps: int,
        phase_variance_sum: float,
    ) -> None:
        """Restore the controller state recorded at a checkpoint boundary."""
        if active_phase not in {"eggroll", "gradient"}:
            raise ValueError("active_phase must be eggroll or gradient")
        if completed_steps < 0:
            raise ValueError("completed_steps must be non-negative")
        if not 0 <= completed_phase_steps < self._phase_steps:
            raise ValueError("completed_phase_steps must fit inside the variance window")
        if not math.isfinite(phase_variance_sum) or phase_variance_sum < 0.0:
            raise ValueError("phase_variance_sum must be finite and non-negative")
        self._active_phase = active_phase
        self._completed_steps = completed_steps
        self._completed_phase_steps = completed_phase_steps
        self._phase_variance_sum = phase_variance_sum
        self._cycle = completed_steps // self._phase_steps + 1

    @property
    def evaluation_results(self) -> tuple[EvaluationRecord[EvaluationT], ...]:
        """Return evaluations and their phase, epoch, or partial-phase labels."""
        return tuple(self._evaluation_results)

    def train_step(
        self,
        example: object,
        *,
        epoch: int,
        example_position: int,
    ) -> ResultT:
        """Train on one example using the engine selected for its next step."""
        return self._train_records((example,), epoch=epoch, example_position=example_position)

    def train_records(
        self,
        examples: Sequence[object],
        *,
        epoch: int,
        example_position: int,
        records_until_logging_boundary: int | None = None,
    ) -> ResultT:
        """Train on the next contiguous records without crossing active boundaries.

        Gradient engines always receive one record. EGGROLL engines receive at
        most their configured batch size, the records left in the observation
        window, and the records left before a progress boundary.
        """
        if not examples:
            raise ValueError("train_records requires at least one unconsumed example")
        if records_until_logging_boundary is not None and records_until_logging_boundary < 1:
            raise ValueError("records_until_logging_boundary must be at least 1")
        return self._train_records(
            examples,
            epoch=epoch,
            example_position=example_position,
            records_until_logging_boundary=records_until_logging_boundary,
        )

    def _train_records(
        self,
        examples: Sequence[object],
        *,
        epoch: int,
        example_position: int,
        records_until_logging_boundary: int | None = None,
    ) -> ResultT:
        update_method = self._active_phase
        engine = self._eggroll_engine if update_method == "eggroll" else self._gradient_engine
        max_consumed_records = getattr(engine, "max_consumed_records", 1)
        if (
            isinstance(max_consumed_records, bool)
            or not isinstance(max_consumed_records, int)
            or max_consumed_records < 1
        ):
            raise ValueError("training engine max_consumed_records must be a positive integer")
        record_count = min(
            len(examples),
            self.records_until_observation_boundary,
            max_consumed_records,
        )
        if records_until_logging_boundary is not None:
            record_count = min(record_count, records_until_logging_boundary)
        if record_count < 1:
            raise ValueError("scheduler could not select a positive record batch")
        global_step = self._completed_steps + record_count
        phase_step = self._completed_phase_steps + record_count
        optimizer_call_count = self._optimizer_call_counts[update_method] + 1
        position = ExperimentPosition(
            update_method=update_method,
            cycle=self._cycle,
            global_step=global_step,
            epoch=epoch,
            example_position=example_position + record_count - 1,
            phase_step=phase_step,
            optimizer_call_count=optimizer_call_count,
            consumed_record_count=record_count,
        )
        engine_input: object = examples[0] if record_count == 1 else tuple(examples[:record_count])
        result = engine.train_step(engine_input, position)
        variance = getattr(result, "shared_variance", None)
        consumed_record_count = getattr(result, "consumed_record_count", 1)
        if (
            isinstance(variance, bool)
            or not isinstance(variance, (int, float))
            or not math.isfinite(float(variance))
            or float(variance) < 0.0
        ):
            raise ValueError("training result shared_variance must be finite and non-negative")
        if (
            isinstance(consumed_record_count, bool)
            or not isinstance(consumed_record_count, int)
            or consumed_record_count != record_count
        ):
            raise ValueError("training result consumed_record_count must match the scheduled batch")
        self._completed_steps = global_step
        self._completed_phase_steps = phase_step
        self._optimizer_call_counts[update_method] = optimizer_call_count
        self._phase_variance_sum += float(variance) * consumed_record_count
        if phase_step == self._phase_steps:
            average_variance = self._phase_variance_sum / self._phase_steps
            if self._active_phase == "eggroll" and average_variance >= self._variance_upper_threshold:
                self._active_phase = "gradient"
            elif self._active_phase == "gradient" and average_variance <= self._variance_lower_threshold:
                self._active_phase = "eggroll"
            self._completed_phase_steps = 0
            self._phase_variance_sum = 0.0
            self._cycle += 1
            self._evaluate_at_boundary(position, PHASE_BOUNDARY)
        return result

    def evaluate_epoch_boundary(self, position: ExperimentPosition) -> None:
        """Evaluate after an epoch, merging its label with a phase boundary."""
        self._evaluate_at_boundary(position, EPOCH_BOUNDARY)

    def evaluate_final_partial_phase(self, position: ExperimentPosition) -> None:
        """Evaluate an unfinished final phase without re-evaluating shared boundaries."""
        if self._completed_phase_steps == 0:
            return
        self._evaluate_at_boundary(position, PARTIAL_PHASE_BOUNDARY)

    def _evaluate_at_boundary(self, position: ExperimentPosition, boundary: str) -> None:
        if self._evaluator is None:
            return
        if self._evaluation_results and self._evaluation_results[-1].position == position:
            previous = self._evaluation_results[-1]
            self._evaluation_results[-1] = EvaluationRecord(
                result=previous.result,
                position=position,
                boundaries=previous.boundaries | {boundary},
            )
            return
        self._evaluation_results.append(
            EvaluationRecord(
                result=self._evaluator.evaluate(position),
                position=position,
                boundaries=frozenset({boundary}),
            )
        )
