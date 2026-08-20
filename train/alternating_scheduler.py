"""Fixed-budget scheduler for alternating optimizer engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from train.alternating_config import validate_scheduler_config
from train.training_results import ExperimentPosition


ResultT = TypeVar("ResultT")
EvaluationT = TypeVar("EvaluationT")

PHASE_BOUNDARY = "phase"
EPOCH_BOUNDARY = "epoch"
PARTIAL_PHASE_BOUNDARY = "partial_phase"


@dataclass(frozen=True)
class EvaluationRecord(Generic[EvaluationT]):
    """One evaluation result and every boundary reached at its position."""

    result: EvaluationT
    position: ExperimentPosition
    boundaries: frozenset[str]


class TrainingEngine(Protocol[ResultT]):
    """Minimal update interface used by the scheduler."""

    def train_step(self, example: object, position: ExperimentPosition) -> ResultT:
        """Apply one update and return its method-specific result."""


class PhaseEvaluator(Protocol[EvaluationT]):
    """Evaluate the shared model at a completed phase boundary."""

    def evaluate(self, position: ExperimentPosition) -> EvaluationT:
        """Return the evaluation labeled with the completed phase position."""


class FixedBudgetScheduler:
    """Alternate Eggroll and gradient updates after equal completed-step budgets."""

    def __init__(
        self,
        *,
        phase_steps: int,
        eggroll_engine: TrainingEngine[ResultT],
        gradient_engine: TrainingEngine[ResultT],
        evaluator: PhaseEvaluator[EvaluationT] | None = None,
    ) -> None:
        validate_scheduler_config(phase_steps=phase_steps, epochs=1)
        self._phase_steps = phase_steps
        self._eggroll_engine = eggroll_engine
        self._gradient_engine = gradient_engine
        self._evaluator = evaluator
        self._completed_steps = 0
        self._evaluation_results: list[EvaluationRecord[EvaluationT]] = []

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
        global_step = self._completed_steps + 1
        phase_index = self._completed_steps // self._phase_steps
        phase_step = self._completed_steps % self._phase_steps + 1
        is_eggroll_phase = phase_index % 2 == 0
        update_method = "eggroll" if is_eggroll_phase else "gradient"
        engine = self._eggroll_engine if is_eggroll_phase else self._gradient_engine
        position = ExperimentPosition(
            update_method=update_method,
            cycle=phase_index + 1,
            global_step=global_step,
            epoch=epoch,
            example_position=example_position,
            phase_step=phase_step,
        )
        result = engine.train_step(example, position)
        self._completed_steps = global_step
        if phase_step == self._phase_steps:
            self._evaluate_at_boundary(position, PHASE_BOUNDARY)
        return result

    def evaluate_epoch_boundary(self, position: ExperimentPosition) -> None:
        """Evaluate after an epoch, merging its label with a phase boundary."""
        self._evaluate_at_boundary(position, EPOCH_BOUNDARY)

    def evaluate_final_partial_phase(self, position: ExperimentPosition) -> None:
        """Evaluate an unfinished final phase without re-evaluating shared boundaries."""
        if position.phase_step == self._phase_steps:
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
