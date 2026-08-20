"""Fixed-budget scheduler for alternating optimizer engines."""

from __future__ import annotations

from typing import Protocol, TypeVar

from train.alternating_config import validate_scheduler_config
from train.training_results import ExperimentPosition


ResultT = TypeVar("ResultT")
EvaluationT = TypeVar("EvaluationT")


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
        self._evaluation_results: list[EvaluationT] = []

    @property
    def evaluation_results(self) -> tuple[EvaluationT, ...]:
        """Return evaluations from completed phases in production order."""
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
        if phase_step == self._phase_steps and self._evaluator is not None:
            self._evaluation_results.append(self._evaluator.evaluate(position))
        return result
