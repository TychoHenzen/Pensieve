"""Immutable records for comparable training and evaluation measurements."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentPosition:
    """The location and update method that produced a measurement."""

    update_method: str
    cycle: int
    global_step: int
    epoch: int
    example_position: int
    phase_step: int


@dataclass(frozen=True)
class StepResult:
    """Measurements from one parameter update.

    ``total_objective`` is method-specific. ``language_model_loss`` remains
    directly comparable between update methods. ``regularizer_loss`` contains
    the sum of objective terms other than language-model loss.
    """

    position: ExperimentPosition
    language_model_loss: float
    total_objective: float
    regularizer_loss: float
    shared_variance: float


@dataclass(frozen=True)
class EvaluationResult:
    """Measurements from the unperturbed shared model at one position."""

    position: ExperimentPosition
    language_model_loss: float
    shared_variance: float
    answer_exact_match: float
