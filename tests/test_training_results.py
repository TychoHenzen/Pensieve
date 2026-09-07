from dataclasses import FrozenInstanceError

import pytest

from train.training_results import EvaluationResult, ExperimentPosition, StepResult


def test_step_result_keeps_each_metric_and_position_distinct() -> None:
    position = ExperimentPosition(
        update_method="eggroll",
        cycle=2,
        global_step=17,
        epoch=3,
        example_position=8,
        phase_step=4,
    )
    result = StepResult(
        position=position,
        language_model_loss=1.25,
        total_objective=-0.75,
        regularizer_loss=-2.0,
        shared_variance=0.5,
    )

    assert result.position.update_method == "eggroll"
    assert result.position.cycle == 2
    assert result.position.global_step == 17
    assert result.position.epoch == 3
    assert result.position.example_position == 8
    assert result.position.phase_step == 4
    assert result.language_model_loss == 1.25
    assert result.total_objective == -0.75
    assert result.regularizer_loss == -2.0
    assert result.shared_variance == 0.5
    assert result.language_model_loss != result.total_objective
    assert result.total_objective != result.regularizer_loss


def test_evaluation_result_keeps_language_model_loss_and_variance_distinct() -> None:
    result = EvaluationResult(
        position=ExperimentPosition("gradient", 1, 9, 2, 5, 3),
        language_model_loss=0.8,
        shared_variance=0.3,
        answer_exact_match=0.6,
    )

    assert result.language_model_loss == 0.8
    assert result.shared_variance == 0.3
    assert result.answer_exact_match == 0.6
    assert result.position.global_step == 9


def test_result_records_are_immutable() -> None:
    position = ExperimentPosition("gradient", 0, 1, 0, 1, 1)
    result = StepResult(position, 1.0, 1.2, 0.2, 0.4)

    with pytest.raises(FrozenInstanceError):
        result.total_objective = 2.0
    with pytest.raises(FrozenInstanceError):
        position.global_step = 2
