from __future__ import annotations

from io import StringIO
import json
from pathlib import Path

import pytest

from train import run_alternating
from train.alternating_evaluation import DEFAULT_EVAL_PROBLEM_COUNT
from train.alternating_scheduler import EvaluationRecord
from train.eggroll_trainer import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_LR as DEFAULT_EGGROLL_LR,
    DEFAULT_NUM_STEPS,
    DEFAULT_POP_SIZE,
    DEFAULT_RANK,
    DEFAULT_SIGMA,
    DEFAULT_VARIANCE_WEIGHT,
)
from train.trainer import DEFAULT_LR as DEFAULT_GRADIENT_LR
from train.training_results import EvaluationResult, ExperimentPosition, StepResult
from workspace.concept_slots import DEFAULT_SLOT_COUNT


def _position() -> ExperimentPosition:
    return ExperimentPosition(
        update_method="gradient",
        cycle=4,
        global_step=27,
        epoch=2,
        example_position=6,
        phase_step=3,
    )


def test_training_progress_record_has_exact_comparable_content() -> None:
    record = run_alternating._training_record(
        StepResult(
            position=_position(),
            language_model_loss=1.25,
            total_objective=1.75,
            regularizer_loss=0.5,
            shared_variance=0.4,
        )
    )

    assert record == {
        "record_type": "training",
        "update_method": "gradient",
        "cycle": 4,
        "global_step": 27,
        "epoch": 2,
        "example_position": 6,
        "phase_step": 3,
        "language_model_loss": 1.25,
        "shared_variance": 0.4,
    }


def test_evaluation_progress_record_has_sorted_boundaries_and_exact_metrics() -> None:
    position = _position()
    record = run_alternating._evaluation_record(
        EvaluationRecord(
            result=EvaluationResult(
                position=position,
                language_model_loss=0.75,
                shared_variance=0.6,
                answer_exact_match=0.25,
            ),
            position=position,
            boundaries=frozenset({"phase", "epoch"}),
        )
    )

    assert record == {
        "record_type": "evaluation",
        "update_method": "gradient",
        "cycle": 4,
        "global_step": 27,
        "epoch": 2,
        "example_position": 6,
        "phase_step": 3,
        "boundaries": ["epoch", "phase"],
        "language_model_loss": 0.75,
        "shared_variance": 0.6,
        "answer_exact_match": 0.25,
    }


def test_checkpoint_progress_record_has_exact_paths() -> None:
    assert run_alternating._checkpoint_record(
        [Path("runs/phase-27.pt"), Path("runs/epoch-2.pt")]
    ) == {
        "record_type": "checkpoint",
        "paths": ["runs\\phase-27.pt", "runs\\epoch-2.pt"],
    }


def test_progress_record_is_written_as_one_compact_json_line() -> None:
    output = StringIO()

    run_alternating._write_progress_record(
        {"record_type": "checkpoint", "paths": ["phase-27.pt"]}, output
    )

    assert output.getvalue().count("\n") == 1
    assert json.loads(output.getvalue()) == {
        "record_type": "checkpoint",
        "paths": ["phase-27.pt"],
    }


def test_alternating_command_defaults() -> None:
    args = run_alternating._parse_args([])

    assert args.epochs == 5
    assert args.phase_steps == 500
    assert args.slot_count == DEFAULT_SLOT_COUNT
    assert args.num_steps == DEFAULT_NUM_STEPS
    assert args.gradient_lr == DEFAULT_GRADIENT_LR
    assert args.eggroll_lr == DEFAULT_EGGROLL_LR
    assert args.pop_size == DEFAULT_POP_SIZE
    assert args.sigma == DEFAULT_SIGMA
    assert args.rank == DEFAULT_RANK
    assert args.variance_weight == DEFAULT_VARIANCE_WEIGHT
    assert args.eval_batch_size == DEFAULT_EVAL_BATCH_SIZE
    assert args.use_amp is False
    assert args.eval_problem_count == DEFAULT_EVAL_PROBLEM_COUNT
    assert args.problem_count is None
    assert args.log_every == 10
    assert args.save_dir == "checkpoints/alternating/"
    assert args.resume is None


def test_alternating_command_accepts_overrides() -> None:
    args = run_alternating._parse_args(
        [
            "--epochs", "7",
            "--phase-steps", "25",
            "--slot-count", "6",
            "--num-steps", "4",
            "--gradient-lr", "0.03",
            "--eggroll-lr", "0.04",
            "--pop-size", "32",
            "--sigma", "0.05",
            "--rank", "3",
            "--variance-weight", "0.7",
            "--eval-batch-size", "4",
            "--amp",
            "--eval-problem-count", "16",
            "--problem-count", "100",
            "--device", "cpu",
            "--log-every", "8",
            "--save-dir", "runs/alternating",
            "--resume", "latest",
        ]
    )

    assert vars(args) == {
        "epochs": 7,
        "phase_steps": 25,
        "slot_count": 6,
        "num_steps": 4,
        "gradient_lr": 0.03,
        "eggroll_lr": 0.04,
        "pop_size": 32,
        "sigma": 0.05,
        "rank": 3,
        "variance_weight": 0.7,
        "eval_batch_size": 4,
        "use_amp": True,
        "eval_problem_count": 16,
        "problem_count": 100,
        "device": "cpu",
        "log_every": 8,
        "save_dir": "runs/alternating",
        "resume": "latest",
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--epochs", "0"], "epochs must be at least 1"),
        (["--phase-steps", "0"], "phase_steps must be at least 1"),
    ],
)
def test_invalid_schedule_is_rejected_before_production_loading(
    arguments: list[str], message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_production_loading(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("production loading was attempted")

    monkeypatch.setattr(
        "eval.stream.generators.gsm8k._load_split", reject_production_loading
    )
    monkeypatch.setattr(
        "train.training_state.TrainingState.__init__", reject_production_loading
    )

    with pytest.raises(ValueError, match=message):
        run_alternating.main(arguments)
