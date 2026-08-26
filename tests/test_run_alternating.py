from __future__ import annotations

import copy
import hashlib
from io import StringIO
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from train import run_alternating
from train.alternating_evaluation import DEFAULT_EVAL_PROBLEM_COUNT
from train.alternating_checkpoint import (
    CheckpointSchedule,
    build_alternating_checkpoint,
    stage0_parameter_paths,
)
from train.alternating_config import (
    DEFAULT_VARIANCE_LOWER_THRESHOLD,
    DEFAULT_VARIANCE_UPPER_THRESHOLD,
)
from train.alternating_scheduler import EvaluationRecord
from train.eggroll_trainer import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_FITNESS_BATCH_SIZE,
    DEFAULT_LR as DEFAULT_EGGROLL_LR,
    DEFAULT_NUM_STEPS,
    DEFAULT_POP_SIZE,
    DEFAULT_RANK,
    DEFAULT_SIGMA,
    DEFAULT_VARIANCE_WEIGHT,
)
from train.training_state import EGGROLL_PARAMETER_PATHS, GRADIENT_PARAMETER_PATHS
from train.answer_objective import DEFAULT_PROMPT_ALIGNMENT_WEIGHT
from train.trainer import DEFAULT_LR as DEFAULT_GRADIENT_LR
from train.training_results import EvaluationResult, ExperimentPosition, StepResult
from workspace.concept_slots import DEFAULT_SLOT_COUNT
from test_stage0_checkpoint_resume import RUN_CONFIG, _metadata


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
    assert args.phase_steps == 50
    assert (args.variance_lower_threshold, args.variance_upper_threshold) == (0.01, 0.02)
    assert args.variance_lower_threshold == DEFAULT_VARIANCE_LOWER_THRESHOLD
    assert args.variance_upper_threshold == DEFAULT_VARIANCE_UPPER_THRESHOLD
    assert args.slot_count == DEFAULT_SLOT_COUNT
    assert args.num_steps == DEFAULT_NUM_STEPS
    assert args.gradient_lr == DEFAULT_GRADIENT_LR
    assert args.eggroll_lr == DEFAULT_EGGROLL_LR
    assert args.pop_size == DEFAULT_POP_SIZE
    assert args.sigma == DEFAULT_SIGMA
    assert args.rank == DEFAULT_RANK
    assert args.variance_weight == DEFAULT_VARIANCE_WEIGHT
    assert args.prompt_alignment_weight == DEFAULT_PROMPT_ALIGNMENT_WEIGHT
    assert args.eval_batch_size == DEFAULT_EVAL_BATCH_SIZE
    assert args.fitness_batch_size == DEFAULT_FITNESS_BATCH_SIZE
    assert args.use_amp is False
    assert args.eval_problem_count == DEFAULT_EVAL_PROBLEM_COUNT
    assert args.problem_count is None
    assert args.log_every == 50
    assert args.save_dir == "checkpoints/alternating/"
    assert args.resume is None
    assert args.stability_report is None


# covers: train/eggroll-execution :: EGGROLL uses safe Stage 0 defaults :: Default stabilized configuration
def test_eggroll_defaults_use_the_stabilized_sgd_configuration() -> None:
    args = run_alternating._parse_args([])

    assert (args.pop_size, args.eval_batch_size, args.rank) == (128, 8, 4)
    assert args.sigma == 0.001
    assert args.eggroll_lr == 0.1


# covers: train/eggroll-execution :: EGGROLL uses safe Stage 0 defaults :: Invalid stabilized configuration
def test_invalid_eggroll_values_fail_during_argument_parsing() -> None:
    invalid_values = [
        ("--pop-size", "1", "pop_size must be an even integer of at least 2"),
        ("--pop-size", "3", "pop_size must be an even integer of at least 2"),
        ("--rank", "0", "rank must be at least 1"),
        ("--sigma", "0", "sigma must be finite and positive"),
        ("--sigma", "nan", "sigma must be finite and positive"),
        ("--eggroll-lr", "inf", "lr must be finite and positive"),
        ("--eval-batch-size", "1", "eval_batch_size must be at least 2"),
    ]
    observed_messages: list[str] = []

    for option, value, expected_message in invalid_values:
        with pytest.raises(ValueError) as error:
            run_alternating._parse_args([option, value])
        observed_messages.append(str(error.value))

    assert observed_messages == [item[2] for item in invalid_values]


def test_alternating_command_accepts_overrides() -> None:
    args = run_alternating._parse_args(
        [
            "--epochs", "7",
            "--phase-steps", "25",
            "--variance-lower-threshold", "0.03",
            "--variance-upper-threshold", "0.08",
            "--slot-count", "6",
            "--num-steps", "4",
            "--gradient-lr", "0.03",
            "--eggroll-lr", "0.04",
            "--pop-size", "32",
            "--sigma", "0.05",
            "--rank", "3",
            "--variance-weight", "0.7",
            "--prompt-alignment-weight", "0.6",
            "--eval-batch-size", "4",
            "--fitness-batch-size", "6",
            "--amp",
            "--eval-problem-count", "16",
            "--problem-count", "100",
            "--device", "cpu",
            "--log-every", "8",
            "--save-dir", "runs/alternating",
            "--resume", "latest",
            "--stability-report", "report.json",
        ]
    )

    assert vars(args) == {
        "epochs": 7,
        "phase_steps": 25,
        "variance_lower_threshold": 0.03,
        "variance_upper_threshold": 0.08,
        "slot_count": 6,
        "num_steps": 4,
        "gradient_lr": 0.03,
        "eggroll_lr": 0.04,
        "pop_size": 32,
        "sigma": 0.05,
        "rank": 3,
        "variance_weight": 0.7,
        "prompt_alignment_weight": 0.6,
        "eval_batch_size": 4,
        "fitness_batch_size": 6,
        "use_amp": True,
        "eval_problem_count": 16,
        "problem_count": 100,
        "device": "cpu",
        "log_every": 8,
        "save_dir": "runs/alternating",
        "resume": "latest",
        "stability_report": "report.json",
    }


def test_run_config_records_every_typed_alternating_setting(tmp_path: Path) -> None:
    report_path = tmp_path / "stability.json"
    report_path.write_text('{"status":"passed"}', encoding="utf-8")
    args = run_alternating._parse_args(
        [
            "--problem-count", "3", "--eval-problem-count", "2", "--phase-steps", "2",
            "--sigma", "0.02", "--eggroll-lr", "0.001",
            "--stability-report", str(report_path),
        ]
    )

    expected = copy.deepcopy(RUN_CONFIG)
    expected["gradient_optimizer"] = {
        "type": "Adam",
        "momentum": 0.0,
        "learning_rate": DEFAULT_GRADIENT_LR,
        "parameter_paths": list(GRADIENT_PARAMETER_PATHS),
    }
    expected["eggroll_optimizer"] = {
        "type": "SGD",
        "momentum": 0.0,
        "learning_rate": 0.001,
        "parameter_paths": list(EGGROLL_PARAMETER_PATHS),
    }
    expected["eggroll_population"].update(
        {
            "variance_lower_threshold": 0.01,
            "variance_upper_threshold": 0.02,
            "fitness_batch_size": DEFAULT_FITNESS_BATCH_SIZE,
        }
    )
    expected["stability_report_identity"] = hashlib.sha256(
        b'{"status":"passed"}'
    ).hexdigest()
    assert run_alternating._run_config(args) == expected


def test_incompatible_run_config_is_rejected_before_model_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _metadata()
    saved_config = copy.deepcopy(fixture["run_config"])
    saved_config["model_shape"]["slot_count"] = 8
    fixture["run_config"] = saved_config
    checkpoint = SimpleNamespace(metadata=fixture)
    train_selection = object()
    held_out_selection = object()
    stage0_dataset = SimpleNamespace(
        train_selection=train_selection,
        held_out_selection=held_out_selection,
    )
    constructions: list[object] = []

    monkeypatch.setattr(run_alternating, "load_checkpoint", lambda _path: checkpoint)
    monkeypatch.setattr(run_alternating, "load_stage0_dataset", lambda: stage0_dataset)
    monkeypatch.setattr(run_alternating, "training_examples", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        run_alternating,
        "_selection_metadata",
        lambda selection, _count: fixture["selections"][
            "train" if selection is train_selection else "held_out"
        ],
    )
    monkeypatch.setattr(
        run_alternating, "_runtime_identity", lambda: fixture["identity"]["runtime"]
    )
    monkeypatch.setattr(
        run_alternating,
        "TrainingState",
        lambda *_args: constructions.append(object()),
    )

    with pytest.raises(ValueError, match=r"\$\.run_config\.model_shape"):
        run_alternating.main(
            [
                "--resume",
                str(tmp_path / "phase-2.ckpt"),
                "--problem-count",
                "3",
                "--eval-problem-count",
                "2",
                "--phase-steps",
                "2",
            ]
        )

    assert constructions == []


def test_partial_window_resume_requires_saved_variance_sum() -> None:
    fixture = _metadata(epoch_boundary=True)
    del fixture["metrics"]["phase_variance_sum"]

    with pytest.raises(ValueError, match=r"\$\.metrics\.phase_variance_sum"):
        run_alternating._validate_resume_metadata(
            SimpleNamespace(metadata=fixture),
            identity=fixture["identity"],
            selections=fixture["selections"],
            run_config=fixture["run_config"],
        )


def test_pre_alignment_checkpoint_cannot_resume_changed_objective() -> None:
    fixture = _metadata()
    del fixture["run_config"]["eggroll_population"]["prompt_alignment_weight"]

    with pytest.raises(
        ValueError,
        match=r"\$\.run_config\.eggroll_population",
    ):
        run_alternating._validate_resume_metadata(
            SimpleNamespace(metadata=fixture),
            identity=fixture["identity"],
            selections=fixture["selections"],
            run_config=RUN_CONFIG,
        )


@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_log_every_is_rejected_before_production_loading(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_production_loading(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("production loading was attempted")

    monkeypatch.setattr(run_alternating, "load_stage0_dataset", reject_production_loading)
    monkeypatch.setattr(run_alternating, "TrainingState", reject_production_loading)

    with pytest.raises(ValueError, match="--log-every"):
        run_alternating.main(["--log-every", value])


def test_non_integer_log_every_is_rejected_by_cli_before_production_loading(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        run_alternating, "load_stage0_dataset", lambda *_args: pytest.fail("production loading was attempted")
    )

    with pytest.raises(SystemExit):
        run_alternating.main(["--log-every", "not-an-integer"])
    assert "--log-every" in capsys.readouterr().err


@pytest.mark.parametrize("weight", ["-0.1", "nan", "inf"])
def test_invalid_prompt_alignment_weight_is_rejected_before_production_loading(
    weight: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_production_loading(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("production loading was attempted")

    monkeypatch.setattr(run_alternating, "load_stage0_dataset", reject_production_loading)
    monkeypatch.setattr(run_alternating, "TrainingState", reject_production_loading)

    with pytest.raises(ValueError, match="prompt_alignment_weight"):
        run_alternating.main(["--prompt-alignment-weight", weight])


@pytest.mark.parametrize("epochs", ["0", "-1"])
def test_invalid_epoch_counts_are_rejected_before_production_loading(
    epochs: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    production_loading_attempts: list[tuple[object, ...]] = []

    def reject_production_loading(*_args: object, **_kwargs: object) -> None:
        production_loading_attempts.append(_args)
        raise AssertionError("production loading was attempted")

    monkeypatch.setattr(run_alternating, "load_stage0_dataset", reject_production_loading)
    monkeypatch.setattr(run_alternating, "TrainingState", reject_production_loading)

    with pytest.raises(ValueError, match="epochs must be at least 1"):
        run_alternating.main(["--epochs", epochs])

    assert production_loading_attempts == []


# covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Invalid epoch count
def test_invalid_epoch_count_is_rejected_before_production_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    production_loading_attempts: list[tuple[object, ...]] = []

    def reject_production_loading(*_args: object, **_kwargs: object) -> None:
        production_loading_attempts.append(_args)
        raise AssertionError("production loading was attempted")

    monkeypatch.setattr(run_alternating, "load_stage0_dataset", reject_production_loading)
    monkeypatch.setattr(run_alternating, "TrainingState", reject_production_loading)

    with pytest.raises(ValueError, match="epochs must be at least 1"):
        run_alternating.main(["--epochs", "0"])

    assert production_loading_attempts == []


def test_invalid_phase_budget_is_rejected_before_production_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_production_loading(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("production loading was attempted")

    monkeypatch.setattr(run_alternating, "load_stage0_dataset", reject_production_loading)
    monkeypatch.setattr(run_alternating, "TrainingState", reject_production_loading)

    with pytest.raises(ValueError, match="phase_steps must be at least 1"):
        run_alternating.main(["--phase-steps", "0"])


class _BatchEngine:
    """Dependency-light engine that records every consumed input record."""

    def __init__(self, method: str, *, max_consumed_records: int, variance: float) -> None:
        self.method = method
        self.max_consumed_records = max_consumed_records
        self.variance = variance
        self.batches: list[tuple[object, ...]] = []
        self.positions: list[ExperimentPosition] = []

    def train_step(self, example: object, position: ExperimentPosition) -> StepResult:
        records = example if isinstance(example, tuple) else (example,)
        self.batches.append(records)
        self.positions.append(position)
        return StepResult(
            position=position,
            language_model_loss=1.0,
            total_objective=1.0,
            regularizer_loss=0.0,
            shared_variance=self.variance,
            consumed_record_count=len(records),
            next_example_position=position.example_position + 1,
        )


class _NoopEvaluator:
    def evaluate(self, position: ExperimentPosition) -> EvaluationResult:
        return EvaluationResult(position, 1.0, 0.01, 0.0)


def _save_nothing(
    _schedule: CheckpointSchedule,
    *,
    phase_boundary: bool,
    epoch_boundary: bool,
    evaluation_record: EvaluationRecord[EvaluationResult],
) -> tuple[Path, ...]:
    del phase_boundary, epoch_boundary, evaluation_record
    return ()


# covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Observation window ends within an epoch
def test_traversal_caps_eggroll_batches_and_continues_within_the_epoch() -> None:
    eggroll = _BatchEngine("eggroll", max_consumed_records=2, variance=0.03)
    gradient = _BatchEngine("gradient", max_consumed_records=1, variance=0.005)

    run_alternating._run_schedule(
        examples=["a", "b", "c", "d", "e"],
        epochs=1,
        phase_steps=3,
        variance_lower_threshold=0.01,
        variance_upper_threshold=0.02,
        log_every=10,
        eggroll_engine=eggroll,
        gradient_engine=gradient,
        evaluator=_NoopEvaluator(),
        save_boundary=_save_nothing,
        output=StringIO(),
    )

    assert eggroll.batches == [("a", "b"), ("c",)]
    assert gradient.batches == [("d",), ("e",)]
    assert eggroll.positions[-1] == ExperimentPosition("eggroll", 1, 3, 1, 2, 3)
    assert gradient.positions[0] == ExperimentPosition("gradient", 2, 4, 1, 3, 1)


# covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Epoch boundary within an observation window
def test_traversal_preserves_the_observation_window_across_epochs() -> None:
    eggroll = _BatchEngine("eggroll", max_consumed_records=2, variance=0.03)
    gradient = _BatchEngine("gradient", max_consumed_records=1, variance=0.005)

    run_alternating._run_schedule(
        examples=["a", "b"],
        epochs=2,
        phase_steps=3,
        variance_lower_threshold=0.01,
        variance_upper_threshold=0.02,
        log_every=10,
        eggroll_engine=eggroll,
        gradient_engine=gradient,
        evaluator=_NoopEvaluator(),
        save_boundary=_save_nothing,
        output=StringIO(),
    )

    assert eggroll.batches == [("a", "b"), ("a",)]
    assert eggroll.positions[-1] == ExperimentPosition("eggroll", 1, 3, 2, 0, 3)
    assert gradient.batches == [("b",)]
    assert gradient.positions == [ExperimentPosition("gradient", 2, 4, 2, 1, 1)]


# covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Full default run
def test_default_five_epoch_traversal_consumes_exactly_2850_ordered_examples() -> None:
    examples = list(range(570))
    eggroll = _BatchEngine("eggroll", max_consumed_records=8, variance=0.015)
    gradient = _BatchEngine("gradient", max_consumed_records=1, variance=0.005)

    run_alternating._run_schedule(
        examples=examples,
        epochs=5,
        phase_steps=50,
        log_every=50,
        eggroll_engine=eggroll,
        gradient_engine=gradient,
        evaluator=_NoopEvaluator(),
        save_boundary=_save_nothing,
        output=StringIO(),
    )

    visits = [record for batch in eggroll.batches for record in batch]
    assert visits == examples * 5
    assert len(visits) == 2850
    assert len(eggroll.batches) == 399
    assert gradient.batches == []


def test_main_builds_shared_production_run_and_executes_schedule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}
    shared_state = object()
    gradient = type("Gradient", (), {"optimizer": object()})()
    eggroll = type("Eggroll", (), {"optimizer": object()})()

    stage0_dataset = SimpleNamespace(
        train_selection=object(),
        held_out_selection=object(),
        held_out_records=lambda: ["validation-record"],
    )
    monkeypatch.setattr(run_alternating, "load_stage0_dataset", lambda: stage0_dataset)
    monkeypatch.setattr(
        run_alternating,
        "training_examples",
        lambda *_args, **_kwargs: [("q", "1")],
    )
    monkeypatch.setattr(
        run_alternating,
        "load_held_out_problems",
        lambda records, count: ["held-out"],
    )
    monkeypatch.setattr(
        run_alternating,
        "_selection_metadata",
        lambda selection, count: {
            "identity": "a" * 64,
            "split": "train" if selection is stage0_dataset.train_selection else "validation",
            "seed": 0,
            "problem_count": count,
            "ordered_item_ids": ["item-0"],
        },
    )
    monkeypatch.setattr(
        run_alternating, "TrainingState", lambda *args: shared_state
    )

    def build_gradient(**kwargs: object) -> object:
        calls["gradient_state"] = kwargs["state"]
        return gradient

    def build_eggroll(**kwargs: object) -> object:
        calls["eggroll_state"] = kwargs["state"]
        return eggroll

    monkeypatch.setattr(run_alternating, "LatentCoreTrainer", build_gradient)
    monkeypatch.setattr(run_alternating, "EggrollTrainer", build_eggroll)
    monkeypatch.setattr(run_alternating, "_Evaluator", lambda model, held: "evaluator")

    def run_schedule(**kwargs: object) -> None:
        calls.update(kwargs)

    monkeypatch.setattr(run_alternating, "_run_schedule", run_schedule)

    run_alternating.main([
        "--epochs", "1", "--phase-steps", "2", "--problem-count", "1",
        "--eval-problem-count", "1", "--save-dir", str(tmp_path),
    ])

    assert calls["gradient_state"] is shared_state
    assert calls["eggroll_state"] is shared_state
    assert calls["examples"] == [("q", "1")]
    assert calls["epochs"] == 1
    assert calls["phase_steps"] == 2
    assert calls["evaluator"] == "evaluator"


def test_main_phase_checkpoint_serializes_latest_boundary_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    position = ExperimentPosition("eggroll", 1, 2, 1, 1, 2)
    evaluation = EvaluationResult(position, 0.75, 0.5, 1.0)
    evaluator = SimpleNamespace(latest=None)
    evaluator.evaluate = lambda _position: setattr(evaluator, "latest", evaluation) or evaluation
    state = SimpleNamespace(trainable_params={})
    gradient = SimpleNamespace(optimizer=object())
    eggroll = SimpleNamespace(optimizer=object())
    stage0_dataset = SimpleNamespace(
        train_selection=object(),
        held_out_selection=object(),
        held_out_records=lambda: ["validation-record"],
    )
    captured_checkpoint: dict[str, object] = {}

    monkeypatch.setattr(run_alternating, "load_stage0_dataset", lambda: stage0_dataset)
    monkeypatch.setattr(
        run_alternating,
        "training_examples",
        lambda *_args, **_kwargs: [("q", "1"), ("q2", "2")],
    )
    monkeypatch.setattr(
        run_alternating,
        "load_held_out_problems",
        lambda _records, _count: ["held-out"],
    )
    monkeypatch.setattr(
        run_alternating,
        "_selection_metadata",
        lambda selection, _count: {
            "identity": "a" * 64,
            "split": (
                "train" if selection is stage0_dataset.train_selection else "validation"
            ),
            "seed": 0,
            "problem_count": 1,
            "ordered_item_ids": ["item-0"],
        },
    )
    monkeypatch.setattr(run_alternating, "TrainingState", lambda *_args: state)
    monkeypatch.setattr(run_alternating, "LatentCoreTrainer", lambda **_kwargs: gradient)
    monkeypatch.setattr(run_alternating, "EggrollTrainer", lambda **_kwargs: eggroll)
    monkeypatch.setattr(run_alternating, "_Evaluator", lambda _model, _held: evaluator)

    def capture_checkpoint(**kwargs: object) -> object:
        captured_checkpoint.update(kwargs)
        return object()

    monkeypatch.setattr(run_alternating, "build_alternating_checkpoint", capture_checkpoint)
    monkeypatch.setattr(
        run_alternating,
        "save_boundary_checkpoints",
        lambda *_args, **_kwargs: (tmp_path / "phase-2.ckpt",),
    )

    def run_schedule(**kwargs: object) -> None:
        result = kwargs["evaluator"].evaluate(position)  # type: ignore[attr-defined]
        assert result is evaluation
        kwargs["save_boundary"](  # type: ignore[operator]
            CheckpointSchedule("gradient", 0, 2, 1, 1),
            phase_boundary=True,
            epoch_boundary=False,
            evaluation_record=EvaluationRecord(
                result=evaluation,
                position=position,
                boundaries=frozenset({"phase"}),
            ),
        )

    monkeypatch.setattr(run_alternating, "_run_schedule", run_schedule)

    run_alternating.main(
        [
            "--epochs",
            "1",
            "--phase-steps",
            "2",
            "--problem-count",
            "2",
            "--eval-problem-count",
            "1",
            "--save-dir",
            str(tmp_path),
        ]
    )

    assert captured_checkpoint["metrics"] == {
        "record_type": "evaluation",
        "update_method": "eggroll",
        "cycle": 1,
        "global_step": 2,
        "epoch": 1,
        "example_position": 1,
        "phase_step": 2,
        "boundaries": ["phase"],
        "language_model_loss": 0.75,
        "shared_variance": 0.5,
        "answer_exact_match": 1.0,
        "phase_variance_sum": 0.0,
    }


def test_compatible_checkpoint_restores_model_optimizers_and_every_rng_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = {
        name: nn.Parameter(torch.tensor([float(index)]))
        for index, name in enumerate(stage0_parameter_paths())
    }
    gradient = torch.optim.Adam(parameters.values(), lr=1e-4)
    eggroll_parameters = [parameters[name] for name in EGGROLL_PARAMETER_PATHS]
    eggroll = torch.optim.SGD(eggroll_parameters, lr=0.1, momentum=0.0)
    for optimizer, owned_parameters in (
        (gradient, list(parameters.values())),
        (eggroll, eggroll_parameters),
    ):
        for parameter in owned_parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad()
    cuda_rng = torch.tensor([3, 1, 4, 1], dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [cuda_rng.clone()])
    fixture = _metadata()
    fixture["identity"]["runtime"]["device_topology"] = ["cuda:0"]
    run_config = copy.deepcopy(RUN_CONFIG)
    run_config["eggroll_optimizer"]["learning_rate"] = 0.1
    checkpoint = build_alternating_checkpoint(
        identity=fixture["identity"],
        selections=fixture["selections"],
        model_state=parameters,
        eggroll_optimizer=eggroll,
        gradient_optimizer=gradient,
        schedule=CheckpointSchedule(
            "gradient",
            0,
            2,
            1,
            1,
            gradient_optimizer_calls=1,
            eggroll_optimizer_calls=1,
        ),
        phase_steps=2,
        next_dataset_position=2,
        metrics={"loss": 1.25},
        run_config=run_config,
    )
    for parameter in parameters.values():
        parameter.data.add_(100.0)
    restored_gradient = torch.optim.Adam(parameters.values(), lr=9e-4)
    restored_eggroll = torch.optim.SGD(
        [parameters[name] for name in EGGROLL_PARAMETER_PATHS],
        lr=9e-3,
        momentum=0.0,
    )
    random.random()
    np.random.random()
    torch.rand(1)
    restored_cuda: list[list[torch.Tensor]] = []
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda values: restored_cuda.append(list(values)),
    )

    run_alternating._restore_checkpoint_state(
        checkpoint,
        SimpleNamespace(trainable_params=parameters),
        restored_gradient,
        restored_eggroll,
    )

    assert all(
        torch.equal(parameter, checkpoint.tensors[f"model.{name}"])
        for name, parameter in parameters.items()
    )
    assert restored_gradient.param_groups[0]["lr"] == 1e-4
    assert restored_eggroll.param_groups[0]["lr"] == 0.1
    assert all(restored_gradient.state[parameter] for parameter in parameters.values())
    assert set(restored_eggroll.state) == set()
    python_rng = checkpoint.metadata["rng"]["python"]
    assert random.getstate() == (
        python_rng["version"],
        tuple(python_rng["state"]),
        python_rng["gaussian_cache"],
    )
    numpy_rng = checkpoint.metadata["rng"]["numpy"]
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_rng["bit_generator"]
    assert np.array_equal(
        restored_numpy[1], checkpoint.tensors[numpy_rng["state_tensor"]].numpy()
    )
    assert torch.equal(
        torch.get_rng_state(),
        checkpoint.tensors[checkpoint.metadata["rng"]["pytorch_cpu"]["state_tensor"]],
    )
    assert len(restored_cuda) == 1
    assert len(restored_cuda[0]) == 1
    assert torch.equal(restored_cuda[0][0], cuda_rng)


def test_small_injected_run_crosses_boundaries_and_resumes_without_revisiting_examples(
    tmp_path: Path,
) -> None:
    visits: list[tuple[str, int, int, str]] = []

    class FakeEngine:
        def __init__(self, method: str) -> None:
            self.method = method

        def train_step(
            self, example: object, position: ExperimentPosition
        ) -> StepResult:
            visits.append(
                (str(example), position.epoch, position.example_position, self.method)
            )
            variance = 0.03 if self.method == "eggroll" else 0.005
            return StepResult(position, 1.0, 1.0, 0.0, variance)

    class FakeEvaluator:
        def evaluate(self, position: ExperimentPosition) -> EvaluationResult:
            return EvaluationResult(position, 0.75, 0.5, 1.0)

    saved: list[CheckpointSchedule] = []
    saved_evaluations: list[EvaluationRecord[EvaluationResult]] = []

    def save(
        schedule: CheckpointSchedule,
        *,
        phase_boundary: bool,
        epoch_boundary: bool,
        evaluation_record: EvaluationRecord[EvaluationResult],
    ) -> tuple[Path, ...]:
        saved.append(schedule)
        saved_evaluations.append(evaluation_record)
        names = []
        if phase_boundary:
            names.append(tmp_path / f"phase-{schedule.global_step}.pt")
        if epoch_boundary:
            names.append(tmp_path / f"epoch-{schedule.epoch}.pt")
        for path in names:
            path.write_text(str(schedule.to_dict()), encoding="utf-8")
        return tuple(names)

    first_output = StringIO()
    run_alternating._run_schedule(
        examples=["a", "b", "c"],
        epochs=1,
        phase_steps=2,
        variance_lower_threshold=0.01,
        variance_upper_threshold=0.02,
        log_every=3,
        eggroll_engine=FakeEngine("eggroll"),
        gradient_engine=FakeEngine("gradient"),
        evaluator=FakeEvaluator(),
        save_boundary=save,
        output=first_output,
    )

    resumed_output = StringIO()
    run_alternating._run_schedule(
        examples=["a", "b", "c"],
        epochs=2,
        phase_steps=2,
        variance_lower_threshold=0.01,
        variance_upper_threshold=0.02,
        log_every=3,
        eggroll_engine=FakeEngine("eggroll"),
        gradient_engine=FakeEngine("gradient"),
        evaluator=FakeEvaluator(),
        save_boundary=save,
        output=resumed_output,
        resume=saved[-1],
    )

    assert visits == [
        ("a", 1, 0, "eggroll"),
        ("b", 1, 1, "eggroll"),
        ("c", 1, 2, "gradient"),
        ("a", 2, 0, "gradient"),
        ("b", 2, 1, "eggroll"),
        ("c", 2, 2, "eggroll"),
    ]
    records = [
        json.loads(line)
        for line in first_output.getvalue().splitlines()
        + resumed_output.getvalue().splitlines()
    ]
    training_records = [
        record for record in records if record["record_type"] == "training"
    ]
    assert [record["global_step"] for record in training_records] == [3, 6]
    assert training_records == [
        {
            "record_type": "training",
            "update_method": "gradient",
            "cycle": 2,
            "global_step": 3,
            "epoch": 1,
            "example_position": 2,
            "phase_step": 1,
            "language_model_loss": 1.0,
            "shared_variance": 0.005,
            "next_example_position": 3,
            "observation_window_examples": 1,
            "records_consumed": 1,
            "batch_size": 1,
            "optimizer_call_count": 1,
        },
        {
            "record_type": "training",
            "update_method": "eggroll",
            "cycle": 3,
            "global_step": 6,
            "epoch": 2,
            "example_position": 2,
            "phase_step": 2,
            "language_model_loss": 1.0,
            "shared_variance": 0.03,
            "next_example_position": 3,
            "observation_window_examples": 2,
            "records_consumed": 1,
            "batch_size": 1,
            "optimizer_call_count": 4,
        },
    ]
    evaluation_records = [
        record for record in records if record["record_type"] == "evaluation"
    ]
    assert [record["global_step"] for record in evaluation_records] == [3, 6]
    assert evaluation_records[0]["boundaries"] == ["epoch", "partial_phase"]
    assert evaluation_records[1]["boundaries"] == ["epoch", "phase"]

    checkpoint_records = [
        record for record in records if record["record_type"] == "checkpoint"
    ]
    assert checkpoint_records == [
        {
            "record_type": "checkpoint",
            "paths": [str(tmp_path / "epoch-1.pt")],
        },
        {
            "record_type": "checkpoint",
            "paths": [str(tmp_path / "phase-6.pt"), str(tmp_path / "epoch-2.pt")],
        },
    ]
    assert saved == [
        CheckpointSchedule(
            "gradient", 0, 2, 1, 1, eggroll_optimizer_calls=2
        ),
        CheckpointSchedule(
            "gradient",
            1,
            3,
            1,
            2,
            phase_variance_sum=0.005,
            gradient_optimizer_calls=1,
            eggroll_optimizer_calls=2,
        ),
        CheckpointSchedule(
            "eggroll",
            0,
            4,
            2,
            0,
            gradient_optimizer_calls=2,
            eggroll_optimizer_calls=2,
        ),
        CheckpointSchedule(
            "gradient",
            0,
            6,
            2,
            2,
            gradient_optimizer_calls=2,
            eggroll_optimizer_calls=4,
        ),
    ]
    assert [record.boundaries for record in saved_evaluations] == [
        frozenset({"phase"}),
        frozenset({"epoch", "partial_phase"}),
        frozenset({"phase"}),
        frozenset({"epoch", "phase"}),
    ]
    assert saved[1] == CheckpointSchedule(
        "gradient",
        1,
        3,
        1,
        2,
        phase_variance_sum=0.005,
        gradient_optimizer_calls=1,
        eggroll_optimizer_calls=2,
    )
    assert saved[-1] == CheckpointSchedule(
        "gradient",
        0,
        6,
        2,
        2,
        gradient_optimizer_calls=2,
        eggroll_optimizer_calls=4,
    )
