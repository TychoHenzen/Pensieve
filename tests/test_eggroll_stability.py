from __future__ import annotations

import builtins
import copy
import json
import random
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from eval.stream.generators.asdiv_a import AsdivRecord
from train.eggroll_stability import (
    BASELINE_PROBLEM_COUNT,
    FailedThreshold,
    ImplementationIdentity,
    ParameterRms,
    StabilityMetrics,
    StabilityProgress,
    StabilityReportValidationError,
    build_stability_asset_identity,
    build_stability_configuration,
    load_compatible_stability_report,
    measure_fresh_baseline,
    run_bounded_stability_gate,
)
from train.training_state import EGGROLL_PARAMETER_PATHS


class FakeWorkspace:
    def __init__(self) -> None:
        self.slots = torch.tensor([[1.0, 2.0]])

    def snapshot(self) -> torch.Tensor:
        return self.slots.clone()

    def restore(self, state: torch.Tensor) -> None:
        self.slots = state.clone()


class FakeState:
    def __init__(self) -> None:
        self.trainable_params = {
            path: nn.Parameter(torch.full((2, 2), 1.0 + index)) for index, path in enumerate(EGGROLL_PARAMETER_PATHS)
        }
        self.workspace = FakeWorkspace()

    def eggroll_parameters(self):
        return iter(self.trainable_params.values())


class FakeTrainer:
    def __init__(self) -> None:
        self.state = FakeState()
        self.optimizer = torch.optim.SGD(self.state.eggroll_parameters(), lr=0.1, momentum=0.0)
        self.fitness_batch_size = 8
        self.pop_size = 128
        self.sigma = 0.001
        self.rank = 4
        self.eval_batch_size = 8
        self.variance_weight = 1.0
        self.prompt_alignment_weight = 0.0
        self.seen_ids: list[str] = []

    def train_fitness_batch(
        self,
        fitness_batch,
        position=None,
        *,
        optimizer_call_count: int = 1,
    ):
        del position, optimizer_call_count
        self.seen_ids.extend(record.id for record in fitness_batch.records)
        with torch.no_grad():
            for parameter in self.state.eggroll_parameters():
                parameter.add_(0.001)
        return SimpleNamespace(consumed_record_count=fitness_batch.consumed_record_count)


def _records(split: str, count: int) -> tuple[AsdivRecord, ...]:
    return tuple(
        AsdivRecord(
            id=f"{split}-{index:03d}",
            split=split,
            question=f"question {index}",
            target=str(index),
        )
        for index in range(count)
    )


def _metrics(
    trainer: FakeTrainer,
    *,
    language_model_loss: float = 1.0,
    separation_retention: float = 0.8,
    exact_accuracy: float = 0.125,
    first_token_accuracy: float = 0.25,
) -> StabilityMetrics:
    return StabilityMetrics(
        problem_count=BASELINE_PROBLEM_COUNT,
        parameter_rms=tuple(
            ParameterRms(
                path,
                float(torch.sqrt(torch.mean(parameter.detach().square())).item()),
            )
            for path, parameter in trainer.state.trainable_params.items()
        ),
        language_model_loss=language_model_loss,
        exact_accuracy=exact_accuracy,
        first_token_accuracy=first_token_accuracy,
        valid_answer_rate=0.75,
        output_diversity=0.5,
        output_dominance=0.25,
        shared_slot_variance=0.02,
        student_teacher_mse=0.4,
        student_cross_problem_cosine=0.6,
        teacher_cross_problem_cosine=0.5,
        separation_retention=separation_retention,
    )


def _configuration(trainer: FakeTrainer):
    held_out = _records("validation", BASELINE_PROBLEM_COUNT)
    return build_stability_configuration(
        trainer,
        asset_identity=build_stability_asset_identity(
            runtime={"device": "cpu", "torch": torch.__version__},
            held_out_records=held_out,
        ),
        implementation=ImplementationIdentity(
            sha256="a" * 64,
            sources=(("train/eggroll_stability.py", "b" * 64),),
        ),
        initialization_seed=0,
    )


def _optimizer_copy(trainer: FakeTrainer) -> dict[str, object]:
    return copy.deepcopy(trainer.optimizer.state_dict())


def _numpy_copy() -> tuple[str, np.ndarray, int, int, float]:
    state = np.random.get_state()
    return (state[0], state[1].copy(), state[2], state[3], state[4])


def _numpy_equal(
    left: tuple[str, np.ndarray, int, int, float],
    right: tuple[str, np.ndarray, int, int, float],
) -> bool:
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


# covers: train/eggroll-stability-gate :: Stability gate measures a fresh absolute baseline :: Record the untrained baseline
def test_records_one_immutable_zero_example_baseline_before_updates() -> None:
    trainer = FakeTrainer()
    held_out = _records("validation", BASELINE_PROBLEM_COUNT)
    seen_problems: list[tuple[str, str]] = []

    def evaluator(_trainer, problems):
        seen_problems.extend(problems)
        return _metrics(trainer)

    checkpoint = measure_fresh_baseline(
        trainer,
        held_out,
        StabilityProgress(),
        evaluator,
        elapsed_seconds=1.25,
    )
    encoded = checkpoint.to_dict()

    assert checkpoint.consumed_examples == 0
    assert checkpoint.optimizer_call_count == 0
    assert checkpoint.baseline is checkpoint.current
    assert dict(checkpoint.deltas) == {
        "language_model_loss": 0.0,
        "exact_accuracy": 0.0,
        "first_token_accuracy": 0.0,
        "valid_answer_rate": 0.0,
        "output_diversity": 0.0,
        "output_dominance": 0.0,
        "shared_slot_variance": 0.0,
        "student_teacher_mse": 0.0,
        "student_cross_problem_cosine": 0.0,
        "teacher_cross_problem_cosine": 0.0,
        "separation_retention": 0.0,
    }
    assert seen_problems == [(record.question, record.target) for record in held_out]
    assert encoded["current"]["problem_count"] == BASELINE_PROBLEM_COUNT
    assert encoded["current"]["parameter_rms"] == [
        {"path": path, "rms": float(index + 1)} for index, path in enumerate(EGGROLL_PARAMETER_PATHS)
    ]
    with pytest.raises(FrozenInstanceError):
        checkpoint.consumed_examples = 1  # type: ignore[misc]


# covers: train/eggroll-stability-gate :: Stability gate measures a fresh absolute baseline :: Baseline evaluation is isolated
def test_baseline_evaluation_restores_model_optimizer_workspace_cursor_and_rng() -> None:
    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    trainer = FakeTrainer()
    progress = StabilityProgress()
    parameters_before = tuple(parameter.detach().clone() for parameter in trainer.state.trainable_params.values())
    optimizer_before = _optimizer_copy(trainer)
    workspace_before = trainer.state.workspace.snapshot()
    python_before = copy.deepcopy(random.getstate())
    numpy_before = _numpy_copy()
    torch_before = torch.get_rng_state().clone()

    def destructive_evaluator(_trainer, problems):
        assert len(problems) == BASELINE_PROBLEM_COUNT
        with torch.no_grad():
            for parameter in trainer.state.trainable_params.values():
                parameter.add_(99.0)
        trainer.optimizer.param_groups[0]["lr"] = 8.0
        trainer.state.workspace.slots.add_(7.0)
        progress.consumed_examples = 31
        progress.optimizer_call_count = 4
        random.random()
        np.random.random()
        torch.rand(3)
        return _metrics(trainer)

    checkpoint = measure_fresh_baseline(
        trainer,
        _records("validation", BASELINE_PROBLEM_COUNT),
        progress,
        destructive_evaluator,
    )
    numpy_after = np.random.get_state()

    assert checkpoint.consumed_examples == 0
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(trainer.state.trainable_params.values(), parameters_before, strict=True)
    )
    assert trainer.optimizer.state_dict() == optimizer_before
    assert torch.equal(trainer.state.workspace.snapshot(), workspace_before)
    assert progress == StabilityProgress(consumed_examples=0, optimizer_call_count=0)
    assert random.getstate() == python_before
    assert _numpy_equal(numpy_after, numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)


# covers: train/eggroll-stability-gate :: Stability gate runs bounded development checkpoints :: Emit every healthy development checkpoint
def test_healthy_development_emits_zero_eight_thirty_two_and_256() -> None:
    trainer = FakeTrainer()
    emitted = []
    clock_value = 0.0

    def clock() -> float:
        nonlocal clock_value
        clock_value += 1.0
        return clock_value

    def evaluator(_trainer, problems):
        assert len(problems) == BASELINE_PROBLEM_COUNT
        consumed = len(trainer.seen_ids)
        return _metrics(
            trainer,
            language_model_loss=1.0 - consumed / 10_000,
            separation_retention=0.8,
        )

    report = run_bounded_stability_gate(
        trainer,
        _records("train", 300),
        _records("validation", BASELINE_PROBLEM_COUNT),
        _configuration(trainer),
        evaluator,
        observer=emitted.append,
        clock=clock,
    )
    records = report.checkpoints

    assert report.status == "failed"
    assert report.outcome_code == 1
    assert [record.consumed_examples for record in records] == [0, 8, 32, 256]
    assert [record.optimizer_call_count for record in records] == [0, 1, 4, 32]
    assert tuple(trainer.seen_ids) == tuple(f"train-{index:03d}" for index in range(256))
    assert all(record.baseline is records[0].baseline for record in records)
    assert records[1].max_update_relative_matrix_rms > 0.0
    assert records[2].max_update_relative_matrix_rms >= records[1].max_update_relative_matrix_rms
    assert records[1].eta_seconds == pytest.approx(62.0)
    assert records[-1].eta_seconds == 0.0
    assert [json.loads(event.canonical_json_line())["checkpoint"]["consumed_examples"] for event in emitted] == [
        0,
        8,
        32,
        256,
    ]
    assert all(event.canonical_json_line().endswith("\n") for event in emitted)


# covers: train/eggroll-stability-gate :: Stability gate runs bounded development checkpoints :: Stop on absolute regression
def test_absolute_regression_stops_and_retains_exact_failed_thresholds() -> None:
    trainer = FakeTrainer()

    def evaluator(_trainer, problems):
        assert len(problems) == BASELINE_PROBLEM_COUNT
        if not trainer.seen_ids:
            return _metrics(trainer, language_model_loss=1.0, separation_retention=0.8)
        return _metrics(trainer, language_model_loss=1.051, separation_retention=0.75)

    report = run_bounded_stability_gate(
        trainer,
        _records("train", 300),
        _records("validation", BASELINE_PROBLEM_COUNT),
        _configuration(trainer),
        evaluator,
        clock=lambda: 1.0,
    )
    failures = report.failed_thresholds

    assert report.status == "failed"
    assert report.outcome_code == 1
    assert [record.consumed_examples for record in report.checkpoints] == [0, 8]
    assert len(trainer.seen_ids) == 8
    assert failures == (
        FailedThreshold("language_model_loss", 1.051, "<=", 1.05, 1.0),
        FailedThreshold("separation_retention", 0.75, ">=", 0.76, 0.8),
    )
    assert report.checkpoints[-1].failed_thresholds == failures
    assert report.to_dict()["failed_thresholds"] == [
        {
            "metric": "language_model_loss",
            "observed": 1.051,
            "operator": "<=",
            "threshold": 1.05,
            "baseline": 1.0,
        },
        {
            "metric": "separation_retention",
            "observed": 0.75,
            "operator": ">=",
            "threshold": 0.76,
            "baseline": 0.8,
        },
    ]


def test_gate_uses_injected_metrics_without_importing_alignment_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = FakeTrainer()
    original_import = builtins.__import__
    sys.modules.pop("train.search_alignment_weight", None)

    def reject_search_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "train.search_alignment_weight":
            raise AssertionError("chunk 3 must not import task 7 alignment search")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_search_import)
    checkpoint = measure_fresh_baseline(
        trainer,
        _records("validation", BASELINE_PROBLEM_COUNT),
        StabilityProgress(),
        lambda active_trainer, _problems: _metrics(active_trainer),
    )

    assert checkpoint.current.language_model_loss == 1.0
    assert checkpoint.consumed_examples == 0
    assert "train.search_alignment_weight" not in sys.modules


# covers: train/eggroll-stability-gate :: Passing health requires task improvement and bounded updates :: Configuration passes the absolute gate
def test_final_checkpoint_passes_only_when_every_health_condition_holds() -> None:
    trainer = FakeTrainer()

    def evaluator(_trainer, problems):
        assert len(problems) == BASELINE_PROBLEM_COUNT
        if not trainer.seen_ids:
            return _metrics(
                trainer,
                language_model_loss=1.0,
                exact_accuracy=0.0,
                first_token_accuracy=0.0,
            )
        return _metrics(
            trainer,
            language_model_loss=0.9,
            exact_accuracy=0.125,
            first_token_accuracy=0.25,
        )

    report = run_bounded_stability_gate(
        trainer,
        _records("train", 300),
        _records("validation", BASELINE_PROBLEM_COUNT),
        _configuration(trainer),
        evaluator,
        clock=lambda: 1.0,
    )

    assert report.status == "passed"
    assert report.outcome_code == 0
    assert report.failed_thresholds == ()
    assert report.checkpoints[-1].failed_thresholds == ()
    assert report.checkpoints[-1].consumed_examples == 256
    assert report.checkpoints[-1].current.language_model_loss < report.checkpoints[0].current.language_model_loss
    assert report.checkpoints[-1].current.exact_accuracy > report.checkpoints[0].current.exact_accuracy
    assert report.checkpoints[-1].current.first_token_accuracy > report.checkpoints[0].current.first_token_accuracy
    assert (
        report.checkpoints[-1].current.separation_retention >= 0.95 * report.checkpoints[0].current.separation_retention
    )
    assert report.checkpoints[-1].max_update_relative_matrix_rms <= 0.01


# covers: train/eggroll-stability-gate :: Passing health requires task improvement and bounded updates :: Configuration lacks task improvement
def test_final_checkpoint_lists_every_missing_task_improvement() -> None:
    trainer = FakeTrainer()

    def evaluator(_trainer, problems):
        assert len(problems) == BASELINE_PROBLEM_COUNT
        return _metrics(
            trainer,
            language_model_loss=1.0 if not trainer.seen_ids else 0.9,
            exact_accuracy=0.125,
            first_token_accuracy=0.25,
        )

    report = run_bounded_stability_gate(
        trainer,
        _records("train", 300),
        _records("validation", BASELINE_PROBLEM_COUNT),
        _configuration(trainer),
        evaluator,
        clock=lambda: 1.0,
    )
    failed_metrics = [failure.metric for failure in report.failed_thresholds]

    assert report.status == "failed"
    assert report.outcome_code != 0
    assert failed_metrics == ["exact_accuracy", "first_token_accuracy"]
    assert report.checkpoints[-1].failed_thresholds == report.failed_thresholds
    assert report.checkpoints[-1].current.language_model_loss < report.checkpoints[0].current.language_model_loss
    assert (
        report.checkpoints[-1].current.separation_retention >= 0.95 * report.checkpoints[0].current.separation_retention
    )


def _passing_report(trainer: FakeTrainer):
    def evaluator(_trainer, _problems):
        if not trainer.seen_ids:
            return _metrics(
                trainer,
                language_model_loss=1.0,
                exact_accuracy=0.0,
                first_token_accuracy=0.0,
            )
        return _metrics(
            trainer,
            language_model_loss=0.9,
            exact_accuracy=0.125,
            first_token_accuracy=0.25,
        )

    return run_bounded_stability_gate(
        trainer,
        _records("train", 300),
        _records("validation", BASELINE_PROBLEM_COUNT),
        _configuration(trainer),
        evaluator,
        clock=lambda: 1.0,
    )


# covers: train/eggroll-stability-gate :: Full EGGROLL workflows require a compatible passing report :: Start with a compatible passing report
def test_compatible_passing_report_validates_without_loading_models_or_data(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = FakeTrainer()
    report = _passing_report(trainer)
    report_path = tmp_path / "stability.json"
    report_path.write_text(report.canonical_json() + "\n", encoding="utf-8")
    original_import = builtins.__import__

    def reject_runtime_loaders(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"datasets", "transformers", "sentence_transformers"}:
            raise AssertionError(f"validator loaded runtime dependency {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_runtime_loaders)
    validated = load_compatible_stability_report(report_path, report.configuration)

    assert validated.report.status == "passed"
    assert validated.report.configuration == report.configuration
    assert validated.report.checkpoints[-1].consumed_examples == 256
    assert len(validated.sha256) == 64
    assert all(character in "0123456789abcdef" for character in validated.sha256)


# covers: train/eggroll-stability-gate :: Full EGGROLL workflows require a compatible passing report :: Reject a mismatched report
def test_report_validator_names_every_mismatched_canonical_field(tmp_path) -> None:
    trainer = FakeTrainer()
    report = _passing_report(trainer)
    payload = report.to_dict()
    payload["configuration"]["implementation"]["sha256"] = "c" * 64
    payload["configuration"]["eggroll"]["population"] = 64
    payload["configuration"]["eggroll"]["sigma"] = 0.002
    report_path = tmp_path / "stale.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StabilityReportValidationError) as caught:
        load_compatible_stability_report(report_path, report.configuration)

    issues = caught.value.issues
    assert any("$.configuration.implementation.sha256" in issue for issue in issues)
    assert any("$.configuration.eggroll.population" in issue for issue in issues)
    assert any("$.configuration.eggroll.sigma" in issue for issue in issues)
    assert len(issues) == 3
    assert "implementation.sha256" in str(caught.value)


def test_report_validator_rejects_missing_failed_and_malformed_reports(tmp_path) -> None:
    trainer = FakeTrainer()
    configuration = _configuration(trainer)
    missing = tmp_path / "missing.json"

    with pytest.raises(StabilityReportValidationError, match="cannot read report"):
        load_compatible_stability_report(missing, configuration)

    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"status":"passed",', encoding="utf-8")
    with pytest.raises(StabilityReportValidationError, match="malformed JSON"):
        load_compatible_stability_report(malformed, configuration)

    report = _passing_report(trainer).to_dict()
    report["status"] = "failed"
    report["outcome_code"] = 1
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(StabilityReportValidationError) as caught:
        load_compatible_stability_report(failed, configuration)

    assert any("$.status" in issue for issue in caught.value.issues)
    assert any("$.outcome_code" in issue for issue in caught.value.issues)


def test_report_validator_recomputes_passing_health_conditions(tmp_path) -> None:
    trainer = FakeTrainer()
    report = _passing_report(trainer)
    payload = report.to_dict()
    payload["checkpoints"][-1]["current"]["exact_accuracy"] = payload["checkpoints"][-1]["baseline"]["exact_accuracy"]
    report_path = tmp_path / "false-pass.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StabilityReportValidationError) as caught:
        load_compatible_stability_report(report_path, report.configuration.to_dict())

    assert any("$.checkpoints[-1].exact_accuracy" in issue for issue in caught.value.issues)


def test_report_with_empty_device_topology_round_trips_as_array(tmp_path) -> None:
    trainer = FakeTrainer()
    report = _passing_report(trainer)
    payload = report.to_dict()
    payload["configuration"]["asset_identity"]["runtime"]["device_topology"] = []
    expected = report.configuration.to_dict()
    expected["asset_identity"]["runtime"]["device_topology"] = []
    report_path = tmp_path / "cpu-only.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    validated = load_compatible_stability_report(report_path, expected)

    topology = validated.report.configuration.to_dict()["asset_identity"]["runtime"]["device_topology"]
    assert topology == []
    assert isinstance(topology, list)
