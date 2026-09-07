"""Bounded absolute-health measurements for fresh Stage 0 EGGROLL runs.

The command-line entry point is intentionally separate.  This module owns the
immutable report contract, evaluation isolation, implementation identity, and
the deterministic 0/8/32/256-example development loop.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import torch

from eval.stage0_identity import canonical_json_bytes, training_identity
from eval.stream.generators.asdiv_a import AsdivRecord
from train.stage0_data import FitnessBatch
from train.training_results import ExperimentPosition
from train.training_state import EGGROLL_PARAMETER_PATHS

STABILITY_SCHEMA_VERSION = 1
BASELINE_PROBLEM_COUNT = 64
DEVELOPMENT_EXAMPLE_COUNT = 256
DEVELOPMENT_CHECKPOINTS = (8, 32, 256)
MAX_LOSS_RATIO = 1.05
MIN_SEPARATION_RATIO = 0.95
MAX_UPDATE_RELATIVE_MATRIX_RMS = 0.01
STABILITY_IMPLEMENTATION_PATHS = (
    "codecs_module/decoder.py",
    "codecs_module/encoder.py",
    "core/latent_loop.py",
    "core/qwen_tap.py",
    "eval/gate/answer_scoring.py",
    "eval/stage0_identity.py",
    "train/answer_objective.py",
    "train/eggroll_factorized.py",
    "train/eggroll_perturbations.py",
    "train/eggroll_stability.py",
    "train/eggroll_stability_evaluation.py",
    "train/eggroll_trainer.py",
    "train/eggroll_updates.py",
    "train/run_eggroll_stability.py",
    "train/stage0_data.py",
    "train/training_results.py",
    "train/training_state.py",
    "train/vicreg.py",
    "workspace/concept_slots.py",
)


class StabilityTrainer(Protocol):
    """The EGGROLL surface used by the bounded development gate."""

    state: Any
    optimizer: Any
    fitness_batch_size: int
    pop_size: int
    sigma: float
    rank: int
    eval_batch_size: int
    variance_weight: float
    prompt_alignment_weight: float

    def train_fitness_batch(
        self,
        fitness_batch: FitnessBatch,
        position: ExperimentPosition | None = None,
        *,
        optimizer_call_count: int = 1,
    ) -> Any: ...


@dataclass(frozen=True)
class ImplementationIdentity:
    """Canonical digest of the ordered source files that define the gate."""

    sha256: str
    sources: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.sha256):
            raise ValueError("implementation identity must use a SHA-256 digest")
        if not self.sources or any(not path or not _is_sha256(digest) for path, digest in self.sources):
            raise ValueError("implementation sources require paths and SHA-256 digests")
        if tuple(path for path, _ in self.sources) != tuple(sorted(path for path, _ in self.sources)):
            raise ValueError("implementation sources must use canonical path order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "sources": dict(self.sources),
        }


@dataclass(frozen=True)
class StabilityConfiguration:
    """Every field that makes a stability result reusable or stale."""

    asset_identity: tuple[tuple[str, Any], ...]
    implementation: ImplementationIdentity
    optimizer_contract: str
    parameter_scope: tuple[str, ...]
    population: int
    sigma: float
    rank: int
    fitness_batch_size: int
    evaluation_batch_size: int
    variance_weight: float
    prompt_alignment_weight: float
    learning_rate: float
    initialization_seed: int
    held_out_problem_count: int = BASELINE_PROBLEM_COUNT
    development_example_count: int = DEVELOPMENT_EXAMPLE_COUNT
    checkpoints: tuple[int, ...] = DEVELOPMENT_CHECKPOINTS

    def __post_init__(self) -> None:
        if self.optimizer_contract != "torch.optim.SGD(momentum=0.0)":
            raise ValueError("stability configuration requires momentum-free SGD")
        if self.parameter_scope != EGGROLL_PARAMETER_PATHS:
            raise ValueError("stability configuration requires the declared matrix scope")
        if self.held_out_problem_count != BASELINE_PROBLEM_COUNT:
            raise ValueError("stability configuration has the wrong held-out count")
        if self.development_example_count != DEVELOPMENT_EXAMPLE_COUNT:
            raise ValueError("stability configuration has the wrong development count")
        if self.checkpoints != DEVELOPMENT_CHECKPOINTS:
            raise ValueError("stability configuration has noncanonical checkpoints")
        if isinstance(self.initialization_seed, bool) or not isinstance(self.initialization_seed, int):
            raise TypeError("initialization_seed must be a non-boolean integer")
        _require_finite(self.to_dict(), "configuration")

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_identity": _thaw_mapping(self.asset_identity),
            "implementation": self.implementation.to_dict(),
            "eggroll": {
                "optimizer_contract": self.optimizer_contract,
                "parameter_scope": list(self.parameter_scope),
                "population": self.population,
                "sigma": self.sigma,
                "rank": self.rank,
                "fitness_batch_size": self.fitness_batch_size,
                "evaluation_batch_size": self.evaluation_batch_size,
                "variance_weight": self.variance_weight,
                "prompt_alignment_weight": self.prompt_alignment_weight,
                "learning_rate": self.learning_rate,
            },
            "initialization_seed": self.initialization_seed,
            "held_out_problem_count": self.held_out_problem_count,
            "development_example_count": self.development_example_count,
            "checkpoints": list(self.checkpoints),
        }


@dataclass(frozen=True)
class ParameterRms:
    path: str
    rms: float

    def __post_init__(self) -> None:
        if not self.path or not math.isfinite(self.rms) or self.rms < 0.0:
            raise ValueError("parameter RMS requires a path and finite nonnegative value")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "rms": self.rms}


@dataclass(frozen=True)
class StabilityMetrics:
    """Absolute task and representation measurements for one checkpoint."""

    problem_count: int
    parameter_rms: tuple[ParameterRms, ...]
    language_model_loss: float
    exact_accuracy: float
    first_token_accuracy: float
    valid_answer_rate: float
    output_diversity: float
    output_dominance: float
    shared_slot_variance: float
    student_teacher_mse: float
    student_cross_problem_cosine: float
    teacher_cross_problem_cosine: float
    separation_retention: float

    def __post_init__(self) -> None:
        if self.problem_count != BASELINE_PROBLEM_COUNT:
            raise ValueError(f"stability metrics require {BASELINE_PROBLEM_COUNT} problems")
        _require_finite(self.to_dict(), "metrics")

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_count": self.problem_count,
            "parameter_rms": [item.to_dict() for item in self.parameter_rms],
            "language_model_loss": self.language_model_loss,
            "exact_accuracy": self.exact_accuracy,
            "first_token_accuracy": self.first_token_accuracy,
            "valid_answer_rate": self.valid_answer_rate,
            "output_diversity": self.output_diversity,
            "output_dominance": self.output_dominance,
            "shared_slot_variance": self.shared_slot_variance,
            "student_teacher_mse": self.student_teacher_mse,
            "student_cross_problem_cosine": self.student_cross_problem_cosine,
            "teacher_cross_problem_cosine": self.teacher_cross_problem_cosine,
            "separation_retention": self.separation_retention,
        }


@dataclass(frozen=True)
class FailedThreshold:
    metric: str
    observed: float
    operator: str
    threshold: float
    baseline: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "observed": self.observed,
            "operator": self.operator,
            "threshold": self.threshold,
            "baseline": self.baseline,
        }


@dataclass(frozen=True)
class StabilityCheckpoint:
    """One immutable report row, including its baseline and metric deltas."""

    consumed_examples: int
    optimizer_call_count: int
    baseline: StabilityMetrics
    current: StabilityMetrics
    deltas: tuple[tuple[str, float], ...]
    max_update_relative_matrix_rms: float
    elapsed_seconds: float
    eta_seconds: float | None
    failed_thresholds: tuple[FailedThreshold, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = {
            "consumed_examples": self.consumed_examples,
            "optimizer_call_count": self.optimizer_call_count,
            "baseline": self.baseline.to_dict(),
            "current": self.current.to_dict(),
            "deltas": dict(self.deltas),
            "max_update_relative_matrix_rms": (self.max_update_relative_matrix_rms),
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "failed_thresholds": [item.to_dict() for item in self.failed_thresholds],
        }
        _require_finite(result, "checkpoint")
        return result


@dataclass(frozen=True)
class StabilityReport:
    """The classified result of one bounded development run."""

    schema_version: int
    configuration: StabilityConfiguration
    status: Literal["passed", "failed"]
    outcome_code: int
    checkpoints: tuple[StabilityCheckpoint, ...]
    failed_thresholds: tuple[FailedThreshold, ...]

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "configuration": self.configuration.to_dict(),
            "status": self.status,
            "outcome_code": self.outcome_code,
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "failed_thresholds": [item.to_dict() for item in self.failed_thresholds],
        }
        _require_finite(result, "report")
        return result

    def canonical_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")


@dataclass(frozen=True)
class StabilityProgressRecord:
    """Strict JSONL record emitted after each retained evaluation."""

    schema_version: int
    kind: Literal["evaluation"]
    checkpoint: StabilityCheckpoint

    def canonical_json_line(self) -> str:
        return (
            canonical_json_bytes(
                {
                    "schema_version": self.schema_version,
                    "kind": self.kind,
                    "checkpoint": self.checkpoint.to_dict(),
                }
            ).decode("utf-8")
            + "\n"
        )


@dataclass
class StabilityProgress:
    """Mutable example and optimizer-call cursors owned by one fresh run."""

    consumed_examples: int = 0
    optimizer_call_count: int = 0


@dataclass(frozen=True)
class ValidatedStabilityReport:
    """A compatible report and the SHA-256 identity of its exact bytes."""

    report: StabilityReport
    sha256: str


class StabilityReportValidationError(ValueError):
    """One or more strict parsing or compatibility failures."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("incompatible stability report: " + "; ".join(self.issues))


@dataclass(frozen=True)
class _IsolationSnapshot:
    parameters: tuple[tuple[str, torch.Tensor], ...]
    optimizer_state: dict[str, Any]
    workspace_state: torch.Tensor
    consumed_examples: int
    optimizer_call_count: int
    python_rng: object
    numpy_rng: tuple[str, np.ndarray, int, int, float]
    torch_cpu_rng: torch.Tensor
    torch_cuda_rng: tuple[torch.Tensor, ...]


MetricEvaluator = Callable[[StabilityTrainer, Sequence[tuple[str, str]]], StabilityMetrics]
ProgressObserver = Callable[[StabilityProgressRecord], None]


def canonical_implementation_identity(
    repository_root: str | Path,
    source_paths: Sequence[str] = STABILITY_IMPLEMENTATION_PATHS,
) -> ImplementationIdentity:
    """Hash ordered repository-relative source files and their canonical index."""
    root = Path(repository_root).resolve()
    sources: list[tuple[str, str]] = []
    for relative in source_paths:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"guarded implementation source is invalid: {relative}")
        sources.append((relative.replace("\\", "/"), hashlib.sha256(path.read_bytes()).hexdigest()))
    combined = hashlib.sha256(canonical_json_bytes(dict(sources))).hexdigest()
    return ImplementationIdentity(combined, tuple(sources))


def build_stability_configuration(
    trainer: StabilityTrainer,
    *,
    asset_identity: Mapping[str, Any],
    implementation: ImplementationIdentity,
    initialization_seed: int = 0,
) -> StabilityConfiguration:
    """Capture the complete EGGROLL and asset identity before evaluation."""
    if len(trainer.optimizer.param_groups) != 1:
        raise ValueError("EGGROLL stability requires one optimizer parameter group")
    if not isinstance(trainer.optimizer, torch.optim.SGD):
        raise TypeError("EGGROLL stability requires torch.optim.SGD")
    optimizer_group = trainer.optimizer.param_groups[0]
    if float(optimizer_group.get("momentum", 0.0)) != 0.0:
        raise ValueError("EGGROLL stability requires zero SGD momentum")
    actual_parameters = tuple(parameter for group in trainer.optimizer.param_groups for parameter in group["params"])
    expected_parameters = tuple(trainer.state.eggroll_parameters())
    if len(actual_parameters) != len(expected_parameters) or any(
        actual is not expected for actual, expected in zip(actual_parameters, expected_parameters, strict=True)
    ):
        raise ValueError("EGGROLL stability optimizer has the wrong parameter scope")
    return StabilityConfiguration(
        asset_identity=_freeze_mapping(asset_identity),
        implementation=implementation,
        optimizer_contract="torch.optim.SGD(momentum=0.0)",
        parameter_scope=EGGROLL_PARAMETER_PATHS,
        population=trainer.pop_size,
        sigma=trainer.sigma,
        rank=trainer.rank,
        fitness_batch_size=trainer.fitness_batch_size,
        evaluation_batch_size=trainer.eval_batch_size,
        variance_weight=trainer.variance_weight,
        prompt_alignment_weight=trainer.prompt_alignment_weight,
        learning_rate=float(optimizer_group["lr"]),
        initialization_seed=initialization_seed,
    )


def build_stability_asset_identity(
    *,
    runtime: Mapping[str, Any],
    held_out_records: Sequence[AsdivRecord],
) -> dict[str, Any]:
    """Build the pinned Stage 0 identity with the exact 64-record cohort."""
    if len(held_out_records) != BASELINE_PROBLEM_COUNT:
        raise ValueError(f"stability identity requires {BASELINE_PROBLEM_COUNT} held-out records")
    if any(record.split != "validation" for record in held_out_records):
        raise ValueError("stability identity accepts only validation records")
    return training_identity(
        runtime=runtime,
        held_out_item_ids=[record.id for record in held_out_records],
    )


def load_compatible_stability_report(
    path: str | Path,
    expected_configuration: StabilityConfiguration | Mapping[str, Any],
) -> ValidatedStabilityReport:
    """Parse and validate a passing report without loading models or datasets."""
    return _load_matching_stability_report(
        path,
        expected_configuration,
        require_passing=True,
    )


def load_matching_stability_report(
    path: str | Path,
    expected_configuration: StabilityConfiguration | Mapping[str, Any],
) -> ValidatedStabilityReport:
    """Parse a compatible report while preserving passed or failed health."""
    return _load_matching_stability_report(
        path,
        expected_configuration,
        require_passing=False,
    )


def read_stability_report(path: str | Path) -> ValidatedStabilityReport:
    """Strictly parse a report before a caller constructs its expected identity."""
    raw, report = _read_stability_report(path)
    issues: list[str] = []
    if report.schema_version != STABILITY_SCHEMA_VERSION:
        issues.append(f"$.schema_version: expected {STABILITY_SCHEMA_VERSION}, found {report.schema_version}")
    if issues:
        raise StabilityReportValidationError(issues)
    return ValidatedStabilityReport(
        report=report,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _read_stability_report(path: str | Path) -> tuple[bytes, StabilityReport]:
    report_path = Path(path)
    try:
        raw = report_path.read_bytes()
    except OSError as error:
        raise StabilityReportValidationError((f"$: cannot read report: {error}",)) from error
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON number {value}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise StabilityReportValidationError((f"$: malformed JSON: {error}",)) from error

    try:
        report = _parse_stability_report(payload)
    except StabilityReportValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise StabilityReportValidationError((f"$: malformed report: {error}",)) from error
    return raw, report


def _load_matching_stability_report(
    path: str | Path,
    expected_configuration: StabilityConfiguration | Mapping[str, Any],
    *,
    require_passing: bool,
) -> ValidatedStabilityReport:
    raw, report = _read_stability_report(path)
    issues: list[str] = []
    if report.schema_version != STABILITY_SCHEMA_VERSION:
        issues.append(f"$.schema_version: expected {STABILITY_SCHEMA_VERSION}, found {report.schema_version}")
    if report.status == "passed":
        if report.outcome_code != 0:
            issues.append(f"$.outcome_code: expected 0, found {report.outcome_code}")
        if report.failed_thresholds:
            issues.append("$.failed_thresholds: passing report must contain no failures")
        if not report.checkpoints or report.checkpoints[-1].consumed_examples != DEVELOPMENT_EXAMPLE_COUNT:
            issues.append(f"$.checkpoints: passing report must end at {DEVELOPMENT_EXAMPLE_COUNT} examples")
    else:
        if report.outcome_code == 0:
            issues.append("$.outcome_code: failed report must be nonzero")
        if not report.failed_thresholds:
            issues.append("$.failed_thresholds: failed report must contain failures")
        if report.checkpoints and report.checkpoints[-1].failed_thresholds != report.failed_thresholds:
            issues.append("$.checkpoints[-1].failed_thresholds: must equal report failures")
    if require_passing:
        if report.status != "passed":
            issues.append(f"$.status: expected 'passed', found {report.status!r}")
        if report.outcome_code != 0:
            issues.append(f"$.outcome_code: expected 0, found {report.outcome_code}")
        if report.failed_thresholds:
            issues.append("$.failed_thresholds: passing report must contain no failures")
    if report.status == "passed" and report.checkpoints:
        expected_examples = (0, *DEVELOPMENT_CHECKPOINTS)
        actual_examples = tuple(item.consumed_examples for item in report.checkpoints)
        if actual_examples != expected_examples:
            issues.append(f"$.checkpoints: expected consumed examples {expected_examples!r}, found {actual_examples!r}")
        final = report.checkpoints[-1]
        for failure in _final_failures(
            final.baseline,
            final.current,
            final.max_update_relative_matrix_rms,
        ):
            issues.append(
                f"$.checkpoints[-1].{failure.metric}: passing report violates {failure.operator} {failure.threshold!r}"
            )
    expected_document = (
        expected_configuration.to_dict()
        if isinstance(expected_configuration, StabilityConfiguration)
        else dict(expected_configuration)
    )
    _collect_configuration_mismatches(
        report.configuration.to_dict(),
        expected_document,
        "$.configuration",
        issues,
    )
    if issues:
        raise StabilityReportValidationError(issues)
    return ValidatedStabilityReport(
        report=report,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def measure_fresh_baseline(
    trainer: StabilityTrainer,
    held_out_records: Sequence[AsdivRecord],
    progress: StabilityProgress,
    evaluator: MetricEvaluator,
    *,
    elapsed_seconds: float = 0.0,
) -> StabilityCheckpoint:
    """Evaluate exactly 64 validation records and restore all mutable state."""
    _validate_fresh_baseline_inputs(held_out_records, progress)
    metrics = _evaluate_isolated(
        trainer,
        held_out_records,
        progress,
        evaluator,
    )
    return StabilityCheckpoint(
        consumed_examples=0,
        optimizer_call_count=0,
        baseline=metrics,
        current=metrics,
        deltas=_metric_deltas(metrics, metrics),
        max_update_relative_matrix_rms=0.0,
        elapsed_seconds=elapsed_seconds,
        eta_seconds=None,
    )


def run_bounded_stability_gate(
    trainer: StabilityTrainer,
    training_records: Sequence[AsdivRecord],
    held_out_records: Sequence[AsdivRecord],
    configuration: StabilityConfiguration,
    evaluator: MetricEvaluator,
    *,
    observer: ProgressObserver | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> StabilityReport:
    """Run fresh EGGROLL updates through the declared development checkpoints."""
    records = _validate_development_records(training_records)
    progress = StabilityProgress()
    started_at = clock()
    baseline_record = measure_fresh_baseline(
        trainer,
        held_out_records,
        progress,
        evaluator,
        elapsed_seconds=max(0.0, clock() - started_at),
    )
    retained = [baseline_record]
    _observe(observer, baseline_record)
    maximum_relative_change = 0.0

    for checkpoint_target in DEVELOPMENT_CHECKPOINTS:
        while progress.consumed_examples < checkpoint_target:
            remaining = checkpoint_target - progress.consumed_examples
            batch_size = min(trainer.fitness_batch_size, remaining)
            batch_start = progress.consumed_examples
            batch = FitnessBatch(
                records=tuple(records[batch_start : batch_start + batch_size]),
                start_position=batch_start,
                next_position=batch_start + batch_size,
            )
            before = tuple(parameter.detach().clone() for parameter in trainer.state.eggroll_parameters())
            progress.optimizer_call_count += 1
            position = ExperimentPosition(
                update_method="eggroll",
                cycle=0,
                global_step=batch.next_position,
                epoch=1,
                example_position=batch.next_position,
                phase_step=batch.next_position,
                optimizer_call_count=progress.optimizer_call_count,
                consumed_record_count=batch.consumed_record_count,
            )
            result = trainer.train_fitness_batch(
                batch,
                position,
                optimizer_call_count=progress.optimizer_call_count,
            )
            if result.consumed_record_count != batch.consumed_record_count:
                raise RuntimeError("trainer consumed-record count differs from its batch")
            progress.consumed_examples = batch.next_position
            maximum_relative_change = max(
                maximum_relative_change,
                _maximum_relative_rms_change(
                    before,
                    tuple(trainer.state.eggroll_parameters()),
                ),
            )

        current = _evaluate_isolated(
            trainer,
            held_out_records,
            progress,
            evaluator,
        )
        elapsed = max(0.0, clock() - started_at)
        failures = (
            _final_failures(
                baseline_record.baseline,
                current,
                maximum_relative_change,
            )
            if checkpoint_target == DEVELOPMENT_EXAMPLE_COUNT
            else _early_failures(baseline_record.baseline, current)
        )
        checkpoint = StabilityCheckpoint(
            consumed_examples=progress.consumed_examples,
            optimizer_call_count=progress.optimizer_call_count,
            baseline=baseline_record.baseline,
            current=current,
            deltas=_metric_deltas(baseline_record.baseline, current),
            max_update_relative_matrix_rms=maximum_relative_change,
            elapsed_seconds=elapsed,
            eta_seconds=_eta(elapsed, progress.consumed_examples),
            failed_thresholds=failures,
        )
        retained.append(checkpoint)
        _observe(observer, checkpoint)
        if failures:
            return StabilityReport(
                schema_version=STABILITY_SCHEMA_VERSION,
                configuration=configuration,
                status="failed",
                outcome_code=1,
                checkpoints=tuple(retained),
                failed_thresholds=failures,
            )

    return StabilityReport(
        schema_version=STABILITY_SCHEMA_VERSION,
        configuration=configuration,
        status="passed",
        outcome_code=0,
        checkpoints=tuple(retained),
        failed_thresholds=(),
    )


def _validate_fresh_baseline_inputs(
    held_out_records: Sequence[AsdivRecord],
    progress: StabilityProgress,
) -> None:
    if progress.consumed_examples != 0 or progress.optimizer_call_count != 0:
        raise ValueError("stability baseline requires fresh zero-example progress")
    if len(held_out_records) != BASELINE_PROBLEM_COUNT:
        raise ValueError(f"stability baseline requires the fixed first {BASELINE_PROBLEM_COUNT} held-out records")
    if any(record.split != "validation" for record in held_out_records):
        raise ValueError("stability baseline accepts only validation records")


def _validate_development_records(
    training_records: Sequence[AsdivRecord],
) -> tuple[AsdivRecord, ...]:
    if len(training_records) < DEVELOPMENT_EXAMPLE_COUNT:
        raise ValueError(f"stability development requires {DEVELOPMENT_EXAMPLE_COUNT} records")
    selected = tuple(training_records[:DEVELOPMENT_EXAMPLE_COUNT])
    if any(record.split != "train" for record in selected):
        raise ValueError("stability development accepts only training records")
    return selected


def _evaluate_isolated(
    trainer: StabilityTrainer,
    held_out_records: Sequence[AsdivRecord],
    progress: StabilityProgress,
    evaluator: MetricEvaluator,
) -> StabilityMetrics:
    snapshot = _capture_isolation_snapshot(trainer, progress)
    problems = tuple((record.question, record.target) for record in held_out_records)
    try:
        return evaluator(trainer, problems)
    finally:
        _restore_isolation_snapshot(trainer, progress, snapshot)


def _capture_isolation_snapshot(
    trainer: StabilityTrainer,
    progress: StabilityProgress,
) -> _IsolationSnapshot:
    return _IsolationSnapshot(
        parameters=tuple(
            (path, parameter.detach().clone()) for path, parameter in trainer.state.trainable_params.items()
        ),
        optimizer_state=_clone_state(trainer.optimizer.state_dict()),
        workspace_state=trainer.state.workspace.snapshot().detach().clone(),
        consumed_examples=progress.consumed_examples,
        optimizer_call_count=progress.optimizer_call_count,
        python_rng=copy.deepcopy(random.getstate()),
        numpy_rng=_clone_numpy_rng(cast(tuple[str, np.ndarray, int, int, float], np.random.get_state())),
        torch_cpu_rng=torch.get_rng_state().detach().cpu().clone(),
        torch_cuda_rng=(
            tuple(item.detach().cpu().clone() for item in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else ()
        ),
    )


def _restore_isolation_snapshot(
    trainer: StabilityTrainer,
    progress: StabilityProgress,
    snapshot: _IsolationSnapshot,
) -> None:
    with torch.no_grad():
        for (path, saved), (current_path, parameter) in zip(
            snapshot.parameters,
            trainer.state.trainable_params.items(),
            strict=True,
        ):
            if path != current_path or saved.shape != parameter.shape:
                raise RuntimeError("training parameter registry changed during evaluation")
            parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))
    trainer.optimizer.load_state_dict(_clone_state(snapshot.optimizer_state))
    trainer.state.workspace.restore(snapshot.workspace_state.clone())
    progress.consumed_examples = snapshot.consumed_examples
    progress.optimizer_call_count = snapshot.optimizer_call_count
    random.setstate(cast(tuple[Any, ...], copy.deepcopy(snapshot.python_rng)))
    np.random.set_state(_clone_numpy_rng(snapshot.numpy_rng))
    torch.set_rng_state(snapshot.torch_cpu_rng.clone())
    if snapshot.torch_cuda_rng:
        torch.cuda.set_rng_state_all([item.clone() for item in snapshot.torch_cuda_rng])


def _parameter_rms(trainer: StabilityTrainer) -> tuple[ParameterRms, ...]:
    return tuple(
        ParameterRms(
            path,
            float(torch.sqrt(torch.mean(parameter.detach().float().square())).item()),
        )
        for path, parameter in trainer.state.trainable_params.items()
    )


def _maximum_relative_rms_change(
    before: Sequence[torch.Tensor],
    after: Sequence[torch.Tensor],
) -> float:
    if len(before) != len(after):
        raise RuntimeError("EGGROLL parameter scope changed during training")
    maximum = 0.0
    for previous, current in zip(before, after, strict=True):
        previous_float = previous.detach().float()
        current_float = current.detach().float()
        change_rms = torch.sqrt(torch.mean((current_float - previous_float).square()))
        parameter_rms = torch.sqrt(torch.mean(previous_float.square()))
        denominator = max(float(parameter_rms.item()), torch.finfo(torch.float32).tiny)
        maximum = max(maximum, float(change_rms.item()) / denominator)
    return maximum


def _metric_deltas(
    baseline: StabilityMetrics,
    current: StabilityMetrics,
) -> tuple[tuple[str, float], ...]:
    baseline_values = baseline.to_dict()
    current_values = current.to_dict()
    return tuple(
        (name, float(current_values[name]) - float(baseline_values[name]))
        for name in baseline_values
        if name not in {"problem_count", "parameter_rms"}
    )


def _early_failures(
    baseline: StabilityMetrics,
    current: StabilityMetrics,
) -> tuple[FailedThreshold, ...]:
    failures = []
    loss_ceiling = baseline.language_model_loss * MAX_LOSS_RATIO
    if current.language_model_loss > loss_ceiling:
        failures.append(
            FailedThreshold(
                "language_model_loss",
                current.language_model_loss,
                "<=",
                loss_ceiling,
                baseline.language_model_loss,
            )
        )
    separation_floor = baseline.separation_retention * MIN_SEPARATION_RATIO
    if current.separation_retention < separation_floor:
        failures.append(
            FailedThreshold(
                "separation_retention",
                current.separation_retention,
                ">=",
                separation_floor,
                baseline.separation_retention,
            )
        )
    return tuple(failures)


def _final_failures(
    baseline: StabilityMetrics,
    current: StabilityMetrics,
    maximum_relative_change: float,
) -> tuple[FailedThreshold, ...]:
    """Return every unmet absolute-health condition in canonical order."""
    failures: list[FailedThreshold] = []
    checks = (
        (
            "language_model_loss",
            current.language_model_loss,
            "<",
            baseline.language_model_loss,
            current.language_model_loss < baseline.language_model_loss,
            baseline.language_model_loss,
        ),
        (
            "exact_accuracy",
            current.exact_accuracy,
            ">",
            baseline.exact_accuracy,
            current.exact_accuracy > baseline.exact_accuracy,
            baseline.exact_accuracy,
        ),
        (
            "first_token_accuracy",
            current.first_token_accuracy,
            ">",
            baseline.first_token_accuracy,
            current.first_token_accuracy > baseline.first_token_accuracy,
            baseline.first_token_accuracy,
        ),
        (
            "separation_retention",
            current.separation_retention,
            ">=",
            baseline.separation_retention * MIN_SEPARATION_RATIO,
            current.separation_retention >= baseline.separation_retention * MIN_SEPARATION_RATIO,
            baseline.separation_retention,
        ),
        (
            "max_update_relative_matrix_rms",
            maximum_relative_change,
            "<=",
            MAX_UPDATE_RELATIVE_MATRIX_RMS,
            maximum_relative_change <= MAX_UPDATE_RELATIVE_MATRIX_RMS,
            0.0,
        ),
    )
    for metric, observed, operator, threshold, passed, baseline_value in checks:
        if not passed:
            failures.append(
                FailedThreshold(
                    metric=metric,
                    observed=observed,
                    operator=operator,
                    threshold=threshold,
                    baseline=baseline_value,
                )
            )
    return tuple(failures)


def _eta(elapsed_seconds: float, consumed_examples: int) -> float:
    if consumed_examples >= DEVELOPMENT_EXAMPLE_COUNT:
        return 0.0
    return elapsed_seconds * (DEVELOPMENT_EXAMPLE_COUNT - consumed_examples) / consumed_examples


def _observe(
    observer: ProgressObserver | None,
    checkpoint: StabilityCheckpoint,
) -> None:
    if observer is not None:
        observer(
            StabilityProgressRecord(
                schema_version=STABILITY_SCHEMA_VERSION,
                kind="evaluation",
                checkpoint=checkpoint,
            )
        )


def _freeze_mapping(value: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple((str(key), _freeze_value(item)) for key, item in sorted(value.items(), key=lambda pair: str(pair[0])))


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"identity values must be JSON compatible, got {type(value).__name__}")


def _thaw_mapping(value: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {key: _thaw_value(item) for key, item in value}


def _thaw_value(value: Any) -> Any:
    if isinstance(value, tuple):
        if value and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return _thaw_mapping(value)
        # An empty frozen container is ambiguous; identity documents contain
        # empty arrays (device_topology on CPU-only hosts) but never empty
        # objects, so thaw () to a list.
        return [_thaw_value(item) for item in value]
    return value


def _clone_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {copy.deepcopy(key): _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return copy.deepcopy(value)


def _clone_numpy_rng(
    state: tuple[str, np.ndarray, int, int, float],
) -> tuple[str, np.ndarray, int, int, float]:
    return (state[0], state[1].copy(), state[2], state[3], state[4])


def _require_finite(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must contain only finite numbers")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite(item, f"{path}[{index}]")


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _parse_stability_report(value: Any) -> StabilityReport:
    obj = _strict_object(
        value,
        "$",
        {"schema_version", "configuration", "status", "outcome_code", "checkpoints", "failed_thresholds"},
    )
    status = _strict_string(obj["status"], "$.status")
    if status not in {"passed", "failed"}:
        raise StabilityReportValidationError((f"$.status: unknown value {status!r}",))
    status = cast(Literal["passed", "failed"], status)
    checkpoints = _strict_list(obj["checkpoints"], "$.checkpoints")
    failures = _strict_list(obj["failed_thresholds"], "$.failed_thresholds")
    return StabilityReport(
        schema_version=_strict_int(obj["schema_version"], "$.schema_version"),
        configuration=_parse_configuration(obj["configuration"]),
        status=status,
        outcome_code=_strict_int(obj["outcome_code"], "$.outcome_code"),
        checkpoints=tuple(_parse_checkpoint(item, f"$.checkpoints[{index}]") for index, item in enumerate(checkpoints)),
        failed_thresholds=tuple(
            _parse_failed_threshold(item, f"$.failed_thresholds[{index}]") for index, item in enumerate(failures)
        ),
    )


def _parse_configuration(value: Any) -> StabilityConfiguration:
    path = "$.configuration"
    obj = _strict_object(
        value,
        path,
        {
            "asset_identity",
            "implementation",
            "eggroll",
            "initialization_seed",
            "held_out_problem_count",
            "development_example_count",
            "checkpoints",
        },
    )
    asset_identity = _strict_object(obj["asset_identity"], f"{path}.asset_identity", None)
    implementation = _strict_object(
        obj["implementation"],
        f"{path}.implementation",
        {"sha256", "sources"},
    )
    sources = _strict_object(
        implementation["sources"],
        f"{path}.implementation.sources",
        None,
    )
    eggroll = _strict_object(
        obj["eggroll"],
        f"{path}.eggroll",
        {
            "optimizer_contract",
            "parameter_scope",
            "population",
            "sigma",
            "rank",
            "fitness_batch_size",
            "evaluation_batch_size",
            "variance_weight",
            "prompt_alignment_weight",
            "learning_rate",
        },
    )
    scope = _strict_list(eggroll["parameter_scope"], f"{path}.eggroll.parameter_scope")
    checkpoints = _strict_list(obj["checkpoints"], f"{path}.checkpoints")
    try:
        return StabilityConfiguration(
            asset_identity=_freeze_mapping(asset_identity),
            implementation=ImplementationIdentity(
                sha256=_strict_string(implementation["sha256"], f"{path}.implementation.sha256"),
                sources=tuple(
                    sorted(
                        (
                            _strict_string(source_path, f"{path}.implementation.sources key"),
                            _strict_string(
                                digest,
                                f"{path}.implementation.sources.{source_path}",
                            ),
                        )
                        for source_path, digest in sources.items()
                    )
                ),
            ),
            optimizer_contract=_strict_string(eggroll["optimizer_contract"], f"{path}.eggroll.optimizer_contract"),
            parameter_scope=tuple(
                _strict_string(item, f"{path}.eggroll.parameter_scope[{index}]") for index, item in enumerate(scope)
            ),
            population=_strict_int(eggroll["population"], f"{path}.eggroll.population"),
            sigma=_strict_float(eggroll["sigma"], f"{path}.eggroll.sigma"),
            rank=_strict_int(eggroll["rank"], f"{path}.eggroll.rank"),
            fitness_batch_size=_strict_int(eggroll["fitness_batch_size"], f"{path}.eggroll.fitness_batch_size"),
            evaluation_batch_size=_strict_int(
                eggroll["evaluation_batch_size"], f"{path}.eggroll.evaluation_batch_size"
            ),
            variance_weight=_strict_float(eggroll["variance_weight"], f"{path}.eggroll.variance_weight"),
            prompt_alignment_weight=_strict_float(
                eggroll["prompt_alignment_weight"],
                f"{path}.eggroll.prompt_alignment_weight",
            ),
            learning_rate=_strict_float(eggroll["learning_rate"], f"{path}.eggroll.learning_rate"),
            initialization_seed=_strict_int(obj["initialization_seed"], f"{path}.initialization_seed"),
            held_out_problem_count=_strict_int(obj["held_out_problem_count"], f"{path}.held_out_problem_count"),
            development_example_count=_strict_int(
                obj["development_example_count"], f"{path}.development_example_count"
            ),
            checkpoints=tuple(
                _strict_int(item, f"{path}.checkpoints[{index}]") for index, item in enumerate(checkpoints)
            ),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, StabilityReportValidationError):
            raise
        raise StabilityReportValidationError((f"{path}: {error}",)) from error


def _parse_checkpoint(value: Any, path: str) -> StabilityCheckpoint:
    obj = _strict_object(
        value,
        path,
        {
            "consumed_examples",
            "optimizer_call_count",
            "baseline",
            "current",
            "deltas",
            "max_update_relative_matrix_rms",
            "elapsed_seconds",
            "eta_seconds",
            "failed_thresholds",
        },
    )
    deltas = _strict_object(obj["deltas"], f"{path}.deltas", None)
    failures = _strict_list(obj["failed_thresholds"], f"{path}.failed_thresholds")
    eta = obj["eta_seconds"]
    return StabilityCheckpoint(
        consumed_examples=_strict_int(obj["consumed_examples"], f"{path}.consumed_examples"),
        optimizer_call_count=_strict_int(obj["optimizer_call_count"], f"{path}.optimizer_call_count"),
        baseline=_parse_metrics(obj["baseline"], f"{path}.baseline"),
        current=_parse_metrics(obj["current"], f"{path}.current"),
        deltas=tuple((name, _strict_float(item, f"{path}.deltas.{name}")) for name, item in sorted(deltas.items())),
        max_update_relative_matrix_rms=_strict_float(
            obj["max_update_relative_matrix_rms"],
            f"{path}.max_update_relative_matrix_rms",
        ),
        elapsed_seconds=_strict_float(obj["elapsed_seconds"], f"{path}.elapsed_seconds"),
        eta_seconds=None if eta is None else _strict_float(eta, f"{path}.eta_seconds"),
        failed_thresholds=tuple(
            _parse_failed_threshold(item, f"{path}.failed_thresholds[{index}]") for index, item in enumerate(failures)
        ),
    )


def _parse_metrics(value: Any, path: str) -> StabilityMetrics:
    fields = {
        "problem_count",
        "parameter_rms",
        "language_model_loss",
        "exact_accuracy",
        "first_token_accuracy",
        "valid_answer_rate",
        "output_diversity",
        "output_dominance",
        "shared_slot_variance",
        "student_teacher_mse",
        "student_cross_problem_cosine",
        "teacher_cross_problem_cosine",
        "separation_retention",
    }
    obj = _strict_object(value, path, fields)
    rms_values = _strict_list(obj["parameter_rms"], f"{path}.parameter_rms")
    parsed_rms = []
    for index, item in enumerate(rms_values):
        item_path = f"{path}.parameter_rms[{index}]"
        rms = _strict_object(item, item_path, {"path", "rms"})
        parsed_rms.append(
            ParameterRms(
                _strict_string(rms["path"], f"{item_path}.path"),
                _strict_float(rms["rms"], f"{item_path}.rms"),
            )
        )
    return StabilityMetrics(
        problem_count=_strict_int(obj["problem_count"], f"{path}.problem_count"),
        parameter_rms=tuple(parsed_rms),
        **{name: _strict_float(obj[name], f"{path}.{name}") for name in fields - {"problem_count", "parameter_rms"}},
    )


def _parse_failed_threshold(value: Any, path: str) -> FailedThreshold:
    obj = _strict_object(
        value,
        path,
        {"metric", "observed", "operator", "threshold", "baseline"},
    )
    return FailedThreshold(
        metric=_strict_string(obj["metric"], f"{path}.metric"),
        observed=_strict_float(obj["observed"], f"{path}.observed"),
        operator=_strict_string(obj["operator"], f"{path}.operator"),
        threshold=_strict_float(obj["threshold"], f"{path}.threshold"),
        baseline=_strict_float(obj["baseline"], f"{path}.baseline"),
    )


def _strict_object(
    value: Any,
    path: str,
    fields: set[str] | None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StabilityReportValidationError((f"{path}: expected object",))
    if fields is not None:
        actual = set(value)
        missing = sorted(fields - actual)
        extra = sorted(actual - fields)
        issues = [f"{path}.{name}: missing field" for name in missing]
        issues.extend(f"{path}.{name}: unexpected field" for name in extra)
        if issues:
            raise StabilityReportValidationError(issues)
    return value


def _strict_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise StabilityReportValidationError((f"{path}: expected array",))
    return value


def _strict_string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise StabilityReportValidationError((f"{path}: expected string",))
    return value


def _strict_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StabilityReportValidationError((f"{path}: expected integer",))
    return value


def _strict_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StabilityReportValidationError((f"{path}: expected number",))
    result = float(value)
    if not math.isfinite(result):
        raise StabilityReportValidationError((f"{path}: expected finite number",))
    return result


def _collect_configuration_mismatches(
    actual: Any,
    expected: Any,
    path: str,
    issues: list[str],
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            issues.append(f"{path}: expected object")
            return
        for key in sorted(set(actual) | set(expected)):
            child = f"{path}.{key}"
            if key not in actual:
                issues.append(f"{child}: missing field")
            elif key not in expected:
                issues.append(f"{child}: unexpected field")
            else:
                _collect_configuration_mismatches(actual[key], expected[key], child, issues)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            issues.append(f"{path}: expected array")
            return
        if len(actual) != len(expected):
            issues.append(f"{path}: expected {len(expected)} items, found {len(actual)}")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=False)):
            _collect_configuration_mismatches(actual_item, expected_item, f"{path}[{index}]", issues)
        return
    if isinstance(actual, bool) != isinstance(expected, bool) or actual != expected:
        issues.append(f"{path}: expected {expected!r}, found {actual!r}")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
