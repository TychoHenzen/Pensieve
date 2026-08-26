"""Bounded absolute-health measurements for fresh Stage 0 EGGROLL runs.

The command-line entry point is intentionally separate.  This module owns the
immutable report contract, evaluation isolation, implementation identity, and
the deterministic 0/8/32/256-example development loop.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Literal, Protocol

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
    "train/eggroll_trainer.py",
    "train/eggroll_updates.py",
    "train/stage0_data.py",
    "train/training_results.py",
    "train/training_state.py",
    "train/vicreg.py",
    "workspace/concept_slots.py",
)


class StabilityTrainer(Protocol):
    """The EGGROLL surface used by the bounded development gate."""

    state: Any
    optimizer: torch.optim.Optimizer
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
        if not self.sources or any(
            not path or not _is_sha256(digest) for path, digest in self.sources
        ):
            raise ValueError("implementation sources require paths and SHA-256 digests")
        if tuple(path for path, _ in self.sources) != tuple(
            sorted(path for path, _ in self.sources)
        ):
            raise ValueError("implementation sources must use canonical path order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "sources": {path: digest for path, digest in self.sources},
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
        if isinstance(self.initialization_seed, bool) or not isinstance(
            self.initialization_seed, int
        ):
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
            raise ValueError(
                f"stability metrics require {BASELINE_PROBLEM_COUNT} problems"
            )
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
            "deltas": {name: value for name, value in self.deltas},
            "max_update_relative_matrix_rms": (
                self.max_update_relative_matrix_rms
            ),
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "failed_thresholds": [item.to_dict() for item in self.failed_thresholds],
        }
        _require_finite(result, "checkpoint")
        return result


@dataclass(frozen=True)
class StabilityReport:
    """The partial or development-complete result before final classification."""

    schema_version: int
    configuration: StabilityConfiguration
    status: Literal["development_complete", "failed_early"]
    outcome_code: int | None
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
        return canonical_json_bytes(
            {
                "schema_version": self.schema_version,
                "kind": self.kind,
                "checkpoint": self.checkpoint.to_dict(),
            }
        ).decode("utf-8") + "\n"


@dataclass
class StabilityProgress:
    """Mutable example and optimizer-call cursors owned by one fresh run."""

    consumed_examples: int = 0
    optimizer_call_count: int = 0


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
    combined = hashlib.sha256(
        canonical_json_bytes({path: digest for path, digest in sources})
    ).hexdigest()
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
        raise ValueError("EGGROLL stability requires torch.optim.SGD")
    optimizer_group = trainer.optimizer.param_groups[0]
    if float(optimizer_group.get("momentum", 0.0)) != 0.0:
        raise ValueError("EGGROLL stability requires zero SGD momentum")
    actual_parameters = tuple(
        parameter for group in trainer.optimizer.param_groups for parameter in group["params"]
    )
    expected_parameters = tuple(trainer.state.eggroll_parameters())
    if len(actual_parameters) != len(expected_parameters) or any(
        actual is not expected
        for actual, expected in zip(actual_parameters, expected_parameters, strict=True)
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
        raise ValueError(
            f"stability identity requires {BASELINE_PROBLEM_COUNT} held-out records"
        )
    if any(record.split != "validation" for record in held_out_records):
        raise ValueError("stability identity accepts only validation records")
    return training_identity(
        runtime=runtime,
        held_out_item_ids=[record.id for record in held_out_records],
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
            before = tuple(
                parameter.detach().clone()
                for parameter in trainer.state.eggroll_parameters()
            )
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
        failures = _early_failures(baseline_record.baseline, current)
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
                status="failed_early",
                outcome_code=1,
                checkpoints=tuple(retained),
                failed_thresholds=failures,
            )

    return StabilityReport(
        schema_version=STABILITY_SCHEMA_VERSION,
        configuration=configuration,
        status="development_complete",
        outcome_code=None,
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
        raise ValueError(
            f"stability baseline requires the fixed first {BASELINE_PROBLEM_COUNT} "
            "held-out records"
        )
    if any(record.split != "validation" for record in held_out_records):
        raise ValueError("stability baseline accepts only validation records")


def _validate_development_records(
    training_records: Sequence[AsdivRecord],
) -> tuple[AsdivRecord, ...]:
    if len(training_records) < DEVELOPMENT_EXAMPLE_COUNT:
        raise ValueError(
            f"stability development requires {DEVELOPMENT_EXAMPLE_COUNT} records"
        )
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
            (path, parameter.detach().clone())
            for path, parameter in trainer.state.trainable_params.items()
        ),
        optimizer_state=_clone_state(trainer.optimizer.state_dict()),
        workspace_state=trainer.state.workspace.snapshot().detach().clone(),
        consumed_examples=progress.consumed_examples,
        optimizer_call_count=progress.optimizer_call_count,
        python_rng=copy.deepcopy(random.getstate()),
        numpy_rng=_clone_numpy_rng(np.random.get_state()),
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
    random.setstate(copy.deepcopy(snapshot.python_rng))
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


def _eta(elapsed_seconds: float, consumed_examples: int) -> float:
    if consumed_examples >= DEVELOPMENT_EXAMPLE_COUNT:
        return 0.0
    return (
        elapsed_seconds
        * (DEVELOPMENT_EXAMPLE_COUNT - consumed_examples)
        / consumed_examples
    )


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
    return tuple(
        (str(key), _freeze_value(item))
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    )


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
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return _thaw_mapping(value)
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


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
