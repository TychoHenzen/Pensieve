"""Bounded trainability investigation for Stage 0 objectives and update methods.

Distinguishes untrainable objectives from optimizer-specific failures through
fixed fresh-state probes and equal-budget method comparisons.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

from codecs_module.decoder import DEFAULT_MAX_TOKENS
from eval.stage0_identity import STAGE0_IDENTITY, canonical_json_bytes
from eval.stream.generators.asdiv_a import AsdivRecord
from train.answer_objective import (
    DEFAULT_PROMPT_ALIGNMENT_WEIGHT,
    SUBJECT_LATENT_RUNS_PER_ANSWER,
    prepare_training_example,
    prompt_aligned_answer_objective,
    prompt_teacher_state,
)
from train.eggroll_stability import (
    DEVELOPMENT_CHECKPOINTS,
    DEVELOPMENT_EXAMPLE_COUNT,
    EGGROLL_PARAMETER_PATHS,
    STABILITY_IMPLEMENTATION_PATHS,
    ImplementationIdentity,
    ParameterRms,
    StabilityMetrics,
    _is_sha256,
    _require_finite,
    read_stability_report,
)
from train.eggroll_stability import (
    StabilityReportValidationError as CanonicalStabilityReportValidationError,
)
from train.eggroll_stability_evaluation import evaluate_stability_metrics
from train.eggroll_trainer import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_FITNESS_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_POP_SIZE,
    DEFAULT_RANK,
    DEFAULT_SIGMA,
    DEFAULT_VARIANCE_WEIGHT,
    EggrollTrainer,
)
from train.stage0_data import FitnessBatch
from train.standalone_checkpoint import configure_deterministic_runtime, runtime_identity
from train.trainer import LatentCoreTrainer
from train.vicreg import slot_variance_penalty

TRAINABILITY_SCHEMA_VERSION = 1
TRAINABILITY_IMPLEMENTATION_PATHS = STABILITY_IMPLEMENTATION_PATHS

OVERFIT_LEARNING_RATES = (0.0001, 0.001, 0.01)
OVERFIT_CHECKPOINTS = (0, 1, 4, 16, 64)
MIN_SEPARATION_RATIO = 0.95
MAX_UPDATE_RELATIVE_MATRIX_RMS = 0.01
MAX_LOSS_REDUCTION_RATIO = 0.5

MethodName = Literal["gradient", "eggroll"]
ProbeKind = Literal["overfit", "gradient_causal", "eggroll_causal"]
ArmKind = Literal["no_update", "gradient_only", "eggroll_only"]


@dataclass(frozen=True)
class TrainabilityAssetIdentity:
    """Complete identity of fixed records and configuration from the stability report."""

    stability_report_digest: str
    held_out_record_identifiers: tuple[tuple[str, str], ...]
    training_record_identifiers_overfit: tuple[tuple[str, str], ...]
    training_record_identifiers_32: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.stability_report_digest):
            raise ValueError("trainability asset identity requires a SHA-256 digest for the report")
        if not self.held_out_record_identifiers:
            raise ValueError("trainability asset identity requires held-out records")
        if not self.training_record_identifiers_overfit:
            raise ValueError("trainability asset identity requires overfit training records")
        if not self.training_record_identifiers_32:
            raise ValueError("trainability asset identity requires 32-record training records")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stability_report_digest": self.stability_report_digest,
            "held_out_record_count": len(self.held_out_record_identifiers),
            "training_record_count_overfit": len(self.training_record_identifiers_overfit),
            "training_record_count_32": len(self.training_record_identifiers_32),
            "held_out_record_identifiers": [list(item) for item in self.held_out_record_identifiers],
            "training_record_identifiers_overfit": [list(item) for item in self.training_record_identifiers_overfit],
            "training_record_identifiers_32": [list(item) for item in self.training_record_identifiers_32],
        }


@dataclass(frozen=True)
class TrainabilityConfiguration:
    """Every field that makes a trainability investigation reusable or stale."""

    asset_identity: TrainabilityAssetIdentity
    implementation: ImplementationIdentity
    stability_configuration: Mapping[str, Any]
    overfit_learning_rates: tuple[float, ...] = OVERFIT_LEARNING_RATES
    overfit_checkpoints: tuple[int, ...] = OVERFIT_CHECKPOINTS
    held_out_evaluation_checkpoints: tuple[int, ...] = (0, 8, 32)
    min_separation_ratio: float = MIN_SEPARATION_RATIO
    max_update_relative_matrix_rms: float = MAX_UPDATE_RELATIVE_MATRIX_RMS
    max_loss_reduction_ratio: float = MAX_LOSS_REDUCTION_RATIO

    def __post_init__(self) -> None:
        _require_finite(self.to_dict(), "trainability configuration")

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_identity": self.asset_identity.to_dict(),
            "implementation": self.implementation.to_dict(),
            "stability_configuration": dict(self.stability_configuration),
            "overfit_learning_rates": list(self.overfit_learning_rates),
            "overfit_checkpoints": list(self.overfit_checkpoints),
            "held_out_evaluation_checkpoints": list(self.held_out_evaluation_checkpoints),
            "min_separation_ratio": self.min_separation_ratio,
            "max_update_relative_matrix_rms": self.max_update_relative_matrix_rms,
            "max_loss_reduction_ratio": self.max_loss_reduction_ratio,
        }


@dataclass(frozen=True)
class OverfitAttempt:
    """One complete memorization attempt at a declared learning rate."""

    learning_rate: float
    status: Literal["passed", "failed", "non_finite", "inconclusive"]
    baseline_metrics: StabilityMetrics | None
    checkpoints: tuple[tuple[int, StabilityMetrics | None], ...] = ()
    failed_reason: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("overfit attempt requires a positive finite learning rate")
        if self.status == "passed":
            if not self.baseline_metrics:
                raise ValueError("passed overfit attempt must have baseline metrics")
            if not self.checkpoints:
                raise ValueError("passed overfit attempt must have checkpoints")
        if self.status in ("non_finite", "inconclusive") and not self.failed_reason:
            raise ValueError(f"{self.status} overfit attempt must specify the failed field")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "learning_rate": self.learning_rate,
            "status": self.status,
            "baseline_metrics": (self.baseline_metrics.to_dict() if self.baseline_metrics else None),
            "checkpoints": [
                {
                    "optimizer_call_count": count,
                    "metrics": metrics.to_dict() if metrics else None,
                }
                for count, metrics in self.checkpoints
            ],
        }
        if self.failed_reason:
            result["failed_reason"] = self.failed_reason
        _require_finite(result, "overfit attempt")
        return result


@dataclass(frozen=True)
class OverfitProbeResult:
    """Classification of the shared objective memorization probe."""

    status: Literal["passed", "objective_untrainable", "inconclusive"]
    attempts: tuple[OverfitAttempt, ...]
    failed_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status == "passed" and not any(attempt.status == "passed" for attempt in self.attempts):
            raise ValueError("passed overfit probe must have a passing attempt")
        if self.status == "inconclusive" and not self.failed_conditions:
            raise ValueError("inconclusive probe must specify failed conditions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "attempt_count": len(self.attempts),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "failed_conditions": list(self.failed_conditions),
        }


@dataclass(frozen=True)
class CausalProbeResult:
    """Classification of one method's causal update validation."""

    method: MethodName
    status: Literal["passed", "direction_mismatch", "unsafe_update", "inconclusive"]
    baseline_objective: float | None
    predicted_objective_delta: float | None
    observed_objective_delta: float | None
    max_update_relative_matrix_rms: float | None
    baseline_separation: float | None
    post_update_separation: float | None
    failed_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("passed", "direction_mismatch", "unsafe_update", "inconclusive"):
            raise ValueError(f"invalid causal probe status: {self.status}")
        if self.status == "passed":
            if not all(
                v is not None
                for v in [
                    self.baseline_objective,
                    self.predicted_objective_delta,
                    self.observed_objective_delta,
                    self.max_update_relative_matrix_rms,
                    self.baseline_separation,
                    self.post_update_separation,
                ]
            ):
                raise ValueError("passed causal probe must have all metrics")
            predicted = self.predicted_objective_delta
            observed = self.observed_objective_delta
            if not (
                predicted is not None and observed is not None and math.isfinite(predicted) and math.isfinite(observed)
            ):
                raise ValueError("causal probe deltas must be finite")
        if self.status == "inconclusive" and not self.failed_conditions:
            raise ValueError("inconclusive probe must specify failed conditions")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "method": self.method,
            "status": self.status,
            "baseline_objective": self.baseline_objective,
            "predicted_objective_delta": self.predicted_objective_delta,
            "observed_objective_delta": self.observed_objective_delta,
            "max_update_relative_matrix_rms": (self.max_update_relative_matrix_rms),
            "baseline_separation": self.baseline_separation,
            "post_update_separation": self.post_update_separation,
        }
        if self.failed_conditions:
            result["failed_conditions"] = list(self.failed_conditions)
        _require_finite(result, "causal probe result")
        return result


@dataclass(frozen=True)
class ArmCheckpoint:
    """One evaluation checkpoint in an equal-budget arm run."""

    consumed_examples: int
    optimizer_call_count: int
    metrics: StabilityMetrics
    max_update_relative_matrix_rms: float | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    training_record_identifiers: tuple[tuple[str, str], ...] = ()
    held_out_record_identifiers: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.consumed_examples < 0 or self.optimizer_call_count < 0:
            raise ValueError("arm checkpoint requires non-negative example counts")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("arm checkpoint elapsed time must be finite and non-negative")
        if self.eta_seconds is not None and (not math.isfinite(self.eta_seconds) or self.eta_seconds < 0):
            raise ValueError("arm checkpoint ETA must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "consumed_examples": self.consumed_examples,
            "optimizer_call_count": self.optimizer_call_count,
            "metrics": self.metrics.to_dict(),
        }
        if self.max_update_relative_matrix_rms is not None:
            result["max_update_relative_matrix_rms"] = self.max_update_relative_matrix_rms
        result["elapsed_seconds"] = self.elapsed_seconds
        result["eta_seconds"] = self.eta_seconds
        result["training_record_identifiers"] = [list(item) for item in self.training_record_identifiers]
        result["held_out_record_identifiers"] = [list(item) for item in self.held_out_record_identifiers]
        _require_finite(result, "arm checkpoint")
        return result


@dataclass(frozen=True)
class ArmResult:
    """Classification of one equal-budget method arm."""

    arm: ArmKind
    method: MethodName | None
    status: Literal[
        "passed",
        "viable",
        "no_improvement",
        "loss_behavior_conflict",
        "direction_mismatch",
        "unsafe_update",
        "non_finite",
        "inconclusive",
    ]
    baseline_checkpoint: ArmCheckpoint | None
    checkpoints: tuple[ArmCheckpoint, ...]
    recalibration_eligible: bool
    failed_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.baseline_checkpoint is None and self.status != "inconclusive":
            raise ValueError("non-inconclusive arm must have a baseline checkpoint")
        if self.arm == "no_update":
            if self.method is not None:
                raise ValueError("no-update arm must not specify a method")
            if self.status not in ("passed", "inconclusive"):
                raise ValueError("no-update arm must have passed or inconclusive status")
        else:
            if self.method not in ("gradient", "eggroll"):
                raise ValueError("updated arm must specify gradient or eggroll")
            if self.status in ("direction_mismatch",) and not self.recalibration_eligible:
                raise ValueError("direction_mismatch arm cannot be recalibration_eligible")
        if self.status == "inconclusive" and not self.failed_conditions:
            raise ValueError("inconclusive arm must specify failed conditions")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "arm": self.arm,
            "method": self.method,
            "status": self.status,
            "baseline": self.baseline_checkpoint.to_dict() if self.baseline_checkpoint else None,
            "checkpoint_count": len(self.checkpoints),
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
            "recalibration_eligible": self.recalibration_eligible,
        }
        if self.failed_conditions:
            result["failed_conditions"] = list(self.failed_conditions)
        _require_finite(result, "arm result")
        return result


@dataclass(frozen=True)
class TrainabilityReport:
    """Complete bounded trainability investigation result."""

    schema_version: int
    configuration: TrainabilityConfiguration
    asset_identity_digest: str
    initial_state_digest: str
    overall_status: Literal[
        "bounded_trainability_observed",
        "shared_loss_behavior_conflict",
        "objective_untrainable",
        "method_specific_failure",
        "inconclusive",
    ]
    overfit_probe: OverfitProbeResult
    causal_probes: tuple[CausalProbeResult, ...]
    arms: tuple[ArmResult, ...]
    elapsed_seconds: float
    eta_seconds: float | None = None
    failed_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != TRAINABILITY_SCHEMA_VERSION:
            raise ValueError(f"trainability report requires schema version {TRAINABILITY_SCHEMA_VERSION}")
        if not _is_sha256(self.asset_identity_digest):
            raise ValueError("trainability report requires SHA-256 asset identity")
        if not _is_sha256(self.initial_state_digest):
            raise ValueError("trainability report requires SHA-256 initial state")
        if not math.isfinite(self.elapsed_seconds):
            raise ValueError("elapsed_seconds must be finite")
        if self.eta_seconds is not None and not math.isfinite(self.eta_seconds):
            raise ValueError("eta_seconds must be finite or None")
        _require_finite(self.to_dict(), "trainability report")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "configuration": self.configuration.to_dict(),
            "asset_identity_digest": self.asset_identity_digest,
            "initial_state_digest": self.initial_state_digest,
            "overall_status": self.overall_status,
            "overfit_probe": self.overfit_probe.to_dict(),
            "causal_probe_count": len(self.causal_probes),
            "causal_probes": [cp.to_dict() for cp in self.causal_probes],
            "arm_count": len(self.arms),
            "arms": [arm.to_dict() for arm in self.arms],
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
        }
        if self.failed_conditions:
            result["failed_conditions"] = list(self.failed_conditions)
        _require_finite(result, "trainability report")
        return result

    def canonical_json(self) -> str:
        """Return the canonical JSON representation."""
        return canonical_json_bytes(self.to_dict()).decode("utf-8")


@dataclass(frozen=True)
class TrainabilityProgressRecord:
    """Strict JSONL record emitted after each investigation event."""

    schema_version: int
    sequence_number: int
    kind: Literal[
        "overfit_attempt_start",
        "overfit_checkpoint",
        "overfit_attempt_complete",
        "overfit_stop",
        "causal_probe_start",
        "causal_probe_complete",
        "arm_start",
        "arm_checkpoint",
        "arm_complete",
        "arm_stop",
        "classification",
    ]
    probe_or_arm: str
    consumed_examples: int = 0
    optimizer_call_count: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != TRAINABILITY_SCHEMA_VERSION:
            raise ValueError(f"trainability progress requires schema version {TRAINABILITY_SCHEMA_VERSION}")
        if self.sequence_number < 0:
            raise ValueError("progress sequence_number must be non-negative")
        if not math.isfinite(self.consumed_examples) or self.consumed_examples < 0:
            raise ValueError("consumed_examples must be finite and non-negative")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if self.eta_seconds is not None and (not math.isfinite(self.eta_seconds) or self.eta_seconds < 0):
            raise ValueError("eta_seconds must be finite and non-negative or None")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "sequence_number": self.sequence_number,
            "kind": self.kind,
            "probe_or_arm": self.probe_or_arm,
            "consumed_examples": self.consumed_examples,
            "optimizer_call_count": self.optimizer_call_count,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
        }
        if self.payload:
            result["payload"] = self.payload
        _require_finite(result, "trainability progress record")
        return result

    def canonical_json_line(self) -> str:
        """Return the canonical JSON line for JSONL output."""
        return canonical_json_bytes(self.to_dict()).decode("utf-8")


class TrainabilityReportValidationError(ValueError):
    """One or more strict parsing or compatibility failures."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("incompatible trainability report: " + "; ".join(self.issues))


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate fields while parsing an evidence report."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def canonical_trainability_implementation_identity(
    repository_root: str | Path,
    source_paths: Sequence[str] = TRAINABILITY_IMPLEMENTATION_PATHS,
) -> ImplementationIdentity:
    """Hash ordered repository-relative source files for trainability investigation."""
    root = Path(repository_root).resolve()
    sources: list[tuple[str, str]] = []
    for relative in source_paths:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"guarded trainability source is invalid: {relative}")
        sources.append((relative.replace("\\", "/"), hashlib.sha256(path.read_bytes()).hexdigest()))
    combined = hashlib.sha256(canonical_json_bytes(dict(sources))).hexdigest()
    return ImplementationIdentity(combined, tuple(sources))


def load_and_validate_stability_report(
    stability_report_path: str | Path,
    current_implementation: ImplementationIdentity,
    held_out_record_ids_from_report: tuple[tuple[str, str], ...],
    *,
    require_complete: bool = False,
) -> tuple[str, Mapping[str, Any]]:
    """Load and validate a failed stability report for trainability investigation.

    Args:
        stability_report_path: Path to the JSON stability report
        current_implementation: Current implementation identity for compatibility check
        held_out_record_ids_from_report: Expected held-out record identifiers
        require_complete: Require the canonical production stability schema

    Returns:
        Tuple of (report_digest, report_configuration)

    Raises:
        TrainabilityReportValidationError: If report is missing, passing, malformed, or incompatible
    """
    report_path = Path(stability_report_path).resolve()
    issues: list[str] = []

    if not report_path.exists():
        issues.append(f"stability report not found: {report_path}")
        raise TrainabilityReportValidationError(issues)

    try:
        report_bytes = report_path.read_bytes()
        report_data = json.loads(
            report_bytes,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON number {value}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, OSError, ValueError) as e:
        issues.append(f"stability report malformed or unreadable: {e}")
        raise TrainabilityReportValidationError(issues) from e

    if not isinstance(report_data, dict):
        issues.append("stability report root must be an object")
        raise TrainabilityReportValidationError(issues)

    status = report_data.get("status")
    if status == "passed":
        issues.append("stability report must have status=failed (passing reports cannot start investigation)")
    elif status != "failed":
        issues.append(f"stability report must have status=failed, got {status!r}")
    if require_complete and report_data.get("schema_version") != 1:
        issues.append("stability report requires schema_version=1")

    if require_complete:
        try:
            canonical_report = read_stability_report(report_path).report
        except CanonicalStabilityReportValidationError as error:
            issues.extend(f"canonical stability report: {issue}" for issue in error.issues)
        else:
            if canonical_report.status != "failed":
                issues.append("canonical stability report must have status=failed")
            if canonical_report.outcome_code == 0:
                issues.append("canonical stability report failed runs require a nonzero outcome_code")
            if not canonical_report.checkpoints:
                issues.append("canonical stability report requires completed checkpoints")
            elif canonical_report.checkpoints[-1].failed_thresholds != canonical_report.failed_thresholds:
                issues.append("canonical stability report final checkpoint failures must equal report failures")
            if not canonical_report.failed_thresholds:
                issues.append("canonical stability report failed runs require failed_thresholds")

    if issues:
        raise TrainabilityReportValidationError(issues)

    config = report_data.get("configuration", {})
    if not isinstance(config, Mapping):
        issues.append("configuration must be an object")
        config = {}
    else:
        try:
            _require_finite(config, "stability report")
        except ValueError as error:
            issues.append(str(error))

        if require_complete and not held_out_record_ids_from_report:
            issues.append("strict stability validation requires the exact held-out selection identity")

        required_configuration_keys = {
            "implementation",
            "asset_identity",
            "eggroll",
            "initialization_seed",
            "held_out_problem_count",
            "development_example_count",
            "checkpoints",
        }
        if require_complete:
            for key in sorted(required_configuration_keys.difference(config)):
                issues.append(f"stability configuration missing required field: {key}")

        impl_config = config.get("implementation", {})
        if not isinstance(impl_config, Mapping):
            issues.append("implementation must be an object in configuration")
        else:
            if require_complete:
                for key in ("sha256", "sources"):
                    if key not in impl_config:
                        issues.append(f"implementation missing required field: {key}")
            impl_sha = impl_config.get("sha256")
            if impl_sha != current_implementation.sha256:
                issues.append(
                    f"implementation mismatch: report has {impl_sha!r}, current is {current_implementation.sha256!r}"
                )
            report_sources = impl_config.get("sources")
            current_sources = current_implementation.to_dict()["sources"]
            if report_sources is not None and report_sources != current_sources:
                issues.append("implementation sources mismatch")

        asset_config = config.get("asset_identity", {})
        if not isinstance(asset_config, Mapping):
            issues.append("asset_identity must be an object in configuration")
        else:
            report_held_out_count = config.get(
                "held_out_problem_count",
                asset_config.get("held_out_problem_count", asset_config.get("held_out_record_count")),
            )
            expected_count = (
                len(held_out_record_ids_from_report) if require_complete else len(held_out_record_ids_from_report) or 64
            )
            if report_held_out_count != expected_count:
                issues.append(
                    f"held-out record count mismatch: report says {report_held_out_count}, current expects {expected_count}"
                )

            report_item_ids = asset_config.get("held_out_item_ids")
            expected_item_ids = [record_id for record_id, _record_hash in held_out_record_ids_from_report]
            if require_complete and not isinstance(report_item_ids, list):
                issues.append("asset_identity.held_out_item_ids must be a list")
            if report_item_ids is not None and expected_item_ids and report_item_ids != expected_item_ids:
                issues.append("held-out record identifiers mismatch")
            report_record_identifiers = asset_config.get("held_out_record_identifiers")
            if report_record_identifiers is not None and expected_item_ids:
                expected_record_identifiers = [list(item) for item in held_out_record_ids_from_report]
                if report_record_identifiers != expected_record_identifiers:
                    issues.append("held-out record content identifiers mismatch")

            full_identity_keys = set(STAGE0_IDENTITY).union({"runtime", "held_out_item_ids"})
            if require_complete:
                for key in sorted(full_identity_keys.difference(asset_config)):
                    issues.append(f"asset identity missing required field: {key}")
            if full_identity_keys.intersection(asset_config):
                expected_identity = {
                    **dict(STAGE0_IDENTITY),
                    "runtime": runtime_identity(),
                    "held_out_item_ids": expected_item_ids if expected_item_ids else report_item_ids,
                }
                for key in full_identity_keys:
                    if key in asset_config and asset_config[key] != expected_identity.get(key):
                        issues.append(f"asset identity mismatch: {key}")

        eggroll_config = config.get("eggroll")
        expected_eggroll = {
            "optimizer_contract": "torch.optim.SGD(momentum=0.0)",
            "parameter_scope": list(EGGROLL_PARAMETER_PATHS),
            "population": DEFAULT_POP_SIZE,
            "sigma": DEFAULT_SIGMA,
            "rank": DEFAULT_RANK,
            "fitness_batch_size": DEFAULT_FITNESS_BATCH_SIZE,
            "evaluation_batch_size": DEFAULT_EVAL_BATCH_SIZE,
            "variance_weight": DEFAULT_VARIANCE_WEIGHT,
            "prompt_alignment_weight": DEFAULT_PROMPT_ALIGNMENT_WEIGHT,
            "learning_rate": DEFAULT_LR,
        }
        if not isinstance(eggroll_config, Mapping):
            if require_complete:
                issues.append("eggroll must be an object in configuration")
        else:
            for key, expected in expected_eggroll.items():
                if (require_complete or key in eggroll_config) and eggroll_config.get(key) != expected:
                    issues.append(f"EGGROLL configuration mismatch: {key}")

        expected_scalar_fields = {
            "initialization_seed": 0,
            "held_out_problem_count": 64,
            "development_example_count": DEVELOPMENT_EXAMPLE_COUNT,
            "checkpoints": list(DEVELOPMENT_CHECKPOINTS),
        }
        for key, expected in expected_scalar_fields.items():
            if (require_complete or key in config) and config.get(key) != expected:
                issues.append(f"stability configuration mismatch: {key}")

    if issues:
        raise TrainabilityReportValidationError(issues)

    report_digest = hashlib.sha256(report_bytes).hexdigest()

    return report_digest, config


def compute_initial_state_digest(state_parameters: Sequence[tuple[str, Any]]) -> str:
    """Compute SHA-256 digest of fresh trainability initial state.

    Args:
        state_parameters: Ordered tuples of (parameter_name, tensor_data)

    Returns:
        SHA-256 digest as hex string
    """
    state_dict: dict[str, Any] = {}
    for name, value in state_parameters:
        if isinstance(value, torch.Tensor):
            state_dict[name] = {
                "_tensor": True,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "device": str(value.device),
                "data_hash": hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest(),
            }
        else:
            state_dict[name] = value
    combined = canonical_json_bytes(state_dict)
    return hashlib.sha256(combined).hexdigest()


def build_trainability_record_selections(
    dataset: Any,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
]:
    """Build the fixed training and held-out record selections for trainability investigation.

    Args:
        dataset: Stage0Dataset with training and held-out records

    Returns:
        Tuple of (overfit_records, training_32_records, held_out_64_records)
        where each is a tuple of (record_id, record_hash) pairs
    """
    training_records = dataset.training_records(mode="eggroll", epoch=1)
    held_out_records = dataset.held_out_records()

    if len(training_records) < 32:
        raise ValueError(
            f"trainability investigation requires at least 32 training records, got {len(training_records)}"
        )
    if len(held_out_records) < 64:
        raise ValueError(
            f"trainability investigation requires at least 64 held-out records, got {len(held_out_records)}"
        )

    overfit_records = training_records[:1]
    training_32_records = training_records[:32]
    held_out_64_records = held_out_records[:64]

    def record_to_identifier(record: Any) -> tuple[str, str]:
        """Convert a record to (id, content_hash) identifier."""
        content = canonical_json_bytes(
            {
                "id": record.id,
                "split": record.split,
                "question": record.question,
                "target": record.target,
            }
        )
        content_hash = hashlib.sha256(content).hexdigest()
        return (record.id, content_hash)

    overfit_ids = tuple(record_to_identifier(r) for r in overfit_records)
    training_32_ids = tuple(record_to_identifier(r) for r in training_32_records)
    held_out_64_ids = tuple(record_to_identifier(r) for r in held_out_64_records)

    return overfit_ids, training_32_ids, held_out_64_ids


def evaluate_single_record_loss(
    trainer: Any,
    question: str,
    answer: str,
    device: str = "cpu",
) -> tuple[float, float]:
    """Evaluate language model loss on a single record.

    Args:
        trainer: LatentCoreTrainer instance
        question: Training question
        answer: Target answer
        device: Device to use for computation

    Returns:
        Tuple of (language_model_loss, total_objective)
    """
    trainer.optimizer.zero_grad()
    latent_loop = trainer.latent_loop
    tokenizer = trainer.tokenizer
    encoder = trainer.encoder
    workspace = trainer.workspace
    language_model = latent_loop.model

    prepared = prepare_training_example(tokenizer, question, answer)
    teacher_state = prompt_teacher_state(
        language_model,
        prepared.context_input_ids.to(device),
    )
    context_embeds = latent_loop.embed_tokens(prepared.context_input_ids)

    slots = encoder.encode(question)
    workspace.write_slots(slots)
    for _ in range(SUBJECT_LATENT_RUNS_PER_ANSWER):
        latent_loop.run(workspace, context_embeds=context_embeds)
    loop_slots = workspace.read_slots()

    answer_objective = prompt_aligned_answer_objective(
        language_model,
        loop_slots,
        prepared.answer_ids.to(device),
        getattr(tokenizer, "eos_token_id", None),
        teacher_state=teacher_state,
    )
    lm_loss = answer_objective.language_model_loss.mean()
    prompt_alignment_penalty = trainer.prompt_alignment_weight * answer_objective.prompt_alignment_loss.mean()
    collapse_penalty = trainer.variance_weight * slot_variance_penalty(loop_slots).mean()
    total_objective = lm_loss + prompt_alignment_penalty + collapse_penalty

    return lm_loss.item(), total_objective.item()


def _trainer_state_is_finite(trainer: Any) -> bool:
    """Return whether parameters, gradients, and optimizer tensors are finite."""
    state = getattr(trainer, "state", None)
    parameters = getattr(state, "trainable_params", ())
    for parameter in parameters.values() if isinstance(parameters, Mapping) else parameters:
        if not torch.isfinite(parameter.detach()).all():
            return False
        if parameter.grad is not None and not torch.isfinite(parameter.grad.detach()).all():
            return False
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        return True

    def finite_value(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value.detach()).all())
        if isinstance(value, Mapping):
            return all(finite_value(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_value(item) for item in value)
        return not isinstance(value, float) or math.isfinite(value)

    return finite_value(optimizer.state_dict().get("state", {}))


def run_overfit_attempt(
    question: str,
    answer: str,
    learning_rate: float,
    device: str = "cpu",
    checkpoint_steps: tuple[int, ...] = (0, 1, 4, 16, 64),
    *,
    trainer_factory: Callable[[float, str], Any] | None = None,
    metrics_evaluator: Callable[[Any], StabilityMetrics] | None = None,
) -> dict[str, Any]:
    """Run one overfit memorization attempt at a fixed learning rate.

    Args:
        question: Training question
        answer: Target answer
        learning_rate: Learning rate for optimizer
        device: Device to use for computation
        checkpoint_steps: Optimizer call counts where to evaluate

    Returns:
        Dictionary with attempt status, baseline metrics, and checkpoints
    """
    try:
        configure_deterministic_runtime()
        trainer = (
            trainer_factory(learning_rate, device)
            if trainer_factory is not None
            else LatentCoreTrainer(lr=learning_rate, device=device)
        )

        if not _trainer_state_is_finite(trainer):
            return {
                "learning_rate": learning_rate,
                "status": "non_finite",
                "baseline_loss": None,
                "checkpoints": [],
                "failed_reason": "Initial trainer state contained a non-finite value",
            }

        baseline_loss, _ = evaluate_single_record_loss(
            trainer,
            question,
            answer,
            device=device,
        )

        if not math.isfinite(baseline_loss):
            return {
                "learning_rate": learning_rate,
                "status": "non_finite",
                "baseline_loss": None,
                "checkpoints": [],
                "failed_reason": "Initial evaluation produced non-finite loss",
            }

        baseline_metrics: StabilityMetrics | None = None
        if metrics_evaluator is not None:
            try:
                baseline_metrics = metrics_evaluator(trainer)
                _require_finite(baseline_metrics.to_dict(), "overfit baseline metrics")
            except Exception as error:  # noqa: BLE001 - preserve diagnostic evidence
                return {
                    "learning_rate": learning_rate,
                    "status": "inconclusive",
                    "baseline_loss": baseline_loss,
                    "checkpoints": [],
                    "failed_reason": f"Initial metric evaluation failed: {error!s}",
                }

        checkpoints: list[tuple[int, float, float]] = []
        metric_checkpoints: list[tuple[int, StabilityMetrics]] = []
        current_step = 0

        for target_step in checkpoint_steps:
            if current_step >= target_step:
                continue

            steps_to_run = target_step - current_step
            for _ in range(steps_to_run):
                try:
                    result = trainer.train_step(question, answer)
                    if not math.isfinite(result.language_model_loss) or not math.isfinite(result.total_objective):
                        return {
                            "learning_rate": learning_rate,
                            "status": "non_finite",
                            "baseline_loss": baseline_loss,
                            "checkpoints": checkpoints,
                            "failed_reason": f"Non-finite loss at step {current_step + 1}",
                        }
                    if not _trainer_state_is_finite(trainer):
                        return {
                            "learning_rate": learning_rate,
                            "status": "non_finite",
                            "baseline_loss": baseline_loss,
                            "baseline_metrics": baseline_metrics,
                            "checkpoints": checkpoints,
                            "metric_checkpoints": metric_checkpoints,
                            "failed_reason": f"Non-finite trainer state at step {current_step + 1}",
                        }
                    current_step += 1
                except Exception as e:  # noqa: BLE001 - retain failed attempt evidence
                    return {
                        "learning_rate": learning_rate,
                        "status": "inconclusive",
                        "baseline_loss": baseline_loss,
                        "checkpoints": checkpoints,
                        "failed_reason": f"Exception at step {current_step + 1}: {e!s}",
                    }

            checkpoint_loss, _ = evaluate_single_record_loss(
                trainer,
                question,
                answer,
                device=device,
            )

            if not math.isfinite(checkpoint_loss):
                return {
                    "learning_rate": learning_rate,
                    "status": "non_finite",
                    "baseline_loss": baseline_loss,
                    "checkpoints": checkpoints,
                    "failed_reason": f"Non-finite evaluation at step {target_step}",
                }

            loss_ratio = checkpoint_loss / baseline_loss if baseline_loss > 0 else float("inf")
            checkpoints.append((target_step, checkpoint_loss, loss_ratio))

            checkpoint_metrics: StabilityMetrics | None = None
            if metrics_evaluator is not None:
                try:
                    checkpoint_metrics = metrics_evaluator(trainer)
                    _require_finite(checkpoint_metrics.to_dict(), "overfit checkpoint metrics")
                    metric_checkpoints.append((target_step, checkpoint_metrics))
                except Exception as error:  # noqa: BLE001 - preserve diagnostic evidence
                    return {
                        "learning_rate": learning_rate,
                        "status": "inconclusive",
                        "baseline_loss": baseline_loss,
                        "baseline_metrics": baseline_metrics,
                        "checkpoints": checkpoints,
                        "metric_checkpoints": metric_checkpoints,
                        "failed_reason": f"Metric evaluation failed at step {target_step}: {error!s}",
                    }

            max_loss_threshold = 0.5 * baseline_loss
            decoded_pass = checkpoint_metrics is not None and (
                checkpoint_metrics.exact_accuracy == 1.0 and checkpoint_metrics.first_token_accuracy == 1.0
            )
            early_pass = checkpoint_loss <= max_loss_threshold and decoded_pass

            if early_pass:
                return {
                    "learning_rate": learning_rate,
                    "status": "passed",
                    "baseline_loss": baseline_loss,
                    "baseline_metrics": baseline_metrics,
                    "checkpoints": checkpoints,
                    "metric_checkpoints": metric_checkpoints,
                    "checkpoint_steps": checkpoint_steps,
                    "passed_at_step": target_step,
                    "passed_loss_ratio": loss_ratio,
                }

    except Exception as e:  # noqa: BLE001 - retain fatal attempt evidence
        return {
            "learning_rate": learning_rate,
            "status": "inconclusive",
            "baseline_loss": None,
            "checkpoints": [],
            "failed_reason": f"Fatal exception: {e!s}",
        }
    else:
        return {
            "learning_rate": learning_rate,
            "status": "failed",
            "baseline_loss": baseline_loss,
            "baseline_metrics": baseline_metrics,
            "checkpoints": checkpoints,
            "metric_checkpoints": metric_checkpoints,
            "checkpoint_steps": checkpoint_steps,
            "failed_reason": "Did not meet loss reduction criteria within 64 steps",
        }


def classify_overfit_probe(
    attempts: Sequence[OverfitAttempt],
) -> OverfitProbeResult:
    """Classify the overfit probe result based on completed attempts.

    Args:
        attempts: Sequence of completed OverfitAttempt results

    Returns:
        OverfitProbeResult with status and failed conditions

    Raises:
        ValueError: If attempts is empty
    """
    if not attempts:
        raise ValueError("overfit probe classification requires at least one attempt")

    for attempt in attempts:
        if attempt.status != "passed":
            continue

        if not attempt.baseline_metrics or not attempt.checkpoints:
            continue

        for _checkpoint_count, checkpoint_metrics in attempt.checkpoints:
            if checkpoint_metrics is None:
                continue

            if not math.isfinite(checkpoint_metrics.language_model_loss):
                continue
            if not math.isfinite(checkpoint_metrics.exact_accuracy):
                continue
            if not math.isfinite(checkpoint_metrics.first_token_accuracy):
                continue

            baseline_lm_loss = attempt.baseline_metrics.language_model_loss
            if not math.isfinite(baseline_lm_loss) or baseline_lm_loss <= 0:
                continue

            loss_threshold = MAX_LOSS_REDUCTION_RATIO * baseline_lm_loss

            exact_accuracy = checkpoint_metrics.exact_accuracy
            first_token_accuracy = checkpoint_metrics.first_token_accuracy
            current_loss = checkpoint_metrics.language_model_loss

            if exact_accuracy == 1.0 and first_token_accuracy == 1.0 and current_loss <= loss_threshold:
                return OverfitProbeResult(
                    status="passed",
                    attempts=tuple(attempts),
                )

    inconclusive_conditions = tuple(
        f"learning_rate_{attempt.learning_rate}: {attempt.failed_reason or 'inconclusive evidence'}"
        for attempt in attempts
        if attempt.status == "inconclusive"
    )
    if inconclusive_conditions:
        return OverfitProbeResult(
            status="inconclusive",
            attempts=tuple(attempts),
            failed_conditions=inconclusive_conditions,
        )

    return OverfitProbeResult(
        status="objective_untrainable",
        attempts=tuple(attempts),
    )


@dataclass(frozen=True)
class CausalProbeEvaluation:
    """Complete objective evaluation over 8 training records for causal analysis."""

    training_records: tuple[tuple[str, str], ...]
    pre_update_metrics: StabilityMetrics
    post_update_metrics: StabilityMetrics | None = None

    def __post_init__(self) -> None:
        if not self.training_records:
            raise ValueError("causal probe evaluation requires at least one record")
        if len(self.training_records) != 8:
            raise ValueError(f"causal probe evaluation requires exactly 8 records, got {len(self.training_records)}")
        if self.pre_update_metrics is None:
            raise ValueError("pre-update metrics must be provided")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "training_record_count": len(self.training_records),
            "pre_update_metrics": self.pre_update_metrics.to_dict(),
            "post_update_metrics": (self.post_update_metrics.to_dict() if self.post_update_metrics else None),
        }
        _require_finite(result, "causal probe evaluation")
        return result


def evaluate_objective_on_training_records(
    trainer: Any,
    training_records: Sequence[tuple[str, str]],
    max_records: int = 8,
    *,
    device: str = "cpu",
) -> tuple[float, tuple[tuple[str, str], ...]]:
    """Evaluate the complete objective on a set of training records.

    Args:
        trainer: LatentCoreTrainer instance
        training_records: Sequence of (question, answer) tuples
        max_records: Maximum number of records to evaluate (default 8)
        device: Device to use for computation

    Returns:
        Tuple of (mean_total_objective, record_ids) for the evaluated support records

    Raises:
        ValueError: If training records list is empty or too short
    """
    if not training_records:
        raise ValueError("training records are required for causal evaluation")

    if len(training_records) < max_records:
        raise ValueError(f"causal evaluation requires at least {max_records} records, got {len(training_records)}")

    selected_records = training_records[:max_records]

    def record_to_identifier(record: tuple[str, str]) -> tuple[str, str]:
        """Convert a record to (question, answer_hash) identifier."""
        question, answer = record
        content = canonical_json_bytes(
            {
                "question": question,
                "answer": answer,
            }
        )
        content_hash = hashlib.sha256(content).hexdigest()
        return (question, content_hash)

    record_ids = tuple(record_to_identifier(record) for record in selected_records)

    objective_values: list[float] = []
    for question, answer in selected_records:
        lm_loss, total_objective = evaluate_single_record_loss(trainer, question, answer, device=device)
        if math.isfinite(lm_loss) and math.isfinite(total_objective):
            objective_values.append(total_objective)

    if len(objective_values) != len(selected_records):
        raise ValueError("causal evaluation produced a non-finite value for one or more support records")

    return sum(objective_values) / len(objective_values), record_ids


def evaluate_held_out_metrics(
    trainer: Any,
    held_out_records: Sequence[tuple[str, str] | AsdivRecord],
    *,
    device: str = "cpu",
    max_decode_tokens: int,
) -> StabilityMetrics:
    """Evaluate exactly the fixed 64-record held-out cohort with production metrics."""
    if len(held_out_records) != 64:
        raise ValueError(f"held-out evaluation requires exactly 64 records, got {len(held_out_records)}")

    problems: list[tuple[str, str]] = []
    for record in held_out_records:
        if isinstance(record, AsdivRecord):
            problems.append((record.question, record.target))
        else:
            question, answer = record
            problems.append((question, answer))
    metrics = evaluate_stability_metrics(
        trainer,
        problems,
        device=device,
        max_decode_tokens=max_decode_tokens,
    )
    if metrics.problem_count != 64:
        raise ValueError(f"held-out evaluation returned {metrics.problem_count} records instead of 64")
    _require_finite(metrics.to_dict(), "held-out evaluation metrics")
    return metrics


@dataclass(frozen=True)
class UpdateDirection:
    """Captured direction of a proposed update for causal analysis."""

    method: MethodName
    baseline_objective: float
    predicted_delta: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.baseline_objective):
            raise ValueError("baseline objective must be finite")
        if self.baseline_objective <= 0:
            raise ValueError("baseline objective must be positive")
        if not math.isfinite(self.predicted_delta):
            raise ValueError("predicted delta must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "baseline_objective": self.baseline_objective,
            "predicted_delta": self.predicted_delta,
        }


@dataclass(frozen=True)
class UpdatePrediction:
    """First-order prediction of objective change from an update."""

    method: MethodName
    pre_update_objective: float
    predicted_post_update_objective: float
    predicted_delta: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.pre_update_objective):
            raise ValueError("pre-update objective must be finite")
        if not math.isfinite(self.predicted_post_update_objective):
            raise ValueError("predicted post-update objective must be finite")
        if not math.isfinite(self.predicted_delta):
            raise ValueError("predicted delta must be finite")

        expected_predicted = self.pre_update_objective + self.predicted_delta
        delta = abs(expected_predicted - self.predicted_post_update_objective)
        if delta > 1e-6:
            raise ValueError(
                f"predicted objectives are inconsistent: "
                f"pre + delta = {expected_predicted}, "
                f"predicted post = {self.predicted_post_update_objective}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "pre_update_objective": self.pre_update_objective,
            "predicted_post_update_objective": self.predicted_post_update_objective,
            "predicted_delta": self.predicted_delta,
        }


def compute_first_order_prediction(
    baseline_objective: float,
    gradient_norm: float,
    step_size: float,
) -> float:
    """Compute first-order predicted objective change from gradient norm and step size.

    Args:
        baseline_objective: Current objective value
        gradient_norm: Norm of the gradient at current point
        step_size: Update step size (learning rate or similar)

    Returns:
        Predicted change in objective (negative indicates improvement prediction)

    Raises:
        ValueError: If parameters are invalid
    """
    if not math.isfinite(baseline_objective) or baseline_objective <= 0:
        raise ValueError("baseline objective must be positive and finite")
    if not math.isfinite(gradient_norm) or gradient_norm < 0:
        raise ValueError("gradient norm must be non-negative and finite")
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step size must be positive and finite")

    predicted_delta = -step_size * gradient_norm

    return predicted_delta


def classify_causal_result(
    predicted_delta: float,
    observed_delta: float,
    baseline_objective: float,
    post_update_objective: float,
) -> tuple[str, list[str]]:
    """Classify causal probe result based on predicted vs observed changes.

    Args:
        predicted_delta: First-order predicted objective change
        observed_delta: Actual observed objective change (post - pre)
        baseline_objective: Pre-update objective value
        post_update_objective: Post-update objective value

    Returns:
        Tuple of (status, failed_conditions) where status is one of:
        - "passed": both predicted and observed negative (improvement)
        - "direction_mismatch": predicted negative but observed non-negative
        - "non_finite": any value is non-finite

    Raises:
        ValueError: If objective values are invalid
    """
    if not math.isfinite(baseline_objective) or baseline_objective <= 0:
        raise ValueError("baseline objective must be positive and finite")
    if not math.isfinite(post_update_objective):
        raise ValueError("post-update objective must be finite")

    failed_conditions = []

    if not math.isfinite(predicted_delta):
        failed_conditions.append("predicted_delta_non_finite")
        return "non_finite", failed_conditions

    if not math.isfinite(observed_delta):
        failed_conditions.append("observed_delta_non_finite")
        return "non_finite", failed_conditions

    predicted_improves = predicted_delta < 0
    observed_improves = observed_delta < 0

    if not predicted_improves:
        failed_conditions.append("predicted_not_improvement")
        return "non_finite", failed_conditions

    if not observed_improves:
        failed_conditions.append("observed_not_improvement")
        return "direction_mismatch", failed_conditions

    return "passed", []


def validate_causal_safety_checks(
    baseline_separation: float,
    post_update_separation: float,
    max_update_relative_matrix_rms: float,
    min_separation_ratio: float = MIN_SEPARATION_RATIO,
    max_rms_change: float = MAX_UPDATE_RELATIVE_MATRIX_RMS,
) -> tuple[str, list[str]]:
    """Validate safety checks for causal probe results.

    Args:
        baseline_separation: Pre-update held-out separation metric
        post_update_separation: Post-update held-out separation metric
        max_update_relative_matrix_rms: Maximum relative RMS change in parameters
        min_separation_ratio: Minimum allowed separation ratio (default 0.95)
        max_rms_change: Maximum allowed RMS change (default 0.01)

    Returns:
        Tuple of (status, failed_conditions) where status is one of:
        - "passed": all safety checks pass
        - "unsafe_update": separation or RMS violation detected
        - "non_finite": any value is non-finite

    Raises:
        ValueError: If baseline separation is zero or negative
    """
    failed_conditions = []

    try:
        baseline_sep_float = float(baseline_separation)
    except (TypeError, ValueError):
        raise ValueError("baseline separation must be non-negative and finite") from None

    if baseline_sep_float <= 0:
        raise ValueError("baseline separation must be positive for ratio check")

    if not math.isfinite(baseline_separation):
        raise ValueError("baseline separation must be non-negative and finite")

    if not math.isfinite(post_update_separation):
        failed_conditions.append("non_finite_separation")
        return "non_finite", failed_conditions

    if not math.isfinite(max_update_relative_matrix_rms):
        failed_conditions.append("non_finite_rms")
        return "non_finite", failed_conditions

    separation_ratio = post_update_separation / baseline_separation
    if separation_ratio < min_separation_ratio:
        failed_conditions.append(f"separation_retention_below_floor: {separation_ratio:.4f} < {min_separation_ratio}")

    if max_update_relative_matrix_rms > max_rms_change:
        failed_conditions.append(
            f"relative_matrix_rms_above_ceiling: {max_update_relative_matrix_rms:.4f} > {max_rms_change}"
        )

    if failed_conditions:
        return "unsafe_update", failed_conditions

    return "passed", []


def compute_recalibration_eligibility(
    method: MethodName,
    overfit_status: str,
    method_status: str,
    causal_status: str | None = None,
) -> tuple[bool, list[str]]:
    """Compute method-specific recalibration eligibility.

    Args:
        method: Method name (gradient or eggroll)
        overfit_status: Status from overfit probe (passed/objective_untrainable)
        method_status: Status from arm evaluation (viable/no_improvement/etc)
        causal_status: Status from causal probe (passed/direction_mismatch/unsafe_update)

    Returns:
        Tuple of (is_eligible, missing_prerequisites) where is_eligible is True
        only if all prerequisites are met and method_status allows recalibration

    Missing prerequisites list includes:
    - "objective_untrainable" if overfit probe failed
    - "causal_unsupported" if causal probe failed
    - "method_not_viable" if method status prevents recalibration
    """
    missing = []

    if overfit_status != "passed":
        missing.append("objective_untrainable")

    if causal_status != "passed":
        missing.append("causal_status_missing" if causal_status is None else "causal_unsupported")

    if method_status != "viable":
        missing.append("method_not_viable")

    is_eligible = len(missing) == 0

    return is_eligible, missing


def classify_overall_trainability(
    overfit_status: str,
    method_statuses: dict[MethodName, str],
    method_evidence: Mapping[MethodName, Any] | None = None,
) -> tuple[
    Literal[
        "bounded_trainability_observed",
        "shared_loss_behavior_conflict",
        "objective_untrainable",
        "method_specific_failure",
        "inconclusive",
    ],
    list[str],
]:
    """Classify overall trainability from overfit probe and method results.

    Args:
        overfit_status: Status from overfit probe (passed/objective_untrainable)
        method_statuses: Dict of method (gradient/eggroll) to status
        method_evidence: Optional ArmResult mapping used for shared-conflict proof

    Returns:
        Tuple of (overall_status, failed_conditions) where status is one of:
        - "objective_untrainable": overfit probe failed
        - "bounded_trainability_observed": at least one method viable
        - "shared_loss_behavior_conflict": both methods show same failure pattern
        - "method_specific_failure": one method viable, one not
        - "inconclusive": insufficient evidence

    Raises:
        ValueError: If inputs are invalid
    """
    failed_conditions = []

    if overfit_status == "inconclusive":
        failed_conditions.append("overfit_inconclusive")
        return "inconclusive", failed_conditions

    if overfit_status == "objective_untrainable":
        failed_conditions.append("objective_untrainable")
        return "objective_untrainable", failed_conditions

    if not method_statuses:
        failed_conditions.append("no_method_status")
        return "inconclusive", failed_conditions

    if any(status == "inconclusive" for status in method_statuses.values()):
        failed_conditions.append("incomplete_method_evidence")
        return "inconclusive", failed_conditions

    viable_methods = [method for method, status in method_statuses.items() if status == "viable"]

    if len(viable_methods) >= 1:
        return "bounded_trainability_observed", []

    if len(viable_methods) == 0:
        non_viable_statuses = set(method_statuses.values())
        if len(non_viable_statuses) == 1:
            shared_status = next(iter(non_viable_statuses))
            if shared_status in ("no_improvement", "loss_behavior_conflict", "direction_mismatch", "unsafe_update"):
                if method_evidence is not None:
                    shared_conflict_proven = True
                    for method in ("gradient", "eggroll"):
                        arm = method_evidence.get(method)
                        if arm is None:
                            shared_conflict_proven = False
                            break
                        if arm.baseline_checkpoint is None:
                            shared_conflict_proven = False
                            break
                        baseline = arm.baseline_checkpoint.metrics
                        final = (arm.checkpoints[-1] if arm.checkpoints else arm.baseline_checkpoint).metrics
                        lower_loss = (
                            math.isfinite(baseline.language_model_loss)
                            and math.isfinite(final.language_model_loss)
                            and final.language_model_loss < baseline.language_model_loss
                        )
                        decoded_failed = (
                            final.exact_accuracy <= baseline.exact_accuracy
                            or final.first_token_accuracy <= baseline.first_token_accuracy
                        )
                        separation_failed = (
                            baseline.separation_retention <= 0
                            or final.separation_retention / baseline.separation_retention < MIN_SEPARATION_RATIO
                        )
                        if not lower_loss or not (decoded_failed or separation_failed):
                            shared_conflict_proven = False
                            break
                    if not shared_conflict_proven:
                        failed_conditions.append("shared_conflict_evidence_missing")
                        return "inconclusive", failed_conditions
                failed_conditions.append(f"shared_{shared_status}")
                return "shared_loss_behavior_conflict", failed_conditions

        failed_conditions.append("mixed_method_failures")
        return "method_specific_failure", failed_conditions

    return "inconclusive", failed_conditions


def classify_method_status(
    baseline_checkpoint: ArmCheckpoint,
    final_checkpoint: ArmCheckpoint,
    method: MethodName,
    causal_status: str | None = None,
) -> tuple[
    Literal[
        "viable",
        "no_improvement",
        "loss_behavior_conflict",
        "direction_mismatch",
        "unsafe_update",
        "non_finite",
        "inconclusive",
    ],
    list[str],
    bool,
]:
    """Classify method arm status from 32-example evidence.

    Args:
        baseline_checkpoint: Initial checkpoint (no-update arm baseline)
        final_checkpoint: Final checkpoint after updates
        method: Method name (gradient or eggroll)
        causal_status: Optional causal probe status (passed/direction_mismatch/unsafe_update)

    Returns:
        Tuple of (status, failed_conditions, recalibration_eligible) where status is one of:
        - "viable": meets all conditions for bounded trainability
        - "no_improvement": loss increased or no decoded improvement
        - "loss_behavior_conflict": loss improved but metrics didn't
        - "direction_mismatch": contradicts causal prediction
        - "unsafe_update": violated separation/RMS limits
        - "non_finite": any value is non-finite
        - "inconclusive": unclear evidence

    Raises:
        ValueError: If inputs are invalid
    """
    failed_conditions = []

    baseline_metrics = baseline_checkpoint.metrics
    final_metrics = final_checkpoint.metrics

    if not math.isfinite(final_metrics.language_model_loss):
        failed_conditions.append("final_loss_non_finite")
        return "non_finite", failed_conditions, False

    if not math.isfinite(baseline_metrics.language_model_loss):
        failed_conditions.append("baseline_loss_non_finite")
        return "non_finite", failed_conditions, False

    loss_improved = final_metrics.language_model_loss < baseline_metrics.language_model_loss

    if not loss_improved:
        failed_conditions.append("loss_not_improved")
        return "no_improvement", failed_conditions, False

    exact_improved = final_metrics.exact_accuracy > baseline_metrics.exact_accuracy
    first_token_improved = final_metrics.first_token_accuracy > baseline_metrics.first_token_accuracy

    if not exact_improved or not first_token_improved:
        failed_conditions.append("no_decoded_improvement")
        decoded_regressed = (
            final_metrics.exact_accuracy < baseline_metrics.exact_accuracy
            or final_metrics.first_token_accuracy < baseline_metrics.first_token_accuracy
        )
        if decoded_regressed:
            return "loss_behavior_conflict", failed_conditions, False
        else:
            return "no_improvement", failed_conditions, False

    if not math.isfinite(final_metrics.separation_retention):
        failed_conditions.append("final_separation_non_finite")
        return "non_finite", failed_conditions, False

    separation_ratio = (
        final_metrics.separation_retention / baseline_metrics.separation_retention
        if baseline_metrics.separation_retention > 0
        else 0.0
    )
    if separation_ratio < MIN_SEPARATION_RATIO:
        failed_conditions.append(f"separation_retention_ratio_{separation_ratio:.4f}")
        return "unsafe_update", failed_conditions, False

    if final_checkpoint.max_update_relative_matrix_rms is not None:
        max_rms = final_checkpoint.max_update_relative_matrix_rms
        if max_rms > MAX_UPDATE_RELATIVE_MATRIX_RMS:
            failed_conditions.append(f"relative_rms_ceiling_{max_rms:.4f}")
            return "unsafe_update", failed_conditions, False

    if causal_status == "direction_mismatch":
        failed_conditions.append("causal_direction_mismatch")
        return "direction_mismatch", failed_conditions, False
    elif causal_status == "unsafe_update":
        failed_conditions.append("causal_unsafe_update")
        return "unsafe_update", failed_conditions, False
    elif causal_status == "non_finite":
        failed_conditions.append("causal_non_finite")
        return "non_finite", failed_conditions, False

    recalibration_eligible = causal_status == "passed" if causal_status else False

    return "viable", [], recalibration_eligible


@dataclass(frozen=True)
class FreshStateManifest:
    """Canonical checkpoint for independent arm initialization."""

    training_records: tuple[tuple[str, str], ...]
    checkpoint_example_counts: tuple[int, ...] = (0, 8, 32)
    held_out_records: tuple[tuple[str, str], ...] = ()
    training_record_identifiers: tuple[tuple[str, str], ...] = ()
    held_out_record_identifiers: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.training_records:
            raise ValueError("fresh state manifest requires training records")
        if len(self.training_records) < 32:
            raise ValueError(f"fresh state manifest requires at least 32 records, got {len(self.training_records)}")
        if not self.checkpoint_example_counts:
            raise ValueError("fresh state manifest requires checkpoint counts")
        sorted_checkpoints = sorted(self.checkpoint_example_counts)
        if sorted_checkpoints != list(self.checkpoint_example_counts):
            raise ValueError("checkpoint example counts must be sorted")
        if sorted_checkpoints[-1] > 32:
            raise ValueError(f"checkpoint counts must not exceed 32 records, got max {sorted_checkpoints[-1]}")
        if self.held_out_records and len(self.held_out_records) != 64:
            raise ValueError(
                f"fresh state manifest requires exactly 64 held-out records, got {len(self.held_out_records)}"
            )
        if self.training_record_identifiers and len(self.training_record_identifiers) != len(self.training_records):
            raise ValueError("training record identifiers must cover every training record")
        if self.held_out_record_identifiers and len(self.held_out_record_identifiers) != len(self.held_out_records):
            raise ValueError("held-out record identifiers must cover every held-out record")

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_record_count": len(self.training_records),
            "checkpoint_example_counts": list(self.checkpoint_example_counts),
            "held_out_record_count": len(self.held_out_records),
            "training_record_identifiers": [list(item) for item in self.training_record_identifiers],
            "held_out_record_identifiers": [list(item) for item in self.held_out_record_identifiers],
        }


@dataclass(frozen=True)
class ArmEvaluation:
    """Evaluation snapshot at one checkpoint for a method arm."""

    example_count: int
    metrics: StabilityMetrics
    examples_consumed: int = 0
    optimizer_call_count: int = 0
    max_update_relative_matrix_rms: float | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    status: Literal["active", "stopped"] = "active"
    stop_reason: str = ""
    training_record_identifiers: tuple[tuple[str, str], ...] = ()
    held_out_record_identifiers: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.example_count < 0:
            raise ValueError("example count must be non-negative")
        if self.examples_consumed < 0:
            raise ValueError("examples consumed must be non-negative")
        if self.optimizer_call_count < 0:
            raise ValueError("optimizer call count must be non-negative")
        if self.max_update_relative_matrix_rms is not None and (
            not math.isfinite(self.max_update_relative_matrix_rms) or self.max_update_relative_matrix_rms < 0
        ):
            raise ValueError("max update relative matrix RMS must be finite and non-negative")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("arm evaluation elapsed time must be finite and non-negative")
        if self.eta_seconds is not None and (not math.isfinite(self.eta_seconds) or self.eta_seconds < 0):
            raise ValueError("arm evaluation ETA must be finite and non-negative")
        if self.training_record_identifiers and self.example_count > len(self.training_record_identifiers):
            raise ValueError("training record identifiers must cover the evaluation checkpoint")
        if self.held_out_record_identifiers and len(self.held_out_record_identifiers) != 64:
            raise ValueError("held-out record identifiers must contain exactly 64 records")
        if self.status not in ("active", "stopped"):
            raise ValueError(f"invalid status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_count": self.example_count,
            "examples_consumed": self.examples_consumed,
            "optimizer_call_count": self.optimizer_call_count,
            "metrics": self.metrics.to_dict(),
            "max_update_relative_matrix_rms": self.max_update_relative_matrix_rms,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "status": self.status,
            "stop_reason": self.stop_reason if self.stop_reason else None,
            "training_record_identifiers": [list(item) for item in self.training_record_identifiers],
            "held_out_record_identifiers": [list(item) for item in self.held_out_record_identifiers],
        }


@dataclass(frozen=True)
class NoUpdateControl:
    """State snapshot from no-update arm for drift detection."""

    baseline_metrics: StabilityMetrics
    final_metrics: StabilityMetrics
    metrics_drift: dict[str, float]
    state_hash_baseline: str | None = None
    state_hash_final: str | None = None

    def __post_init__(self) -> None:
        if not self.metrics_drift:
            raise ValueError("no-update control requires metrics drift data")

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "final_metrics": self.final_metrics.to_dict(),
            "metrics_drift": dict(self.metrics_drift),
            "state_hash_baseline": self.state_hash_baseline,
            "state_hash_final": self.state_hash_final,
        }


@dataclass(frozen=True)
class ArmSafetyCheck:
    """Independent safety validation for one arm evaluation."""

    status: Literal["passed", "stopped"]
    stop_reason: str = ""
    non_finite_detected: bool = False
    rms_ceiling_violated: bool = False
    separation_floor_violated: bool = False

    def __post_init__(self) -> None:
        if self.status not in ("passed", "stopped"):
            raise ValueError(f"invalid status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stop_reason": self.stop_reason if self.stop_reason else None,
            "non_finite_detected": self.non_finite_detected,
            "rms_ceiling_violated": self.rms_ceiling_violated,
            "separation_floor_violated": self.separation_floor_violated,
        }


def check_arm_evaluation_safety(
    metrics: StabilityMetrics,
    baseline_separation: float = 0.95,
    max_rms_ceiling: float = 0.01,
    min_separation_ratio: float = MIN_SEPARATION_RATIO,
    relative_matrix_rms: float | None = None,
) -> ArmSafetyCheck:
    """Check safety of an arm evaluation snapshot.

    Args:
        metrics: Evaluation metrics to check
        baseline_separation: Baseline separation for ratio check
        max_rms_ceiling: Maximum allowed relative RMS
        min_separation_ratio: Minimum required separation retention ratio
        relative_matrix_rms: Measured update RMS. When omitted, legacy callers use metric RMS values.

    Returns:
        ArmSafetyCheck with independent stop decision
    """
    failed_reasons = []

    if not math.isfinite(metrics.language_model_loss):
        failed_reasons.append("non_finite_lm_loss")
        return ArmSafetyCheck(
            status="stopped",
            stop_reason="Non-finite language model loss",
            non_finite_detected=True,
        )

    if not math.isfinite(metrics.separation_retention):
        failed_reasons.append("non_finite_separation")
        return ArmSafetyCheck(
            status="stopped",
            stop_reason="Non-finite separation retention",
            non_finite_detected=True,
        )

    if baseline_separation > 0:
        separation_ratio = metrics.separation_retention / baseline_separation
        if separation_ratio < min_separation_ratio:
            failed_reasons.append(f"separation_ratio_{separation_ratio:.4f}")
            return ArmSafetyCheck(
                status="stopped",
                stop_reason=(f"Separation retention below floor: {separation_ratio:.4f} < {min_separation_ratio}"),
                separation_floor_violated=True,
            )

    if relative_matrix_rms is None:
        rms_values = [item.rms if isinstance(item, ParameterRms) else 0.0 for item in metrics.parameter_rms]
        measured_rms = max(rms_values) if rms_values else 0.0
        rms_label = "Parameter RMS"
    else:
        measured_rms = relative_matrix_rms
        rms_label = "Relative matrix RMS"
    if not math.isfinite(measured_rms):
        return ArmSafetyCheck(
            status="stopped",
            stop_reason="Non-finite relative matrix RMS",
            non_finite_detected=True,
        )
    if measured_rms > max_rms_ceiling:
        failed_reasons.append(f"rms_ceiling_{measured_rms:.4f}")
        return ArmSafetyCheck(
            status="stopped",
            stop_reason=(f"{rms_label} above ceiling: {measured_rms:.4f} > {max_rms_ceiling}"),
            rms_ceiling_violated=True,
        )

    return ArmSafetyCheck(status="passed")


@dataclass(frozen=True)
class MethodArm:
    """One complete run of a method (no-update, gradient, or EGGROLL) over 32 records."""

    arm_kind: ArmKind
    training_records: tuple[tuple[str, str], ...]
    held_out_records: tuple[tuple[str, str], ...] = ()
    evaluations: tuple[ArmEvaluation, ...] = ()
    final_status: Literal["active", "stopped"] = "active"
    final_stop_reason: str = ""
    training_record_identifiers: tuple[tuple[str, str], ...] = ()
    held_out_record_identifiers: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.training_records:
            raise ValueError("method arm requires training records")
        if len(self.training_records) < 32:
            raise ValueError(f"method arm requires at least 32 records, got {len(self.training_records)}")
        if self.held_out_records and len(self.held_out_records) != 64:
            raise ValueError(f"method arm requires exactly 64 held-out records, got {len(self.held_out_records)}")
        if self.training_record_identifiers and len(self.training_record_identifiers) != len(self.training_records):
            raise ValueError("training record identifiers must cover every training record")
        if self.held_out_record_identifiers and len(self.held_out_record_identifiers) != len(self.held_out_records):
            raise ValueError("held-out record identifiers must cover every held-out record")

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_kind": self.arm_kind,
            "training_record_count": len(self.training_records),
            "held_out_record_count": len(self.held_out_records),
            "held_out_record_pairs": [list(item) for item in self.held_out_records],
            "training_record_identifiers": [list(item) for item in self.training_record_identifiers],
            "held_out_record_identifiers": [list(item) for item in self.held_out_record_identifiers],
            "evaluation_count": len(self.evaluations),
            "final_status": self.final_status,
            "final_stop_reason": self.final_stop_reason if self.final_stop_reason else None,
            "evaluations": [ev.to_dict() for ev in self.evaluations],
        }


def _run_arm_with_updates(
    arm_kind: ArmKind,
    manifest: FreshStateManifest,
    learning_rate: float,
    device: str = "cpu",
    use_eggroll_batches: bool = False,
    progress_observer: Callable[[int, int, dict[str, Any]], None] | None = None,
    max_decode_tokens: int = DEFAULT_MAX_TOKENS,
) -> MethodArm:
    """Run an arm with consistent example counting and optional EGGROLL batching.

    Args:
        arm_kind: Kind of arm (no_update, gradient_only, eggroll_only)
        manifest: Fresh-state manifest with training records and checkpoints
        learning_rate: Learning rate for optimizer (0.0 for no-update)
        device: Device to use for computation
        use_eggroll_batches: Cap updates to EGGROLL batch sizes (8, 32)

    Returns:
        MethodArm with evaluations at each checkpoint
    """
    if len(manifest.training_records) < 32:
        raise ValueError(f"{arm_kind} arm requires at least 32 records, got {len(manifest.training_records)}")
    if not manifest.held_out_records:
        return MethodArm(
            arm_kind=arm_kind,
            training_records=manifest.training_records,
            held_out_records=manifest.held_out_records,
            evaluations=(),
            final_status="stopped",
            final_stop_reason="arm evaluation requires exactly 64 held-out records",
            training_record_identifiers=manifest.training_record_identifiers,
            held_out_record_identifiers=manifest.held_out_record_identifiers,
        )

    configure_deterministic_runtime()
    if arm_kind == "eggroll_only" or use_eggroll_batches:
        trainer: Any = EggrollTrainer(device=device)
    else:
        trainer = LatentCoreTrainer(lr=learning_rate, device=device)

    def evaluate() -> StabilityMetrics:
        return evaluate_held_out_metrics(
            trainer,
            manifest.held_out_records,
            device=device,
            max_decode_tokens=max_decode_tokens,
        )

    def capture_matrix_parameters() -> tuple[torch.Tensor, ...]:
        state = getattr(trainer, "state", None)
        if state is not None and hasattr(state, "eggroll_parameters"):
            return tuple(parameter.detach().clone() for parameter in state.eggroll_parameters())
        return ()

    def relative_matrix_rms(before: tuple[torch.Tensor, ...]) -> float:
        if not before:
            return 0.0
        current = capture_matrix_parameters()
        if len(current) != len(before):
            raise ValueError("trainer matrix parameter registry changed during arm")
        values = []
        for initial, updated in zip(before, current, strict=True):
            denominator = torch.sqrt(torch.mean(initial.float().square())).clamp_min(1e-12)
            values.append(
                float(torch.sqrt(torch.mean((updated.float() - initial.float()).square())).item() / denominator.item())
            )
        return max(values, default=0.0)

    evaluations: list[ArmEvaluation] = []
    examples_consumed = 0
    optimizer_call_count = 0
    final_status: Literal["active", "stopped"] = "active"
    final_stop_reason = ""
    baseline_state_hash = _compute_state_hash(trainer)
    baseline_parameters = capture_matrix_parameters()
    baseline_metrics: StabilityMetrics | None = None
    arm_started_at = time.monotonic()

    def progress_payload(event: str, batch_size: int) -> dict[str, Any]:
        elapsed_seconds = max(0.0, time.monotonic() - arm_started_at)
        eta_seconds = (
            elapsed_seconds * (len(manifest.training_records) - examples_consumed) / examples_consumed
            if examples_consumed > 0
            else None
        )
        return {
            "event": event,
            "arm": arm_kind,
            "batch_size": batch_size,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
        }

    def append_evaluation(target_checkpoint: int) -> bool:
        nonlocal baseline_metrics, final_status, final_stop_reason
        workspace = getattr(getattr(trainer, "state", None), "workspace", None)
        workspace_snapshot = workspace.snapshot() if arm_kind == "no_update" and workspace is not None else None
        try:
            metrics = evaluate()
            if baseline_metrics is None:
                baseline_metrics = metrics
            if workspace is not None and workspace_snapshot is not None:
                workspace.restore(workspace_snapshot)
                workspace_snapshot = None
            rms = relative_matrix_rms(baseline_parameters)
            metrics_drift = _compute_metrics_drift(baseline_metrics, metrics)
            current_state_hash = _compute_state_hash(trainer) if arm_kind == "no_update" else None
            control_state_unavailable = arm_kind == "no_update" and (
                baseline_state_hash is None or current_state_hash is None
            )
            control_state_drifted = (
                arm_kind == "no_update"
                and not control_state_unavailable
                and (current_state_hash != baseline_state_hash)
            )
            control_metrics_drifted = arm_kind == "no_update" and (
                metrics.to_dict() != baseline_metrics.to_dict()
                or not all(math.isfinite(value) for value in metrics_drift.values())
            )
            safety = check_arm_evaluation_safety(
                metrics,
                baseline_separation=baseline_metrics.separation_retention,
                max_rms_ceiling=MAX_UPDATE_RELATIVE_MATRIX_RMS,
                relative_matrix_rms=rms,
            )
            evaluation_status: Literal["active", "stopped"] = "active"
            stop_reason = ""
            if safety.status == "stopped":
                evaluation_status = "stopped"
                stop_reason = safety.stop_reason
                final_status = "stopped"
                final_stop_reason = stop_reason
            control_drift_reasons = []
            if control_state_unavailable:
                control_drift_reasons.append("state hash unavailable")
            if control_state_drifted:
                control_drift_reasons.append("state")
            if control_metrics_drifted:
                control_drift_reasons.append("metrics")
            if control_drift_reasons:
                evaluation_status = "stopped"
                stop_reason = f"no-update control {', '.join(control_drift_reasons)} drifted"
                final_status = "stopped"
                final_stop_reason = stop_reason
            elapsed_seconds = max(0.0, time.monotonic() - arm_started_at)
            eta_seconds = (
                elapsed_seconds * (len(manifest.training_records) - examples_consumed) / examples_consumed
                if examples_consumed > 0
                else None
            )
            evaluations.append(
                ArmEvaluation(
                    example_count=target_checkpoint,
                    metrics=metrics,
                    examples_consumed=examples_consumed,
                    optimizer_call_count=optimizer_call_count,
                    max_update_relative_matrix_rms=rms,
                    elapsed_seconds=elapsed_seconds,
                    eta_seconds=eta_seconds,
                    status=evaluation_status,
                    stop_reason=stop_reason,
                    training_record_identifiers=manifest.training_record_identifiers[:target_checkpoint],
                    held_out_record_identifiers=manifest.held_out_record_identifiers,
                )
            )
        except Exception as error:  # noqa: BLE001 - preserve the stopping evidence
            final_status = "stopped"
            final_stop_reason = f"{type(error).__name__}: {error!s}"
            return False
        else:
            return evaluation_status == "active"
        finally:
            if workspace is not None and workspace_snapshot is not None:
                workspace.restore(workspace_snapshot)

    for target_checkpoint in manifest.checkpoint_example_counts:
        if target_checkpoint == 0:
            if not append_evaluation(0):
                break
        else:
            if target_checkpoint <= examples_consumed:
                continue

            if arm_kind == "no_update":
                examples_consumed = target_checkpoint
                if progress_observer is not None:
                    progress_observer(
                        examples_consumed,
                        optimizer_call_count,
                        progress_payload("cursor_advance", 0),
                    )
            elif arm_kind == "eggroll_only" or use_eggroll_batches:
                while examples_consumed < target_checkpoint:
                    batch_end = min(examples_consumed + 8, target_checkpoint)
                    records = tuple(
                        AsdivRecord(
                            id=f"trainability-{index}",
                            split="train",
                            question=manifest.training_records[index][0],
                            target=manifest.training_records[index][1],
                        )
                        for index in range(examples_consumed, batch_end)
                    )
                    batch = FitnessBatch(
                        records=records,
                        start_position=examples_consumed,
                        next_position=batch_end,
                    )
                    try:
                        result = trainer.train_fitness_batch(
                            batch,
                            optimizer_call_count=optimizer_call_count + 1,
                        )
                        if not math.isfinite(result.language_model_loss) or not math.isfinite(result.total_objective):
                            final_status = "stopped"
                            final_stop_reason = "EGGROLL update returned a non-finite objective"
                            break
                    except Exception as error:  # noqa: BLE001 - preserve the stopping evidence
                        final_status = "stopped"
                        final_stop_reason = f"{type(error).__name__}: {error!s}"
                        break
                    optimizer_call_count += 1
                    examples_consumed = batch_end
                    if progress_observer is not None:
                        progress_observer(
                            examples_consumed,
                            optimizer_call_count,
                            progress_payload("update", len(records)),
                        )
            else:
                while examples_consumed < target_checkpoint:
                    record_idx = examples_consumed
                    try:
                        result = trainer.train_step(
                            manifest.training_records[record_idx][0],
                            manifest.training_records[record_idx][1],
                        )
                        if not math.isfinite(result.language_model_loss) or not math.isfinite(result.total_objective):
                            final_status = "stopped"
                            final_stop_reason = "gradient update returned a non-finite objective"
                            break
                    except Exception as error:  # noqa: BLE001 - preserve the stopping evidence
                        final_status = "stopped"
                        final_stop_reason = f"{type(error).__name__}: {error!s}"
                        break
                    optimizer_call_count += 1
                    examples_consumed += 1
                    if progress_observer is not None:
                        progress_observer(
                            examples_consumed,
                            optimizer_call_count,
                            progress_payload("update", 1),
                        )

            if final_status == "stopped":
                break

            if not append_evaluation(target_checkpoint):
                break

    return MethodArm(
        arm_kind=arm_kind,
        training_records=manifest.training_records,
        held_out_records=manifest.held_out_records,
        evaluations=tuple(evaluations),
        final_status=final_status,
        final_stop_reason=final_stop_reason,
        training_record_identifiers=manifest.training_record_identifiers,
        held_out_record_identifiers=manifest.held_out_record_identifiers,
    )


def _compute_state_hash(trainer: Any) -> str | None:
    """Compute a hash of the trainer's current parameter state."""
    try:
        state = getattr(trainer, "state", None)
        if state is None or not hasattr(state, "trainable_params"):
            return None
        digest = hashlib.sha256()
        for name, parameter in state.trainable_params.items():
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().numpy().tobytes())
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None:
            digest.update(canonical_json_bytes(_serialize_optimizer_state(optimizer.state_dict())))
        digest.update(torch.get_rng_state().numpy().tobytes())
        if torch.cuda.is_available():
            for rng_state in torch.cuda.get_rng_state_all():
                digest.update(rng_state.cpu().numpy().tobytes())
        digest.update(canonical_json_bytes(random.getstate()))
        numpy_state = cast(tuple[str, np.ndarray, int, int, float], np.random.get_state())
        state_name, state_array, state_position, has_gaussian, cached_gaussian = numpy_state
        digest.update(state_name.encode("utf-8"))
        digest.update(state_array.tobytes())
        digest.update(canonical_json_bytes((state_position, has_gaussian, cached_gaussian)))
        workspace = getattr(state, "workspace", None)
        if workspace is not None and hasattr(workspace, "read_slots"):
            digest.update(workspace.read_slots().detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    except Exception:  # noqa: BLE001 - a missing digest invalidates the control arm
        return None


def _serialize_optimizer_state(value: Any) -> Any:
    """Convert optimizer state tensors into finite canonical hash input."""
    if isinstance(value, torch.Tensor):
        return {
            "tensor": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "bytes": hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(key): _serialize_optimizer_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_optimizer_state(item) for item in value]
    return value


def _compute_metrics_drift(baseline: StabilityMetrics, final: StabilityMetrics) -> dict[str, float]:
    """Compute the drift between baseline and final metrics."""
    drift = {}
    if baseline.language_model_loss > 0:
        drift["lm_loss_ratio"] = final.language_model_loss / baseline.language_model_loss
    else:
        drift["lm_loss_ratio"] = 0.0

    drift["exact_accuracy_delta"] = final.exact_accuracy - baseline.exact_accuracy
    drift["first_token_accuracy_delta"] = final.first_token_accuracy - baseline.first_token_accuracy
    drift["separation_retention_ratio"] = (
        final.separation_retention / baseline.separation_retention if baseline.separation_retention > 0 else 0.0
    )

    return drift


def run_no_update_arm(
    manifest: FreshStateManifest,
    device: str = "cpu",
    *,
    progress_observer: Callable[[int, int, dict[str, Any]], None] | None = None,
    max_decode_tokens: int = DEFAULT_MAX_TOKENS,
) -> MethodArm:
    """Run the no-update control arm over 32 records with baseline comparison.

    Args:
        manifest: Fresh-state manifest with training records and checkpoints
        device: Device to use for computation

    Returns:
        MethodArm with evaluations at each checkpoint (0, 8, 32 examples),
        tracking state drift against baseline for control validation
    """
    return _run_arm_with_updates(
        arm_kind="no_update",
        manifest=manifest,
        learning_rate=0.0,
        device=device,
        use_eggroll_batches=False,
        progress_observer=progress_observer,
        max_decode_tokens=max_decode_tokens,
    )


def run_gradient_only_arm(
    manifest: FreshStateManifest,
    device: str = "cpu",
    *,
    progress_observer: Callable[[int, int, dict[str, Any]], None] | None = None,
    max_decode_tokens: int = DEFAULT_MAX_TOKENS,
) -> MethodArm:
    """Run the gradient-only method arm over 32 records.

    Args:
        manifest: Fresh-state manifest with training records and checkpoints
        device: Device to use for computation

    Returns:
        MethodArm with evaluations at each checkpoint (0, 8, 32 examples)
    """
    return _run_arm_with_updates(
        arm_kind="gradient_only",
        manifest=manifest,
        learning_rate=0.001,
        device=device,
        use_eggroll_batches=False,
        progress_observer=progress_observer,
        max_decode_tokens=max_decode_tokens,
    )


def run_eggroll_only_arm(
    manifest: FreshStateManifest,
    device: str = "cpu",
    *,
    progress_observer: Callable[[int, int, dict[str, Any]], None] | None = None,
    max_decode_tokens: int = DEFAULT_MAX_TOKENS,
) -> MethodArm:
    """Run the EGGROLL-only method arm over 32 records with batch capping.

    Args:
        manifest: Fresh-state manifest with training records and checkpoints
        device: Device to use for computation

    Returns:
        MethodArm with evaluations at each checkpoint (0, 8, 32 examples),
        with EGGROLL batch capping at 8 and 32 record boundaries
    """
    return _run_arm_with_updates(
        arm_kind="eggroll_only",
        manifest=manifest,
        learning_rate=0.001,
        device=device,
        use_eggroll_batches=True,
        progress_observer=progress_observer,
        max_decode_tokens=max_decode_tokens,
    )
