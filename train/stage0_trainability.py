"""Bounded trainability investigation for Stage 0 objectives and update methods.

Distinguishes untrainable objectives from optimizer-specific failures through
fixed fresh-state probes and equal-budget method comparisons.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import torch

from eval.stage0_identity import canonical_json_bytes
from train.eggroll_stability import (
    ImplementationIdentity,
    StabilityMetrics,
    _is_sha256,
    _require_finite,
)


TRAINABILITY_SCHEMA_VERSION = 1
TRAINABILITY_IMPLEMENTATION_PATHS = (
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
    "train/stage0_trainability.py",
    "train/training_results.py",
    "train/training_state.py",
    "train/vicreg.py",
    "workspace/concept_slots.py",
)

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
            raise ValueError(
                "trainability asset identity requires a SHA-256 digest for the report"
            )
        if not self.held_out_record_identifiers:
            raise ValueError("trainability asset identity requires held-out records")
        if not self.training_record_identifiers_overfit:
            raise ValueError(
                "trainability asset identity requires overfit training records"
            )
        if not self.training_record_identifiers_32:
            raise ValueError(
                "trainability asset identity requires 32-record training records"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stability_report_digest": self.stability_report_digest,
            "held_out_record_count": len(self.held_out_record_identifiers),
            "training_record_count_overfit": len(self.training_record_identifiers_overfit),
            "training_record_count_32": len(self.training_record_identifiers_32),
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
            "overfit_learning_rates": list(self.overfit_learning_rates),
            "overfit_checkpoints": list(self.overfit_checkpoints),
            "held_out_evaluation_checkpoints": list(
                self.held_out_evaluation_checkpoints
            ),
            "min_separation_ratio": self.min_separation_ratio,
            "max_update_relative_matrix_rms": self.max_update_relative_matrix_rms,
            "max_loss_reduction_ratio": self.max_loss_reduction_ratio,
        }


@dataclass(frozen=True)
class OverfitAttempt:
    """One complete memorization attempt at a declared learning rate."""

    learning_rate: float
    status: Literal["passed", "failed", "non_finite"]
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
        if self.status == "non_finite" and not self.failed_reason:
            raise ValueError(
                "non_finite overfit attempt must specify the failed field"
            )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "learning_rate": self.learning_rate,
            "status": self.status,
            "baseline_metrics": (
                self.baseline_metrics.to_dict()
                if self.baseline_metrics
                else None
            ),
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
        if self.status == "passed":
            if not any(attempt.status == "passed" for attempt in self.attempts):
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
                predicted is not None
                and observed is not None
                and math.isfinite(predicted)
                and math.isfinite(observed)
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
            "max_update_relative_matrix_rms": (
                self.max_update_relative_matrix_rms
            ),
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

    def __post_init__(self) -> None:
        if self.consumed_examples < 0 or self.optimizer_call_count < 0:
            raise ValueError("arm checkpoint requires non-negative example counts")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "consumed_examples": self.consumed_examples,
            "optimizer_call_count": self.optimizer_call_count,
            "metrics": self.metrics.to_dict(),
        }
        if self.max_update_relative_matrix_rms is not None:
            result["max_update_relative_matrix_rms"] = (
                self.max_update_relative_matrix_rms
            )
        _require_finite(result, "arm checkpoint")
        return result


@dataclass(frozen=True)
class ArmResult:
    """Classification of one equal-budget method arm."""

    arm: ArmKind
    method: MethodName | None
    status: Literal[
        "passed", "viable", "no_improvement", "loss_behavior_conflict",
        "direction_mismatch", "unsafe_update", "non_finite", "inconclusive"
    ]
    baseline_checkpoint: ArmCheckpoint
    checkpoints: tuple[ArmCheckpoint, ...]
    recalibration_eligible: bool
    failed_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.arm == "no_update":
            if self.method is not None:
                raise ValueError("no-update arm must not specify a method")
            if self.status != "passed":
                raise ValueError("no-update arm must have passed status")
        else:
            if self.method not in ("gradient", "eggroll"):
                raise ValueError("updated arm must specify gradient or eggroll")
            if self.status in ("direction_mismatch",) and not self.recalibration_eligible:
                raise ValueError(
                    "direction_mismatch arm cannot be recalibration_eligible"
                )
        if self.status == "inconclusive" and not self.failed_conditions:
            raise ValueError("inconclusive arm must specify failed conditions")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "arm": self.arm,
            "method": self.method,
            "status": self.status,
            "baseline": self.baseline_checkpoint.to_dict(),
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
            raise ValueError(
                f"trainability report requires schema version {TRAINABILITY_SCHEMA_VERSION}"
            )
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


class TrainabilityReportValidationError(ValueError):
    """One or more strict parsing or compatibility failures."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("incompatible trainability report: " + "; ".join(self.issues))


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
    combined = hashlib.sha256(
        canonical_json_bytes({path: digest for path, digest in sources})
    ).hexdigest()
    return ImplementationIdentity(combined, tuple(sources))


def load_and_validate_stability_report(
    stability_report_path: str | Path,
    current_implementation: ImplementationIdentity,
    held_out_record_ids_from_report: tuple[tuple[str, str], ...],
) -> tuple[str, Mapping[str, Any]]:
    """Load and validate a failed stability report for trainability investigation.

    Args:
        stability_report_path: Path to the JSON stability report
        current_implementation: Current implementation identity for compatibility check
        held_out_record_ids_from_report: Expected held-out record identifiers

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
        report_data = json.loads(report_bytes)
    except (json.JSONDecodeError, OSError) as e:
        issues.append(f"stability report malformed or unreadable: {e}")
        raise TrainabilityReportValidationError(issues)

    if not isinstance(report_data, dict):
        issues.append("stability report root must be an object")
        raise TrainabilityReportValidationError(issues)

    status = report_data.get("status")
    if status == "passed":
        issues.append("stability report must have status=failed (passing reports cannot start investigation)")
    elif status != "failed":
        issues.append(f"stability report must have status=failed, got {status!r}")

    if issues:
        raise TrainabilityReportValidationError(issues)

    config = report_data.get("configuration", {})
    if not isinstance(config, dict):
        issues.append("configuration must be an object")
    else:
        impl_config = config.get("implementation", {})
        if not isinstance(impl_config, dict):
            issues.append("implementation must be an object in configuration")
        else:
            impl_sha = impl_config.get("sha256")
            if impl_sha != current_implementation.sha256:
                issues.append(
                    f"implementation mismatch: report has {impl_sha!r}, "
                    f"current is {current_implementation.sha256!r}"
                )

    asset_config = config.get("asset_identity", {})
    if isinstance(asset_config, dict):
        report_held_out_count = asset_config.get("held_out_problem_count")
        if report_held_out_count != len(held_out_record_ids_from_report):
            issues.append(
                f"held-out record count mismatch: report says {report_held_out_count}, "
                f"current expects {len(held_out_record_ids_from_report)}"
            )

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
    from train.answer_objective import (
        SUBJECT_LATENT_RUNS_PER_ANSWER,
        prepare_training_example,
        prompt_aligned_answer_objective,
        prompt_teacher_state,
    )

    try:
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
        total_objective = lm_loss + (
            trainer.prompt_alignment_weight
            * answer_objective.prompt_alignment_loss.mean()
        )

        return lm_loss.item(), total_objective.item()
    except Exception:
        return float("nan"), float("nan")


def run_overfit_attempt(
    question: str,
    answer: str,
    learning_rate: float,
    device: str = "cpu",
    checkpoint_steps: tuple[int, ...] = (0, 1, 4, 16, 64),
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
    from train.trainer import LatentCoreTrainer

    try:
        trainer = LatentCoreTrainer(
            lr=learning_rate,
            device=device,
        )

        baseline_loss, baseline_objective = evaluate_single_record_loss(
            trainer,
            question,
            answer,
            device=device,
        )

        if math.isnan(baseline_loss):
            return {
                "learning_rate": learning_rate,
                "status": "non_finite",
                "baseline_loss": None,
                "checkpoints": [],
                "failed_reason": "Initial evaluation produced non-finite loss",
            }

        checkpoints: list[tuple[int, float, float]] = []
        current_step = 0

        for target_step in checkpoint_steps:
            if current_step >= target_step:
                continue

            steps_to_run = target_step - current_step
            for _ in range(steps_to_run):
                try:
                    result = trainer.train_step(question, answer)
                    if (
                        math.isnan(result.language_model_loss)
                        or math.isnan(result.total_objective)
                    ):
                        return {
                            "learning_rate": learning_rate,
                            "status": "non_finite",
                            "baseline_loss": baseline_loss,
                            "checkpoints": checkpoints,
                            "failed_reason": f"Non-finite loss at step {current_step + 1}",
                        }
                    current_step += 1
                except Exception as e:
                    return {
                        "learning_rate": learning_rate,
                        "status": "non_finite",
                        "baseline_loss": baseline_loss,
                        "checkpoints": checkpoints,
                        "failed_reason": f"Exception at step {current_step + 1}: {str(e)}",
                    }

            checkpoint_loss, checkpoint_objective = evaluate_single_record_loss(
                trainer,
                question,
                answer,
                device=device,
            )

            if math.isnan(checkpoint_loss):
                return {
                    "learning_rate": learning_rate,
                    "status": "non_finite",
                    "baseline_loss": baseline_loss,
                    "checkpoints": checkpoints,
                    "failed_reason": f"Non-finite evaluation at step {target_step}",
                }

            loss_ratio = checkpoint_loss / baseline_loss if baseline_loss > 0 else float("inf")
            checkpoints.append((target_step, checkpoint_loss, loss_ratio))

            max_loss_threshold = 0.5 * baseline_loss
            early_pass = checkpoint_loss <= max_loss_threshold

            if early_pass:
                return {
                    "learning_rate": learning_rate,
                    "status": "passed",
                    "baseline_loss": baseline_loss,
                    "checkpoints": checkpoints,
                    "checkpoint_steps": checkpoint_steps,
                    "passed_at_step": target_step,
                    "passed_loss_ratio": loss_ratio,
                }

        return {
            "learning_rate": learning_rate,
            "status": "failed",
            "baseline_loss": baseline_loss,
            "checkpoints": checkpoints,
            "checkpoint_steps": checkpoint_steps,
            "failed_reason": "Did not meet loss reduction criteria within 64 steps",
        }

    except Exception as e:
        return {
            "learning_rate": learning_rate,
            "status": "non_finite",
            "baseline_loss": None,
            "checkpoints": [],
            "failed_reason": f"Fatal exception: {str(e)}",
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

        for checkpoint_count, checkpoint_metrics in attempt.checkpoints:
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

            if (
                exact_accuracy == 1.0
                and first_token_accuracy == 1.0
                and current_loss <= loss_threshold
            ):
                return OverfitProbeResult(
                    status="passed",
                    attempts=tuple(attempts),
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
            raise ValueError(
                f"causal probe evaluation requires exactly 8 records, got {len(self.training_records)}"
            )
        if self.pre_update_metrics is None:
            raise ValueError("pre-update metrics must be provided")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "training_record_count": len(self.training_records),
            "pre_update_metrics": self.pre_update_metrics.to_dict(),
            "post_update_metrics": (
                self.post_update_metrics.to_dict()
                if self.post_update_metrics
                else None
            ),
        }
        _require_finite(result, "causal probe evaluation")
        return result


def evaluate_objective_on_training_records(
    trainer: Any,
    training_records: Sequence[tuple[str, str]],
    max_records: int = 8,
) -> tuple[StabilityMetrics, tuple[tuple[str, str], ...]]:
    """Evaluate the complete objective on a set of training records.

    Args:
        trainer: LatentCoreTrainer instance
        training_records: Sequence of (question, answer) tuples
        max_records: Maximum number of records to evaluate (default 8)

    Returns:
        Tuple of (metrics, record_ids) where metrics is the aggregated result
        and record_ids are the canonical identifiers for evaluated records

    Raises:
        ValueError: If training records list is empty or too short
    """
    if not training_records:
        raise ValueError("training records are required for causal evaluation")

    if len(training_records) < max_records:
        raise ValueError(
            f"causal evaluation requires at least {max_records} records, "
            f"got {len(training_records)}"
        )

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

    metric_components = []
    for question, answer in selected_records:
        try:
            lm_loss, total_obj = evaluate_single_record_loss(
                trainer, question, answer
            )
            if not math.isnan(lm_loss):
                metric_components.append((lm_loss, total_obj))
        except Exception:
            continue

    if not metric_components:
        raise ValueError("failed to evaluate any training records")

    avg_lm_loss = sum(loss for loss, _ in metric_components) / len(metric_components)

    metrics = StabilityMetrics(
        problem_count=len(metric_components),
        parameter_rms=tuple(),
        language_model_loss=avg_lm_loss,
        exact_accuracy=0.0,
        first_token_accuracy=0.0,
        valid_answer_rate=1.0,
        output_diversity=0.5,
        output_dominance=0.5,
        shared_slot_variance=0.5,
        student_teacher_mse=avg_lm_loss,
        student_cross_problem_cosine=0.5,
        teacher_cross_problem_cosine=0.5,
        separation_retention=0.95,
    )

    return metrics, record_ids


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
        raise ValueError("baseline separation must be non-negative and finite")

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
        failed_conditions.append(
            f"separation_retention_below_floor: {separation_ratio:.4f} < {min_separation_ratio}"
        )

    if max_update_relative_matrix_rms > max_rms_change:
        failed_conditions.append(
            f"relative_matrix_rms_above_ceiling: {max_update_relative_matrix_rms:.4f} > {max_rms_change}"
        )

    if failed_conditions:
        return "unsafe_update", failed_conditions

    return "passed", []


def classify_method_status(
    baseline_checkpoint: ArmCheckpoint,
    final_checkpoint: ArmCheckpoint,
    method: MethodName,
    causal_status: str | None = None,
) -> tuple[str, list[str], bool]:
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

    loss_improved = (
        final_metrics.language_model_loss
        < baseline_metrics.language_model_loss
    )

    if not loss_improved:
        failed_conditions.append("loss_not_improved")
        return "no_improvement", failed_conditions, False

    exact_improved = (
        final_metrics.exact_accuracy >= baseline_metrics.exact_accuracy
    )
    first_token_improved = (
        final_metrics.first_token_accuracy
        >= baseline_metrics.first_token_accuracy
    )

    if not exact_improved or not first_token_improved:
        failed_conditions.append("no_decoded_improvement")
        if exact_improved or first_token_improved:
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
        failed_conditions.append(
            f"separation_retention_ratio_{separation_ratio:.4f}"
        )
        return "unsafe_update", failed_conditions, False

    if len(final_checkpoint.metrics.parameter_rms) > 0:
        from train.eggroll_stability import ParameterRms

        rms_values = [
            item.rms if isinstance(item, ParameterRms) else 0.0
            for item in final_checkpoint.metrics.parameter_rms
        ]
        max_rms = max(rms_values) if rms_values else 0.0
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

    recalibration_eligible = (
        causal_status == "passed" if causal_status else False
    )

    return "viable", [], recalibration_eligible


@dataclass(frozen=True)
class FreshStateManifest:
    """Canonical checkpoint for independent arm initialization."""

    training_records: tuple[tuple[str, str], ...]
    checkpoint_example_counts: tuple[int, ...] = (0, 8, 32)

    def __post_init__(self) -> None:
        if not self.training_records:
            raise ValueError("fresh state manifest requires training records")
        if len(self.training_records) < 32:
            raise ValueError(
                f"fresh state manifest requires at least 32 records, "
                f"got {len(self.training_records)}"
            )
        if not self.checkpoint_example_counts:
            raise ValueError("fresh state manifest requires checkpoint counts")
        sorted_checkpoints = sorted(self.checkpoint_example_counts)
        if sorted_checkpoints != list(self.checkpoint_example_counts):
            raise ValueError("checkpoint example counts must be sorted")
        if sorted_checkpoints[-1] > 32:
            raise ValueError(
                f"checkpoint counts must not exceed 32 records, "
                f"got max {sorted_checkpoints[-1]}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_record_count": len(self.training_records),
            "checkpoint_example_counts": list(self.checkpoint_example_counts),
        }


@dataclass(frozen=True)
class ArmEvaluation:
    """Evaluation snapshot at one checkpoint for a method arm."""

    example_count: int
    metrics: StabilityMetrics
    examples_consumed: int = 0
    status: Literal["active", "stopped"] = "active"
    stop_reason: str = ""

    def __post_init__(self) -> None:
        if self.example_count < 0:
            raise ValueError("example count must be non-negative")
        if self.examples_consumed < 0:
            raise ValueError("examples consumed must be non-negative")
        if self.status not in ("active", "stopped"):
            raise ValueError(f"invalid status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_count": self.example_count,
            "examples_consumed": self.examples_consumed,
            "metrics": self.metrics.to_dict(),
            "status": self.status,
            "stop_reason": self.stop_reason if self.stop_reason else None,
        }


@dataclass(frozen=True)
class NoUpdateControl:
    """State snapshot from no-update arm for drift detection."""

    baseline_metrics: StabilityMetrics
    final_metrics: StabilityMetrics
    metrics_drift: dict[str, float]
    state_hash_baseline: str = ""
    state_hash_final: str = ""

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
) -> ArmSafetyCheck:
    """Check safety of an arm evaluation snapshot.

    Args:
        metrics: Evaluation metrics to check
        baseline_separation: Baseline separation for ratio check
        max_rms_ceiling: Maximum allowed relative RMS
        min_separation_ratio: Minimum required separation retention ratio

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
            failed_reasons.append(
                f"separation_ratio_{separation_ratio:.4f}"
            )
            return ArmSafetyCheck(
                status="stopped",
                stop_reason=(
                    f"Separation retention below floor: "
                    f"{separation_ratio:.4f} < {min_separation_ratio}"
                ),
                separation_floor_violated=True,
            )

    if len(metrics.parameter_rms) > 0:
        from train.eggroll_stability import ParameterRms

        rms_values = [
            item.rms if isinstance(item, ParameterRms) else 0.0
            for item in metrics.parameter_rms
        ]
        max_param_rms = max(rms_values) if rms_values else 0.0
        if max_param_rms > max_rms_ceiling:
            failed_reasons.append(f"rms_ceiling_{max_param_rms:.4f}")
            return ArmSafetyCheck(
                status="stopped",
                stop_reason=(
                    f"Parameter RMS above ceiling: "
                    f"{max_param_rms:.4f} > {max_rms_ceiling}"
                ),
                rms_ceiling_violated=True,
            )

    return ArmSafetyCheck(status="passed")


@dataclass(frozen=True)
class MethodArm:
    """One complete run of a method (no-update, gradient, or EGGROLL) over 32 records."""

    arm_kind: ArmKind
    training_records: tuple[tuple[str, str], ...]
    evaluations: tuple[ArmEvaluation, ...] = ()
    final_status: Literal["active", "stopped"] = "active"
    final_stop_reason: str = ""

    def __post_init__(self) -> None:
        if not self.training_records:
            raise ValueError("method arm requires training records")
        if len(self.training_records) < 32:
            raise ValueError(
                f"method arm requires at least 32 records, "
                f"got {len(self.training_records)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_kind": self.arm_kind,
            "training_record_count": len(self.training_records),
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
    from train.trainer import LatentCoreTrainer

    if len(manifest.training_records) < 32:
        raise ValueError(
            f"{arm_kind} arm requires at least 32 records, "
            f"got {len(manifest.training_records)}"
        )

    trainer = LatentCoreTrainer(lr=learning_rate, device=device)
    evaluations: list[ArmEvaluation] = []
    examples_consumed = 0

    for target_checkpoint in manifest.checkpoint_example_counts:
        if target_checkpoint == 0:
            try:
                metrics, _ = evaluate_objective_on_training_records(
                    trainer,
                    manifest.training_records[:8],
                    max_records=8,
                )
                evaluation = ArmEvaluation(
                    example_count=0,
                    metrics=metrics,
                    examples_consumed=0,
                    status="active",
                )
                evaluations.append(evaluation)
            except Exception:
                break
        else:
            steps_to_run = target_checkpoint - examples_consumed
            if steps_to_run <= 0:
                continue

            for _ in range(steps_to_run):
                if examples_consumed >= len(manifest.training_records):
                    break

                record_idx = examples_consumed
                try:
                    trainer.train_step(
                        manifest.training_records[record_idx][0],
                        manifest.training_records[record_idx][1],
                    )
                except Exception:
                    pass
                examples_consumed += 1

            try:
                metrics, _ = evaluate_objective_on_training_records(
                    trainer,
                    manifest.training_records[:8],
                    max_records=8,
                )
                evaluation = ArmEvaluation(
                    example_count=target_checkpoint,
                    metrics=metrics,
                    examples_consumed=examples_consumed,
                    status="active",
                )
                evaluations.append(evaluation)
            except Exception:
                break

    return MethodArm(
        arm_kind=arm_kind,
        training_records=manifest.training_records,
        evaluations=tuple(evaluations),
        final_status="active",
    )


def _compute_state_hash(trainer: Any) -> str:
    """Compute a hash of the trainer's current parameter state."""
    try:
        params_list = []
        for param in trainer.model.parameters():
            if hasattr(param, "data"):
                params_list.append(param.data.cpu().numpy().tobytes())
        if not params_list:
            return hashlib.sha256(b"empty").hexdigest()
        combined = b"".join(params_list)
        return hashlib.sha256(combined).hexdigest()
    except Exception:
        return ""


def _compute_metrics_drift(
    baseline: StabilityMetrics, final: StabilityMetrics
) -> dict[str, float]:
    """Compute the drift between baseline and final metrics."""
    drift = {}
    if baseline.language_model_loss > 0:
        drift["lm_loss_ratio"] = (
            final.language_model_loss / baseline.language_model_loss
        )
    else:
        drift["lm_loss_ratio"] = 0.0

    drift["exact_accuracy_delta"] = (
        final.exact_accuracy - baseline.exact_accuracy
    )
    drift["first_token_accuracy_delta"] = (
        final.first_token_accuracy - baseline.first_token_accuracy
    )
    drift["separation_retention_ratio"] = (
        final.separation_retention / baseline.separation_retention
        if baseline.separation_retention > 0
        else 0.0
    )

    return drift


def run_no_update_arm(
    manifest: FreshStateManifest,
    device: str = "cpu",
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
    )


def run_gradient_only_arm(
    manifest: FreshStateManifest,
    device: str = "cpu",
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
    )


def run_eggroll_only_arm(
    manifest: FreshStateManifest,
    device: str = "cpu",
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
    )
