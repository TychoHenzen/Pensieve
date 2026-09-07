"""Run the bounded Stage 0 trainability investigation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from codecs_module.decoder import DEFAULT_MAX_TOKENS
from eval.stage0_identity import STAGE0_HELD_OUT_ITEM_IDS
from eval.stream.generators.asdiv_a import AsdivRecord
from train.eggroll_trainer import EggrollTrainer
from train.stage0_data import FitnessBatch, load_stage0_dataset
from train.stage0_trainability import (
    OVERFIT_CHECKPOINTS,
    OVERFIT_LEARNING_RATES,
    TRAINABILITY_SCHEMA_VERSION,
    ArmCheckpoint,
    ArmResult,
    CausalProbeResult,
    FreshStateManifest,
    MethodArm,
    MethodName,
    OverfitAttempt,
    OverfitProbeResult,
    StabilityMetrics,
    TrainabilityAssetIdentity,
    TrainabilityConfiguration,
    TrainabilityProgressRecord,
    TrainabilityReport,
    build_trainability_record_selections,
    canonical_json_bytes,
    canonical_trainability_implementation_identity,
    classify_causal_result,
    classify_method_status,
    classify_overall_trainability,
    classify_overfit_probe,
    compute_initial_state_digest,
    compute_recalibration_eligibility,
    evaluate_held_out_metrics,
    evaluate_single_record_loss,
    load_and_validate_stability_report,
    run_eggroll_only_arm,
    run_gradient_only_arm,
    run_no_update_arm,
    run_overfit_attempt,
    validate_causal_safety_checks,
)
from train.standalone_checkpoint import configure_deterministic_runtime
from train.trainer import LatentCoreTrainer


def _compute_exit_code(status: str) -> int:
    """Return zero only when bounded trainability was observed."""
    return 0 if status == "bounded_trainability_observed" else 1


class TrainabilityProgressWriter:
    """Append-only, exclusive-create JSONL progress writer."""

    def __init__(self, progress_path: Path) -> None:
        self.progress_path = progress_path
        self.sequence_number = 0
        self._file_handle = None
        self.final_report: TrainabilityReport | None = None
        self.started_at = time.monotonic()

    def __enter__(self) -> TrainabilityProgressWriter:
        self._file_handle = self.progress_path.open("x", encoding="utf-8")
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        if self._file_handle:
            self._file_handle.close()

    def write_record(self, record: TrainabilityProgressRecord) -> None:
        """Write one canonical record, flush it, and advance its sequence."""
        if not self._file_handle:
            raise RuntimeError("progress writer not in context manager")
        if record.sequence_number != self.sequence_number:
            raise ValueError(f"progress sequence must be {self.sequence_number}, got {record.sequence_number}")
        self._file_handle.write(record.canonical_json_line() + "\n")
        self._file_handle.flush()
        self.sequence_number += 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bounded trainability investigation for Stage 0.")
    parser.add_argument("--stability-report", type=Path, required=True)
    parser.add_argument("--final-output", type=Path, required=True)
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-decode-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid inputs before dataset or model construction."""
    if not args.stability_report.exists():
        raise FileNotFoundError(f"stability report not found: {args.stability_report}")

    progress_path = args.progress_output or args.final_output.with_suffix(".jsonl")
    output_identity = os.path.normcase(str(args.final_output.resolve()))
    progress_identity = os.path.normcase(str(progress_path.resolve()))
    if output_identity == progress_identity:
        raise ValueError("--final-output and --progress-output must use distinct paths")
    if args.final_output.exists():
        raise FileExistsError(f"refusing to overwrite trainability output: {args.final_output}")
    if progress_path.exists():
        raise FileExistsError(f"refusing to overwrite trainability progress: {progress_path}")
    if isinstance(args.max_decode_tokens, bool) or args.max_decode_tokens < 1:
        raise ValueError("--max-decode-tokens must be at least 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"requested CUDA device {args.device!r} is unavailable")

    try:
        report_data = json.loads(
            args.stability_report.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON number {value}")),
        )
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"stability report malformed or unreadable: {error}") from error
    if not isinstance(report_data, dict):
        raise TypeError("stability report root must be an object")
    status = report_data.get("status")
    if status == "passed":
        raise ValueError("stability report must have status=failed (passing reports cannot start investigation)")
    if status != "failed":
        raise ValueError(f"stability report must have status=failed, got {status!r}")


def _emit(
    writer: TrainabilityProgressWriter,
    kind: str,
    probe_or_arm: str,
    *,
    consumed_examples: int = 0,
    optimizer_call_count: int = 0,
    elapsed_seconds: float | None = None,
    eta_seconds: float | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    elapsed = max(0.0, elapsed_seconds if elapsed_seconds is not None else time.monotonic() - writer.started_at)
    writer.write_record(
        TrainabilityProgressRecord(
            schema_version=TRAINABILITY_SCHEMA_VERSION,
            sequence_number=writer.sequence_number,
            kind=kind,  # type: ignore[arg-type]
            probe_or_arm=probe_or_arm,
            consumed_examples=consumed_examples,
            optimizer_call_count=optimizer_call_count,
            elapsed_seconds=elapsed,
            eta_seconds=eta_seconds,
            payload=payload or {},
        )
    )


def _record_pairs(records: Sequence[AsdivRecord]) -> tuple[tuple[str, str], ...]:
    return tuple((record.question, record.target) for record in records)


def _state_digest(trainer: Any) -> str:
    state = getattr(trainer, "state", None)
    parameters = getattr(state, "trainable_params", None)
    if not isinstance(parameters, dict):
        raise TypeError("fresh trainer does not expose its trainable parameter registry")
    return compute_initial_state_digest(
        tuple((name, parameter.detach().clone()) for name, parameter in parameters.items())
    )


def _overfit_metrics(
    trainer: Any, question: str, answer: str, *, device: str, max_decode_tokens: int
) -> StabilityMetrics:
    """Evaluate the one-record target in the shared 64-row metric shape."""
    return evaluate_held_out_metrics(
        trainer,
        tuple((question, answer) for _ in range(64)),
        device=device,
        max_decode_tokens=max_decode_tokens,
    )


def _typed_overfit_attempt(raw: dict[str, Any]) -> OverfitAttempt:
    status = raw.get("status")
    if status not in {"passed", "failed", "non_finite"}:
        status = "non_finite"
    baseline_metrics = raw.get("baseline_metrics")
    checkpoint_metrics = {
        int(step): metrics
        for step, metrics in raw.get("metric_checkpoints", ())
        if isinstance(metrics, StabilityMetrics)
    }
    checkpoints = tuple(
        (int(step), checkpoint_metrics.get(int(step))) for step, _loss, _ratio in raw.get("checkpoints", ())
    )
    if status == "passed" and (not isinstance(baseline_metrics, StabilityMetrics) or not checkpoints):
        status = "failed"
        failed_reason = "decoded metric evidence missing from passing attempt"
    else:
        failed_reason = str(raw.get("failed_reason", ""))
    if status == "non_finite" and not failed_reason:
        failed_reason = "attempt produced non-finite evidence"
    return OverfitAttempt(
        learning_rate=float(raw["learning_rate"]),
        status=status,
        baseline_metrics=baseline_metrics if isinstance(baseline_metrics, StabilityMetrics) else None,
        checkpoints=checkpoints,
        failed_reason=failed_reason,
    )


def _capture_matrix_parameters(trainer: Any) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in trainer.state.eggroll_parameters())


def _relative_matrix_rms(before: tuple[torch.Tensor, ...], trainer: Any) -> float:
    after = _capture_matrix_parameters(trainer)
    if len(before) != len(after):
        raise ValueError("trainer matrix parameter registry changed during causal probe")
    values = []
    for initial, updated in zip(before, after, strict=True):
        denominator = torch.sqrt(torch.mean(initial.float().square())).clamp_min(1e-12)
        values.append(
            float(torch.sqrt(torch.mean((updated.float() - initial.float()).square())).item() / denominator.item())
        )
    return max(values, default=0.0)


def _support_objective(trainer: Any, records: Sequence[tuple[str, str]]) -> float:
    if len(records) != 8:
        raise ValueError(f"causal probe requires exactly 8 support records, got {len(records)}")
    values = [evaluate_single_record_loss(trainer, question, answer)[1] for question, answer in records]
    if not values or not all(torch.isfinite(torch.tensor(value)) for value in values):
        raise ValueError("support objective contains a non-finite value")
    return sum(values) / len(values)


def _run_causal_probe(
    method: str,
    support_records: tuple[tuple[str, str], ...],
    held_out_records: tuple[tuple[str, str], ...],
    *,
    device: str,
    max_decode_tokens: int,
) -> CausalProbeResult:
    """Run one fresh-state, one-update causal comparison."""
    if len(support_records) != 8:
        raise ValueError(f"causal probe requires exactly 8 support records, got {len(support_records)}")
    try:
        configure_deterministic_runtime()
        if method == "gradient":
            trainer: Any = LatentCoreTrainer(lr=0.001, device=device)
        else:
            trainer = EggrollTrainer(device=device)
        baseline_objective = _support_objective(trainer, support_records)
        baseline_held_out = evaluate_held_out_metrics(
            trainer,
            held_out_records,
            device=device,
            max_decode_tokens=max_decode_tokens,
        )
        before_parameters = _capture_matrix_parameters(trainer)

        if method == "gradient":
            if not hasattr(trainer, "train_batch"):
                raise ValueError("gradient trainer does not expose an eight-record batch update")  # noqa: TRY301
            trainer.train_batch(support_records)
        else:
            records = tuple(
                AsdivRecord(
                    id=f"causal-{index}",
                    split="train",
                    question=question,
                    target=answer,
                )
                for index, (question, answer) in enumerate(support_records)
            )
            trainer.train_fitness_batch(
                FitnessBatch(records=records, start_position=0, next_position=len(records)),
                optimizer_call_count=1,
            )

        predicted_delta = getattr(trainer, "last_predicted_objective_delta", None)
        if predicted_delta is None:
            raise ValueError("trainer did not expose the signed first-order prediction")  # noqa: TRY301
        post_objective = _support_objective(trainer, support_records)
        observed_delta = post_objective - baseline_objective
        post_held_out = evaluate_held_out_metrics(
            trainer,
            held_out_records,
            device=device,
            max_decode_tokens=max_decode_tokens,
        )
        max_rms = _relative_matrix_rms(before_parameters, trainer)
        direction_status, direction_conditions = classify_causal_result(
            float(predicted_delta),
            observed_delta,
            baseline_objective,
            post_objective,
        )
        safety_status, safety_conditions = validate_causal_safety_checks(
            baseline_held_out.separation_retention,
            post_held_out.separation_retention,
            max_rms,
        )
        if direction_status != "passed":
            status = direction_status
        elif safety_status != "passed":
            status = safety_status
        else:
            status = "passed"
        return CausalProbeResult(
            method=method,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            baseline_objective=baseline_objective,
            predicted_objective_delta=float(predicted_delta),
            observed_objective_delta=observed_delta,
            max_update_relative_matrix_rms=max_rms,
            baseline_separation=baseline_held_out.separation_retention,
            post_update_separation=post_held_out.separation_retention,
            failed_conditions=(*direction_conditions, *safety_conditions),
        )
    except Exception as error:  # noqa: BLE001 - an incomplete probe is evidence
        return CausalProbeResult(
            method=method,  # type: ignore[arg-type]
            status="inconclusive",
            baseline_objective=None,
            predicted_objective_delta=None,
            observed_objective_delta=None,
            max_update_relative_matrix_rms=None,
            baseline_separation=None,
            post_update_separation=None,
            failed_conditions=(f"{type(error).__name__}: {error!s}",),
        )


def _arm_result(
    arm: MethodArm,
    *,
    overfit_status: str,
    causal_status: str | None,
) -> ArmResult:
    if not arm.evaluations:
        raise ValueError(f"{arm.arm_kind} arm produced no evaluation checkpoints")
    baseline_evaluation = arm.evaluations[0]
    final_evaluation = arm.evaluations[-1]
    baseline = ArmCheckpoint(
        consumed_examples=baseline_evaluation.example_count,
        optimizer_call_count=baseline_evaluation.optimizer_call_count,
        metrics=baseline_evaluation.metrics,
        max_update_relative_matrix_rms=baseline_evaluation.max_update_relative_matrix_rms,
        elapsed_seconds=baseline_evaluation.elapsed_seconds,
        eta_seconds=baseline_evaluation.eta_seconds,
        training_record_identifiers=baseline_evaluation.training_record_identifiers,
        held_out_record_identifiers=baseline_evaluation.held_out_record_identifiers,
    )
    checkpoints = tuple(
        ArmCheckpoint(
            consumed_examples=evaluation.example_count,
            optimizer_call_count=evaluation.optimizer_call_count,
            metrics=evaluation.metrics,
            max_update_relative_matrix_rms=evaluation.max_update_relative_matrix_rms,
            elapsed_seconds=evaluation.elapsed_seconds,
            eta_seconds=evaluation.eta_seconds,
            training_record_identifiers=evaluation.training_record_identifiers,
            held_out_record_identifiers=evaluation.held_out_record_identifiers,
        )
        for evaluation in arm.evaluations[1:]
    )
    if arm.arm_kind == "no_update":
        failed_conditions = (arm.final_stop_reason,) if arm.final_stop_reason else ()
        return ArmResult(
            arm=arm.arm_kind,
            method=None,
            status="inconclusive" if arm.final_status == "stopped" else "passed",
            baseline_checkpoint=baseline,
            checkpoints=checkpoints,
            recalibration_eligible=False,
            failed_conditions=failed_conditions,
        )

    method: MethodName = "gradient" if arm.arm_kind == "gradient_only" else "eggroll"
    if arm.final_status == "stopped":
        status = "inconclusive"
        conditions = (f"arm_stopped: {arm.final_stop_reason or 'unknown'}",)
    elif causal_status != "passed":
        status = "inconclusive"
        conditions = ("causal_probe_not_passed",)
    else:
        status, conditions, _ = classify_method_status(
            baseline,
            ArmCheckpoint(
                consumed_examples=final_evaluation.example_count,
                optimizer_call_count=final_evaluation.optimizer_call_count,
                metrics=final_evaluation.metrics,
                max_update_relative_matrix_rms=final_evaluation.max_update_relative_matrix_rms,
                elapsed_seconds=final_evaluation.elapsed_seconds,
                eta_seconds=final_evaluation.eta_seconds,
                training_record_identifiers=final_evaluation.training_record_identifiers,
                held_out_record_identifiers=final_evaluation.held_out_record_identifiers,
            ),
            method,
            causal_status=causal_status,
        )
    eligible, eligibility_conditions = compute_recalibration_eligibility(
        method,
        overfit_status,
        status,
        causal_status,
    )
    if arm.final_stop_reason and f"arm_stopped: {arm.final_stop_reason}" not in conditions:
        conditions = (*conditions, f"arm_stopped: {arm.final_stop_reason}")
    conditions = (*conditions, *eligibility_conditions)
    return ArmResult(
        arm=arm.arm_kind,
        method=method,
        status=status,
        baseline_checkpoint=baseline,
        checkpoints=checkpoints,
        recalibration_eligible=eligible,
        failed_conditions=conditions,
    )


def _build_report(
    *,
    implementation: Any,
    report_digest: str,
    stability_configuration: dict[str, Any],
    overfit_ids: tuple[tuple[str, str], ...],
    training_ids: tuple[tuple[str, str], ...],
    held_out_ids: tuple[tuple[str, str], ...],
    initial_state_digest: str,
    overfit_probe: OverfitProbeResult,
    causal_probes: tuple[CausalProbeResult, ...],
    arms: tuple[ArmResult, ...],
    elapsed_seconds: float,
    failed_conditions: tuple[str, ...] = (),
) -> TrainabilityReport:
    configuration = TrainabilityConfiguration(
        asset_identity=TrainabilityAssetIdentity(
            stability_report_digest=report_digest,
            held_out_record_identifiers=held_out_ids,
            training_record_identifiers_overfit=overfit_ids,
            training_record_identifiers_32=training_ids,
        ),
        implementation=implementation,
        stability_configuration=stability_configuration,
    )
    asset_digest = hashlib.sha256(canonical_json_bytes(configuration.asset_identity.to_dict())).hexdigest()
    method_statuses: dict[MethodName, str] = {arm.method: arm.status for arm in arms if arm.method is not None}
    method_evidence: dict[MethodName, ArmResult] = {arm.method: arm for arm in arms if arm.method is not None}
    overall_status, overall_conditions = classify_overall_trainability(
        overfit_probe.status,
        method_statuses,
        method_evidence=method_evidence,
    )
    no_update_arm = next((arm for arm in arms if arm.arm == "no_update"), None)
    if no_update_arm is not None and no_update_arm.failed_conditions:
        overall_status = "inconclusive"
        overall_conditions = (*overall_conditions, "control_drift")
    combined_conditions = tuple(dict.fromkeys((*overall_conditions, *failed_conditions)))
    observed_eta = next(
        (
            checkpoint.eta_seconds
            for arm in reversed(arms)
            for checkpoint in reversed(arm.checkpoints)
            if checkpoint.eta_seconds is not None
        ),
        None,
    )
    return TrainabilityReport(
        schema_version=TRAINABILITY_SCHEMA_VERSION,
        configuration=configuration,
        asset_identity_digest=asset_digest,
        initial_state_digest=initial_state_digest,
        overall_status=overall_status,
        overfit_probe=overfit_probe,
        causal_probes=causal_probes,
        arms=arms,
        elapsed_seconds=max(0.0, elapsed_seconds),
        eta_seconds=observed_eta,
        failed_conditions=combined_conditions,
    )


def _run_investigation(
    args: argparse.Namespace,
    progress_writer: TrainabilityProgressWriter,
) -> tuple[str, list[str]]:
    """Run every bounded probe and attach its typed report to the writer."""
    repository_root = Path(__file__).resolve().parents[1]
    implementation = getattr(args, "implementation", None)
    if implementation is None:
        implementation = canonical_trainability_implementation_identity(repository_root)
    expected_held_out_ids = tuple((record_id, "") for record_id in STAGE0_HELD_OUT_ITEM_IDS)
    load_and_validate_stability_report(
        args.stability_report,
        implementation,
        expected_held_out_ids,
        require_complete=True,
    )
    dataset = load_stage0_dataset()
    overfit_records = dataset.training_records(mode="eggroll", epoch=1)[:1]
    training_records = dataset.training_records(mode="eggroll", epoch=1)[:32]
    held_out_records = dataset.held_out_records()[:64]
    overfit_ids, training_ids, held_out_ids = build_trainability_record_selections(dataset)
    report_digest, stability_configuration = load_and_validate_stability_report(
        args.stability_report,
        implementation,
        held_out_ids,
        require_complete=True,
    )
    if len(overfit_records) != 1 or len(training_records) != 32 or len(held_out_records) != 64:
        raise ValueError("Stage 0 trainability selections do not match the fixed 1/32/64 contract")

    initial_state_digest: str | None = None

    def trainer_factory(learning_rate: float, device: str) -> Any:
        nonlocal initial_state_digest
        trainer = LatentCoreTrainer(lr=learning_rate, device=device)
        if initial_state_digest is None:
            initial_state_digest = _state_digest(trainer)
        return trainer

    attempts: list[OverfitAttempt] = []
    question, answer = overfit_records[0].question, overfit_records[0].target
    for learning_rate in OVERFIT_LEARNING_RATES:
        _emit(progress_writer, "overfit_attempt_start", f"overfit:{learning_rate}")
        raw = run_overfit_attempt(
            question,
            answer,
            learning_rate,
            device=args.device,
            checkpoint_steps=OVERFIT_CHECKPOINTS,
            trainer_factory=trainer_factory,
            metrics_evaluator=lambda trainer: _overfit_metrics(
                trainer,
                question,
                answer,
                device=args.device,
                max_decode_tokens=args.max_decode_tokens,
            ),
        )
        typed_attempt = _typed_overfit_attempt(raw)
        attempts.append(typed_attempt)
        if typed_attempt.status in ("passed", "non_finite"):
            _emit(
                progress_writer,
                "overfit_stop",
                f"overfit:{learning_rate}",
                optimizer_call_count=(typed_attempt.checkpoints[-1][0] if typed_attempt.checkpoints else 0),
                payload={
                    "status": typed_attempt.status,
                    "reason": typed_attempt.failed_reason or "overfit criterion passed",
                },
            )
        for step, metrics in typed_attempt.checkpoints:
            _emit(
                progress_writer,
                "overfit_checkpoint",
                f"overfit:{learning_rate}",
                optimizer_call_count=step,
                payload={"metrics": metrics.to_dict() if metrics else None},
            )
        _emit(
            progress_writer,
            "overfit_attempt_complete",
            f"overfit:{learning_rate}",
            optimizer_call_count=(typed_attempt.checkpoints[-1][0] if typed_attempt.checkpoints else 0),
            payload=typed_attempt.to_dict(),
        )

    overfit_probe = classify_overfit_probe(attempts)
    _emit(progress_writer, "overfit_attempt_complete", "overfit", payload=overfit_probe.to_dict())
    if initial_state_digest is None:
        raise ValueError("overfit probe did not construct a fresh trainer")

    support_records = _record_pairs(training_records[:8])
    held_out_pairs = _record_pairs(held_out_records)
    causal_probes: list[CausalProbeResult] = []
    arms: list[ArmResult] = []
    if overfit_probe.status == "passed":
        for method in ("gradient", "eggroll"):
            _emit(progress_writer, "causal_probe_start", method)
            causal = _run_causal_probe(
                method,
                support_records,
                held_out_pairs,
                device=args.device,
                max_decode_tokens=args.max_decode_tokens,
            )
            causal_probes.append(causal)
            _emit(progress_writer, "causal_probe_complete", method, payload=causal.to_dict())

        manifest = FreshStateManifest(
            training_records=_record_pairs(training_records),
            held_out_records=held_out_pairs,
            training_record_identifiers=training_ids,
            held_out_record_identifiers=held_out_ids,
        )
        causal_by_method = {probe.method: probe.status for probe in causal_probes}
        arm_runners = (
            ("no_update", run_no_update_arm),
            ("gradient_only", run_gradient_only_arm),
            ("eggroll_only", run_eggroll_only_arm),
        )
        for arm_name, runner in arm_runners:
            _emit(progress_writer, "arm_start", arm_name)
            arm = runner(
                manifest,
                device=args.device,
                progress_observer=lambda consumed, calls, payload, arm_label=arm_name: _emit(
                    progress_writer,
                    "arm_checkpoint",
                    arm_label,
                    consumed_examples=consumed,
                    optimizer_call_count=calls,
                    elapsed_seconds=payload.get("elapsed_seconds"),
                    eta_seconds=payload.get("eta_seconds"),
                    payload=payload,
                ),
            )
            for evaluation in arm.evaluations:
                _emit(
                    progress_writer,
                    "arm_checkpoint",
                    arm_name,
                    consumed_examples=evaluation.examples_consumed,
                    optimizer_call_count=evaluation.optimizer_call_count,
                    elapsed_seconds=evaluation.elapsed_seconds,
                    eta_seconds=evaluation.eta_seconds,
                    payload=evaluation.to_dict(),
                )
            method = "gradient" if arm_name == "gradient_only" else "eggroll" if arm_name == "eggroll_only" else None
            arm_result = _arm_result(
                arm,
                overfit_status=overfit_probe.status,
                causal_status=causal_by_method.get(method) if method else None,
            )
            arms.append(arm_result)
            if arm.final_status == "stopped":
                _emit(
                    progress_writer,
                    "arm_stop",
                    arm_name,
                    consumed_examples=arm.evaluations[-1].examples_consumed if arm.evaluations else 0,
                    optimizer_call_count=arm.evaluations[-1].optimizer_call_count if arm.evaluations else 0,
                    payload={"reason": arm.final_stop_reason},
                )
            _emit(progress_writer, "arm_complete", arm_name, payload=arm_result.to_dict())

    report = _build_report(
        implementation=implementation,
        report_digest=report_digest,
        stability_configuration=stability_configuration,
        overfit_ids=overfit_ids,
        training_ids=training_ids,
        held_out_ids=held_out_ids,
        initial_state_digest=initial_state_digest,
        overfit_probe=overfit_probe,
        causal_probes=tuple(causal_probes),
        arms=tuple(arms),
        elapsed_seconds=max(0.0, time.monotonic() - progress_writer.started_at),
        failed_conditions=tuple(overfit_probe.failed_conditions),
    )
    progress_writer.final_report = report
    _emit(
        progress_writer,
        "classification",
        "overall",
        payload={
            "overall_status": report.overall_status,
            "failed_conditions": list(report.failed_conditions),
        },
    )
    return report.overall_status, list(report.failed_conditions)


def _write_report_to_file(report: TrainabilityReport, output_path: Path) -> None:
    """Write one canonical report without replacing an existing path."""
    _write_exclusive_text(output_path, report.canonical_json())


def _write_exclusive_text(output_path: Path, content: str) -> None:
    """Create and flush one output file, refusing replacement at the write boundary."""
    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(content)
        output.flush()


def _write_final_report(
    args: argparse.Namespace,
    status: str,
    failed_conditions: list[str],
    elapsed_seconds: float,
    report: TrainabilityReport | None = None,
) -> int:
    """Write the typed report produced by the investigation."""
    try:
        if report is not None:
            _write_report_to_file(
                replace(report, elapsed_seconds=max(0.0, elapsed_seconds)),
                args.final_output,
            )
        else:
            # Infrastructure failures happen before a compatible typed report exists.
            report_dict = {
                "schema_version": TRAINABILITY_SCHEMA_VERSION,
                "overall_status": status,
                "elapsed_seconds": max(0.0, elapsed_seconds),
                "failed_conditions": failed_conditions,
            }
            _write_exclusive_text(args.final_output, json.dumps(report_dict, indent=2, sort_keys=True))
        return _compute_exit_code(status)
    except OSError as error:
        print(f"Failed to write report: {error}", file=sys.stderr, flush=True)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded trainability investigation."""
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    progress_path = args.progress_output or args.final_output.with_suffix(".jsonl")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    configure_deterministic_runtime()
    repository_root = Path(__file__).resolve().parents[1]
    start_time = time.monotonic()
    try:
        args.implementation = canonical_trainability_implementation_identity(repository_root)
    except ValueError as error:
        print(f"implementation identity error: {error}", file=sys.stderr, flush=True)
        return _write_final_report(args, "inconclusive", [f"identity_error: {error!s}"], time.monotonic() - start_time)

    report: TrainabilityReport | None = None
    try:
        with TrainabilityProgressWriter(progress_path) as progress:
            status, failed_conditions = _run_investigation(args, progress)
            report = progress.final_report
    except (OSError, ValueError, TypeError) as error:
        print(f"investigation error: {error}", file=sys.stderr, flush=True)
        return _write_final_report(
            args,
            "inconclusive",
            [f"investigation_exception: {error!s}"],
            time.monotonic() - start_time,
        )

    return _write_final_report(
        args,
        status,
        failed_conditions,
        time.monotonic() - start_time,
        report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
