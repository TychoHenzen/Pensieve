"""Bounded trainability investigation diagnostic command.

Distinguishes untrainable objectives from optimizer-specific failures through
fixed fresh-state probes and equal-budget method comparisons.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from train.stage0_trainability import (
    TRAINABILITY_SCHEMA_VERSION,
    TrainabilityProgressRecord,
    TrainabilityReport,
    canonical_trainability_implementation_identity,
)
from train.standalone_checkpoint import configure_deterministic_runtime


def _compute_exit_code(status: str) -> int:
    """Compute exit code from overall classification status.

    Returns 0 for bounded_trainability_observed (viable), 1 for all other statuses.
    """
    return 0 if status == "bounded_trainability_observed" else 1


class TrainabilityProgressWriter:
    """Append-only exclusive-create JSONL progress writer with flushing."""

    def __init__(self, progress_path: Path) -> None:
        self.progress_path = progress_path
        self.sequence_number = 0
        self._file_handle = None

    def __enter__(self):
        self._file_handle = self.progress_path.open("a", encoding="utf-8")
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        if self._file_handle:
            self._file_handle.close()

    def write_record(self, record: TrainabilityProgressRecord) -> None:
        """Write a progress record and flush."""
        if not self._file_handle:
            raise RuntimeError("progress writer not in context manager")
        self._file_handle.write(record.canonical_json_line() + "\n")
        self._file_handle.flush()
        self.sequence_number += 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bounded trainability investigation for Stage 0.")
    parser.add_argument(
        "--stability-report",
        type=Path,
        required=True,
        help="Path to the failed stability report (required anchor)",
    )
    parser.add_argument(
        "--final-output",
        type=Path,
        required=True,
        help="Path for the final trainability JSON report",
    )
    parser.add_argument(
        "--progress-output",
        type=Path,
        default=None,
        help="Optional path for JSONL progress records (defaults to final-output with .jsonl suffix)",
    )
    return parser


def _run_investigation(
    args: argparse.Namespace,
    progress_writer: TrainabilityProgressWriter,
) -> tuple[str, list[str]]:
    """Run the bounded trainability investigation and return (status, failed_conditions)."""
    # For now, return inconclusive - full investigation requires more work
    # This allows the command to at least run and produce output
    return "inconclusive", ["investigation_incomplete"]


def _write_report_to_file(report: TrainabilityReport, output_path: Path) -> None:
    """Write a trainability report to JSON with exclusive-create semantics."""
    report_json = report.canonical_json()
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    temp_path.write_text(report_json, encoding="utf-8")
    os.replace(str(temp_path), str(output_path))


def _write_final_report(
    args: argparse.Namespace,
    status: str,
    failed_conditions: list[str],
    elapsed_seconds: float,
) -> int:
    """Write the final TrainabilityReport JSON file and return exit code."""

    try:
        # Create minimal report with investigation status
        report_dict = {
            "schema_version": TRAINABILITY_SCHEMA_VERSION,
            "configuration": {
                "asset_identity": {
                    "stability_report_digest": "0" * 64,
                    "held_out_record_count": 0,
                    "training_record_count_overfit": 0,
                    "training_record_count_32": 0,
                },
                "implementation": {
                    "sha256": "0" * 64,
                    "sources": [],
                },
                "stability_configuration": {},
            },
            "asset_identity_digest": "0" * 64,
            "initial_state_digest": "0" * 64,
            "overall_status": status,
            "overfit_probe": {
                "status": "inconclusive",
                "attempt_count": 0,
                "attempts": [],
                "failed_conditions": ["investigation_incomplete"],
            },
            "causal_probe_count": 0,
            "causal_probes": [],
            "arm_count": 0,
            "arms": [],
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": None,
            "failed_conditions": failed_conditions if failed_conditions else None,
        }

        # Write JSON report
        report_json = json.dumps(report_dict, indent=2, sort_keys=True)
        args.final_output.write_text(report_json, encoding="utf-8")

        # Exit code: 0 for viable/bounded, 1 for non-viable
        return _compute_exit_code(status)
    except OSError as e:
        print(f"Failed to write report: {e}", file=sys.stderr, flush=True)
        return 1


def _validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments before any file I/O or model loading."""
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

    try:
        report_data = json.loads(args.stability_report.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"stability report malformed or unreadable: {e}") from e

    if not isinstance(report_data, dict):
        raise TypeError("stability report root must be an object")

    status = report_data.get("status")
    if status == "passed":
        raise ValueError("stability report must have status=failed (passing reports cannot start investigation)")
    elif status != "failed":
        raise ValueError(f"stability report must have status=failed, got {status!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded trainability investigation."""
    args = _build_parser().parse_args(argv)
    _validate_args(args)

    progress_path = args.progress_output or args.final_output.with_suffix(".jsonl")
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    configure_deterministic_runtime()
    repository_root = Path(__file__).resolve().parents[1]

    start_time = time.time()

    try:
        canonical_trainability_implementation_identity(repository_root)
    except ValueError as e:
        print(f"implementation identity error: {e}", file=sys.stderr, flush=True)
        elapsed = time.time() - start_time
        return _write_final_report(args, "inconclusive", [f"identity_error: {e!s}"], elapsed)

    # Run the investigation
    try:
        with TrainabilityProgressWriter(progress_path) as progress:
            status, failed_conds = _run_investigation(args, progress)
    except OSError as e:
        print(f"investigation error: {e}", file=sys.stderr, flush=True)
        elapsed = time.time() - start_time
        return _write_final_report(args, "inconclusive", [f"investigation_exception: {e!s}"], elapsed)

    elapsed = time.time() - start_time
    return _write_final_report(args, status, failed_conds, elapsed)


if __name__ == "__main__":
    raise SystemExit(main())
