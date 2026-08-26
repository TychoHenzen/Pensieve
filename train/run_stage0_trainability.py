"""Bounded trainability investigation diagnostic command.

Distinguishes untrainable objectives from optimizer-specific failures through
fixed fresh-state probes and equal-budget method comparisons.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import sys

from train.stage0_trainability import (
    canonical_trainability_implementation_identity,
)
from train.standalone_checkpoint import configure_deterministic_runtime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded trainability investigation for Stage 0."
    )
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


def _validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments before any file I/O or model loading."""
    if not args.stability_report.exists():
        raise FileNotFoundError(
            f"stability report not found: {args.stability_report}"
        )

    progress_path = args.progress_output or args.final_output.with_suffix(".jsonl")

    output_identity = os.path.normcase(str(args.final_output.resolve()))
    progress_identity = os.path.normcase(str(progress_path.resolve()))
    if output_identity == progress_identity:
        raise ValueError(
            "--final-output and --progress-output must use distinct paths"
        )

    if args.final_output.exists():
        raise FileExistsError(
            f"refusing to overwrite trainability output: {args.final_output}"
        )

    if progress_path.exists():
        raise FileExistsError(
            f"refusing to overwrite trainability progress: {progress_path}"
        )

    try:
        report_data = json.loads(args.stability_report.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"stability report malformed or unreadable: {e}") from e

    if not isinstance(report_data, dict):
        raise ValueError("stability report root must be an object")

    status = report_data.get("status")
    if status == "passed":
        raise ValueError(
            "stability report must have status=failed (passing reports cannot start investigation)"
        )
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

    try:
        canonical_trainability_implementation_identity(repository_root)
    except ValueError as e:
        print(f"implementation identity error: {e}", file=sys.stderr, flush=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
