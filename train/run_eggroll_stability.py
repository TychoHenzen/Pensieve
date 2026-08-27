"""Public bounded stability command for fresh Stage 0 EGGROLL training."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import torch

from codecs_module.decoder import DEFAULT_MAX_TOKENS
from train.answer_objective import (
    DEFAULT_PROMPT_ALIGNMENT_WEIGHT,
    validate_prompt_alignment_weight,
)
from train.eggroll_stability import (
    BASELINE_PROBLEM_COUNT,
    DEVELOPMENT_EXAMPLE_COUNT,
    StabilityProgressRecord,
    build_stability_asset_identity,
    build_stability_configuration,
    canonical_implementation_identity,
    run_bounded_stability_gate,
)
from train.eggroll_stability_evaluation import evaluate_stability_metrics
from train.eggroll_trainer import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_FITNESS_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_NUM_STEPS,
    DEFAULT_POP_SIZE,
    DEFAULT_RANK,
    DEFAULT_SIGMA,
    DEFAULT_VARIANCE_WEIGHT,
    EggrollTrainer,
    validate_eggroll_config,
)
from train.stage0_data import load_stage0_dataset
from train.standalone_checkpoint import configure_deterministic_runtime, runtime_identity
from workspace.concept_slots import DEFAULT_SLOT_COUNT


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded absolute-health gate for fresh Stage 0 EGGROLL."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument("--slot-count", type=int, default=DEFAULT_SLOT_COUNT)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument(
        "--fitness-batch-size", type=int, default=DEFAULT_FITNESS_BATCH_SIZE
    )
    parser.add_argument("--variance-weight", type=float, default=DEFAULT_VARIANCE_WEIGHT)
    parser.add_argument(
        "--prompt-alignment-weight",
        type=float,
        default=DEFAULT_PROMPT_ALIGNMENT_WEIGHT,
    )
    parser.add_argument("--amp", action="store_true", dest="use_amp")
    parser.add_argument("--max-decode-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    validate_eggroll_config(
        args.pop_size,
        args.sigma,
        args.lr,
        args.rank,
        args.eval_batch_size,
        args.fitness_batch_size,
    )
    for name in ("slot_count", "num_steps", "max_decode_tokens"):
        if isinstance(getattr(args, name), bool) or getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if not math.isfinite(args.variance_weight) or args.variance_weight < 0.0:
        raise ValueError("--variance-weight must be finite and non-negative")
    validate_prompt_alignment_weight(args.prompt_alignment_weight)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"requested CUDA device {args.device!r} is unavailable")


def _load_fresh_records():
    dataset = load_stage0_dataset()
    training_records = dataset.training_records(mode="eggroll", epoch=1)
    held_out_records = dataset.held_out_records()
    if len(training_records) < DEVELOPMENT_EXAMPLE_COUNT:
        raise ValueError(
            f"Stage 0 requires at least {DEVELOPMENT_EXAMPLE_COUNT} train records"
        )
    if len(held_out_records) < BASELINE_PROBLEM_COUNT:
        raise ValueError(
            f"Stage 0 requires at least {BASELINE_PROBLEM_COUNT} validation records"
        )
    selected_train = tuple(training_records[:DEVELOPMENT_EXAMPLE_COUNT])
    selected_held_out = tuple(held_out_records[:BASELINE_PROBLEM_COUNT])
    if any(record.split != "train" for record in selected_train):
        raise ValueError("stability training accepts only train records")
    if any(record.split != "validation" for record in selected_held_out):
        raise ValueError("stability evaluation accepts only validation records")
    return selected_train, selected_held_out


def _build_trainer(args: argparse.Namespace) -> EggrollTrainer:
    return EggrollTrainer(
        slot_count=args.slot_count,
        num_steps=args.num_steps,
        pop_size=args.pop_size,
        sigma=args.sigma,
        lr=args.lr,
        rank=args.rank,
        variance_weight=args.variance_weight,
        prompt_alignment_weight=args.prompt_alignment_weight,
        eval_batch_size=args.eval_batch_size,
        fitness_batch_size=args.fitness_batch_size,
        use_amp=args.use_amp,
        device=args.device,
    )


def _write_final_report(path: Path, canonical_json: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(canonical_json + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    progress_path = args.progress_output or args.output.with_suffix(".jsonl")
    output_identity = os.path.normcase(str(args.output.resolve()))
    progress_identity = os.path.normcase(str(progress_path.resolve()))
    if output_identity == progress_identity:
        raise ValueError("--output and --progress-output must use distinct paths")
    if args.output.exists() or progress_path.exists():
        raise FileExistsError(
            f"refusing to overwrite stability output: {args.output} or {progress_path}"
        )
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    configure_deterministic_runtime()
    repository_root = Path(__file__).resolve().parents[1]
    implementation = canonical_implementation_identity(repository_root)
    training_records, held_out_records = _load_fresh_records()
    trainer = _build_trainer(args)
    configuration = build_stability_configuration(
        trainer,
        asset_identity=build_stability_asset_identity(
            runtime=runtime_identity(),
            held_out_records=held_out_records,
        ),
        implementation=implementation,
        initialization_seed=0,
    )

    def observe(record: StabilityProgressRecord) -> None:
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(record.canonical_json_line())
        checkpoint = record.checkpoint
        print(
            json.dumps(
                {
                    "kind": "eggroll_stability_progress",
                    "consumed_examples": checkpoint.consumed_examples,
                    "optimizer_call_count": checkpoint.optimizer_call_count,
                    "elapsed_seconds": checkpoint.elapsed_seconds,
                    "eta_seconds": checkpoint.eta_seconds,
                },
                sort_keys=True,
                allow_nan=False,
            ),
            file=sys.stderr,
            flush=True,
        )

    started = time.monotonic()
    report = run_bounded_stability_gate(
        trainer,
        training_records,
        held_out_records,
        configuration,
        partial(
            evaluate_stability_metrics,
            device=args.device,
            max_decode_tokens=args.max_decode_tokens,
        ),
        observer=observe,
    )
    _write_final_report(args.output, report.canonical_json())
    summary = {
        "kind": "eggroll_stability_summary",
        "status": report.status,
        "outcome_code": report.outcome_code,
        "consumed_examples": report.checkpoints[-1].consumed_examples,
        "failed_conditions": [failure.metric for failure in report.failed_thresholds],
        "report": str(args.output),
        "progress": str(progress_path),
        "elapsed_seconds": max(0.0, time.monotonic() - started),
        "implementation_sha256": implementation.sha256,
    }
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return report.outcome_code


if __name__ == "__main__":
    raise SystemExit(main())
