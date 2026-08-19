"""Run the full Stage 0 gate: baseline, latent eval, and pass/fail verdict.

Runs token_cot_baseline, latent_eval, and gate_report in sequence with
timestamped logging. Produces gate_results/*.json and prints a final
PASS/FAIL determination.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from eval.gate.gate_report import build_report
from eval.gate.latent_eval import load_subject, run_seed
from eval.gate.token_cot_baseline import run_baseline
from eval.subjects.latent_core import DEFAULT_NUM_STEPS
from workspace.concept_slots import DEFAULT_SLOT_COUNT

DEFAULT_RESULTS_DIR = Path("gate_results")
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
LOG_EVERY = 50


def _ts() -> str:
    t = time.localtime()
    return f"[{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}]"


def _log(msg: str) -> None:
    print(f"{_ts()} {msg}", flush=True)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Stage 0 gate evaluation."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a trained checkpoint from train.run_training.",
    )
    parser.add_argument("--slot-count", type=int, default=DEFAULT_SLOT_COUNT)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--problem-count",
        type=int,
        default=None,
        help="Limit GSM8K test problems (for quick checks). Default uses the full split.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds for latent eval.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def _run_baseline(args: argparse.Namespace) -> dict:
    _log(f"{'=' * 60}")
    _log("  PHASE 1: Token chain-of-thought baseline")
    _log(f"{'=' * 60}")

    results_path = args.results_dir / "token_cot.json"
    if results_path.exists():
        _log(f"found existing baseline at {results_path}, loading it")
        result = json.loads(results_path.read_text(encoding="utf-8"))
        _log(f"baseline accuracy: {result['accuracy']:.4f} "
             f"({result['correct']}/{result['total']})")
        return result

    _log(f"running Pythia-160M token-CoT on GSM8K test split, device={args.device}")
    phase_start = time.monotonic()

    def _on_baseline_problem(
        _idx: int, total: int, _is_correct: bool, correct: int, done: int
    ) -> None:
        if done % LOG_EVERY == 0 or done == total:
            elapsed = time.monotonic() - phase_start
            remaining = (elapsed / done) * (total - done)
            acc = correct / done
            _log(f"  baseline [{done}/{total}] "
                 f"acc={acc:.4f} ({correct}/{done}) "
                 f"ETA {_format_duration(remaining)}")

    result = run_baseline(
        args.problem_count, args.device, on_problem=_on_baseline_problem
    )

    dt = time.monotonic() - phase_start
    _log(f"baseline done ({_format_duration(dt)})")
    _log(f"baseline accuracy: {result['accuracy']:.4f} "
         f"({result['correct']}/{result['total']})")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log(f"saved: {results_path}")

    return result


def _run_latent_eval(args: argparse.Namespace, seeds: list[int]) -> dict:
    _log(f"{'=' * 60}")
    _log("  PHASE 2: Latent reasoning evaluation")
    _log(f"{'=' * 60}")
    _log(f"checkpoint: {args.checkpoint}")
    _log(f"slots={args.slot_count} steps={args.num_steps} seeds={seeds}")

    phase_start = time.monotonic()
    accuracies: list[float] = []

    for i, seed in enumerate(seeds):
        seed_start = time.monotonic()
        _log(f"  seed {seed} ({i + 1}/{len(seeds)})...")

        subject = load_subject(
            args.checkpoint, args.slot_count, args.num_steps, args.device
        )

        def _on_latent_problem(
            _idx: int, _is_correct: bool, correct: int, done: int
        ) -> None:
            if done % LOG_EVERY == 0:
                elapsed = time.monotonic() - seed_start
                acc = correct / done
                _log(f"    seed {seed} [{done}] "
                     f"acc={acc:.4f} ({correct}/{done}) "
                     f"{_format_duration(elapsed)}")

        accuracy = run_seed(
            subject, seed, args.problem_count, on_problem=_on_latent_problem
        )
        accuracies.append(accuracy)

        seed_dt = time.monotonic() - seed_start
        elapsed = time.monotonic() - phase_start
        if i > 0:
            per_seed = elapsed / (i + 1)
            remaining = per_seed * (len(seeds) - i - 1)
            eta = f"ETA {_format_duration(remaining)}"
        else:
            eta = ""
        _log(f"  seed {seed}: accuracy={accuracy:.4f} "
             f"({_format_duration(seed_dt)}) {eta}")

    mean = statistics.mean(accuracies)
    std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0

    dt = time.monotonic() - phase_start
    _log(f"latent eval done ({_format_duration(dt)})")
    _log(f"latent mean accuracy: {mean:.4f} (std={std:.4f})")

    result = {
        "checkpoint": str(args.checkpoint),
        "slot_count": args.slot_count,
        "num_steps": args.num_steps,
        "split": "test",
        "seeds": seeds,
        "accuracies": accuracies,
        "mean": mean,
        "std": std,
    }

    results_path = args.results_dir / "latent_eval.json"
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log(f"saved: {results_path}")

    return result


def _run_verdict(
    args: argparse.Namespace,
    baseline: dict,
    latent: dict,
) -> bool:
    _log(f"{'=' * 60}")
    _log("  PHASE 3: Gate verdict")
    _log(f"{'=' * 60}")

    report = build_report(args.results_dir)

    report_path = args.results_dir / "gate_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    baseline_acc = baseline["accuracy"]
    latent_mean = latent["mean"]
    latent_std = latent["std"]
    delta = latent_mean - baseline_acc

    _log(f"  token-CoT baseline:    {baseline_acc:.4f}")
    _log(f"  latent mean (5 seeds): {latent_mean:.4f} (std={latent_std:.4f})")
    sign = "+" if delta >= 0 else ""
    _log(f"  delta: {sign}{delta:.4f}")

    for name, passed in report["criteria"].items():
        tag = "PASS" if passed else "FAIL"
        _log(f"  {name}: {tag}")

    passed = report["pass"]

    _log(f"{'=' * 60}")
    if passed:
        _log("  GATE: PASS")
        _log("  Latent reasoning meets or exceeds token chain-of-thought.")
    else:
        _log("  GATE: FAIL")
        _log("  Latent reasoning does not meet token chain-of-thought baseline.")
    _log(f"{'=' * 60}")
    _log(f"saved: {report_path}")

    return passed


def main() -> None:
    args = _parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if not args.checkpoint.exists():
        _log(f"checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if len(seeds) < 5:
        _log(f"WARNING: {len(seeds)} seeds given, gate requires at least 5")

    _log(f"device={args.device}")
    _log(f"Stage 0 gate: latent reasoning vs token chain-of-thought on GSM8K")

    gate_start = time.monotonic()

    baseline = _run_baseline(args)
    latent = _run_latent_eval(args, seeds)
    passed = _run_verdict(args, baseline, latent)

    gate_dt = time.monotonic() - gate_start
    _log(f"total gate time: {_format_duration(gate_dt)}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
