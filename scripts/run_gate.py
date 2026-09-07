"""Reproduction gate: naive, EWC, and generative replay on Split-MNIST.

Runs the three baselines from `eval.baselines` on identical
class-incremental Split-MNIST streams across a set of seeds, computes
final-phase probe accuracy per method per seed, and checks the result
against the pass condition locked in
`openspec/changes/stage-minus-1-remainder/pass_condition.md`. Writes a
JSON report (`gate_report.json`, in `--output-dir`) with every number the
pass condition names, plus a pass/fail verdict, and prints a
human-readable summary to stdout.

Architecture and hyperparameters come from
`openspec/changes/stage-minus-1-remainder/investigation-phase7.md`: a
2x400 ReLU MLP, Adam lr=0.001, EWC lambda=1e7 with 1000 Fisher samples,
and a replay VAE with latent dim 100, on 5 tasks of 2 digits each
(covering all 10 MNIST digits).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from eval.baselines.ewc import EWCBaseline
from eval.baselines.naive import NaiveBaseline
from eval.baselines.replay import ReplayBaseline
from eval.metrics import ProbeLogEntry
from eval.metrics.accuracy import task_accuracy
from eval.run import PROBE_LOG_FILENAME
from eval.run.config import RunConfig
from eval.run.runner import run
from eval.stream.config import StreamConfig
from eval.stream.generators.split_classify import SplitClassifyGenerator
from eval.subject import CostCounters, Subject

# Architecture (investigation-phase7.md "Decisions for implementation").
INPUT_DIM = 784
OUTPUT_DIM = 10
HIDDEN_LAYERS = 2
HIDDEN_UNITS = 400
LR = 0.001
EWC_LAMBDA = 1e7
FISHER_SAMPLES = 1000
REPLAY_LATENT_DIM = 100

# 5 tasks of 2 digits each covers all 10 MNIST digits, matching van de Ven
# & Tolias (2019) Table 4's class-incremental Split-MNIST split.
NUM_TASKS = 5
CLASSES_PER_TASK = 2

EXAMPLES_PER_TASK = 2000
PROBES_PER_TASK = 200

METHODS = ("naive", "ewc", "replay")

# Pass condition Criterion 1: per-method tolerance bands, percent accuracy.
PASS_BANDS: dict[str, tuple[float, float]] = {
    "naive": (14.90, 24.90),
    "ewc": (15.01, 25.01),
    "replay": (80.79, 100.00),
}

# Pass condition Criterion 2: qualitative bands, percent accuracy.
QUALITATIVE_BANDS: dict[str, tuple[str, float]] = {
    "naive": ("below", 25.00),
    "ewc": ("below", 25.00),
    "replay": ("above", 80.00),
}

MIN_SEEDS = 5


def _stream_config(data_dir: str) -> StreamConfig:
    """Return the class-incremental Split-MNIST stream config shared by every method."""
    return StreamConfig(
        generator="split-classify",
        params={
            "num_tasks": NUM_TASKS,
            "classes_per_task": CLASSES_PER_TASK,
            "examples_per_task": EXAMPLES_PER_TASK,
            "probes_per_task": PROBES_PER_TASK,
            "data_source": "mnist",
            "data_dir": data_dir,
        },
    )


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_subject(method: str, device: str) -> Subject:
    """Return a freshly initialized baseline for `method` on `device`."""
    if method == "naive":
        return NaiveBaseline(
            INPUT_DIM,
            OUTPUT_DIM,
            HIDDEN_LAYERS,
            HIDDEN_UNITS,
            LR,
            device=device,
        )
    if method == "ewc":
        return EWCBaseline(
            INPUT_DIM,
            OUTPUT_DIM,
            HIDDEN_LAYERS,
            HIDDEN_UNITS,
            LR,
            ewc_lambda=EWC_LAMBDA,
            fisher_samples=FISHER_SAMPLES,
            device=device,
        )
    if method == "replay":
        return ReplayBaseline(
            INPUT_DIM,
            OUTPUT_DIM,
            HIDDEN_LAYERS,
            HIDDEN_UNITS,
            LR,
            latent_dim=REPLAY_LATENT_DIM,
            device=device,
            pixel_mode=True,
        )
    raise ValueError(f"unknown method: {method}")


def _seed_everything(seed: int, deterministic: bool) -> None:
    """Seed every RNG this run touches, and switch on deterministic CUDA kernels if asked.

    `eval.run.runner.run` seeds Python's `random` module itself from
    `config.seed`, but the baselines also draw from `torch`'s global RNG
    (weight init, EWC's Fisher noise-free but VAE reparameterization
    samples `torch.randn_like`), so `torch.manual_seed` is set here too.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _run_config(method: str, seed: int, run_dir: Path, data_dir: str) -> RunConfig:
    """Return the `RunConfig` for one (method, seed) gate run.

    Checkpoint intervals are set far beyond what any gate run reaches, so
    no run pauses to checkpoint; this script always runs a method/seed
    pair to completion in one process.
    """
    return RunConfig(
        subject=method,
        stream=_stream_config(data_dir),
        seed=seed,
        checkpoint_interval_events=10**9,
        checkpoint_interval_seconds=10**9,
        retention_keep_every=1,
        output_dir=run_dir,
    )


def _read_probe_log(run_dir: Path) -> list[ProbeLogEntry]:
    """Read `run_dir`'s probe log back into `ProbeLogEntry` values."""
    path = run_dir / PROBE_LOG_FILENAME
    entries: list[ProbeLogEntry] = []
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            raw = json.loads(line)
            raw["cost_counters"] = CostCounters(**raw["cost_counters"])
            entries.append(ProbeLogEntry(**raw))
    return entries


def _execute_run(
    method: str,
    seed: int,
    run_dir: Path,
    data_dir: str,
    deterministic: bool,
    device: str = "cpu",
) -> Path:
    """Seed every RNG, build `method`'s subject, and drive it through the gate stream."""
    _seed_everything(seed, deterministic)
    config = _run_config(method, seed, run_dir, data_dir)
    generator = SplitClassifyGenerator()
    stream = generator.generate(config.stream, seed)
    subject = _build_subject(method, device)
    return run(subject, stream, config, run_dir=run_dir)


def _seed_accuracy(run_dir: Path) -> float:
    """Return final-phase probe accuracy as a percentage.

    The paper reports accuracy after all tasks have been trained. The
    generator's probe_id format is `split-classify-{phase}-{task}-{pos}`,
    so the final phase's probes start with `split-classify-{NUM_TASKS-1}-`.
    """
    log = _read_probe_log(run_dir)
    final_prefix = f"split-classify-{NUM_TASKS - 1}-"
    final_phase = [entry for entry in log if entry.probe_id.startswith(final_prefix)]
    return task_accuracy(final_phase).pooled * 100.0


def _check_reproducibility(
    method: str,
    seed: int,
    base_dir: Path,
    data_dir: str,
    deterministic: bool,
    device: str = "cpu",
) -> bool:
    """Run `method` at `seed` twice under deterministic mode and compare probe logs.

    Pass condition Criterion 5 requires this check to run in deterministic
    mode; if `--deterministic` was not passed, this returns `False`
    without running anything, since a non-deterministic run cannot satisfy
    the criterion regardless of whether its logs happen to match.
    """
    if not deterministic:
        return False
    run_a = base_dir / f"repro-{method}-{seed}-a"
    run_b = base_dir / f"repro-{method}-{seed}-b"
    _execute_run(method, seed, run_a, data_dir, deterministic=True, device=device)
    _execute_run(method, seed, run_b, data_dir, deterministic=True, device=device)
    log_a = (run_a / PROBE_LOG_FILENAME).read_text(encoding="utf-8")
    log_b = (run_b / PROBE_LOG_FILENAME).read_text(encoding="utf-8")
    return log_a == log_b


def _evaluate_criteria(summary: dict[str, Any], seeds: list[int], reproducibility: dict[str, bool]) -> dict[str, bool]:
    """Check `summary` and `reproducibility` against pass_condition.md's five criteria."""
    criterion_1 = all(
        PASS_BANDS[method][0] <= summary[method]["mean"] <= PASS_BANDS[method][1] for method in PASS_BANDS
    )
    criterion_2 = all(
        (summary[method]["mean"] < bound) if direction == "below" else (summary[method]["mean"] > bound)
        for method, (direction, bound) in QUALITATIVE_BANDS.items()
    )
    criterion_3 = (
        summary["replay"]["mean"] > summary["ewc"]["mean"] and summary["replay"]["mean"] > summary["naive"]["mean"]
    )
    criterion_4 = len(seeds) >= MIN_SEEDS and all(len(summary[method]["accuracies"]) >= MIN_SEEDS for method in summary)
    criterion_5 = all(reproducibility.values())

    return {
        "criterion_1_tolerance": criterion_1,
        "criterion_2_qualitative": criterion_2,
        "criterion_3_ordering": criterion_3,
        "criterion_4_seed_count": criterion_4,
        "criterion_5_reproducibility": criterion_5,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print()
    print("=== Reproduction gate summary ===")
    for method, stats in report["methods"].items():
        print(
            f"{method:8s} mean={stats['mean']:.2f}%  std={stats['std']:.2f}%  "
            f"total={stats['total_seconds']:.1f}s  "
            f"accuracies={[f'{a:.2f}' for a in stats['accuracies']]}"
        )
    print()
    for name, passed in report["criteria"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print()
    print(f"GATE: {'PASS' if report['pass'] else 'FAIL'}")
    print(f"Report written to {report['report_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Stage -1 reproduction gate: naive, EWC, and generative "
        "replay on class-incremental Split-MNIST across several seeds."
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated list of integer seeds (at least 5 required to pass).",
    )
    parser.add_argument(
        "--output-dir",
        default="gate_results",
        help="Directory to write run outputs and the gate report to.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic CUDA kernels and run the Criterion 5 reproducibility check.",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory MNIST is downloaded to / read from.",
    )
    args = parser.parse_args()

    seeds = [int(token) for token in args.seeds.split(",") if token.strip() != ""]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _detect_device()
    print(f"device={device}")

    per_method_accuracy: dict[str, list[float]] = {method: [] for method in METHODS}
    per_method_time: dict[str, list[float]] = {method: [] for method in METHODS}

    for method in METHODS:
        method_start = time.monotonic()
        for seed in seeds:
            run_dir = output_dir / method / f"seed{seed}"
            seed_start = time.monotonic()
            _execute_run(
                method,
                seed,
                run_dir,
                args.data_dir,
                args.deterministic,
                device,
            )
            seed_elapsed = time.monotonic() - seed_start
            accuracy = _seed_accuracy(run_dir)
            per_method_accuracy[method].append(accuracy)
            per_method_time[method].append(seed_elapsed)
            print(f"[{method}] seed={seed} accuracy={accuracy:.2f}% ({seed_elapsed:.1f}s)")
        method_elapsed = time.monotonic() - method_start
        print(f"[{method}] total={method_elapsed:.1f}s")

    summary: dict[str, Any] = {}
    for method in METHODS:
        accuracies = per_method_accuracy[method]
        times = per_method_time[method]
        mean = statistics.mean(accuracies)
        std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        summary[method] = {
            "seeds": seeds,
            "accuracies": accuracies,
            "mean": mean,
            "std": std,
            "times_seconds": times,
            "total_seconds": sum(times),
        }

    reproducibility = {
        method: _check_reproducibility(
            method,
            seeds[0],
            output_dir,
            args.data_dir,
            args.deterministic,
            device,
        )
        for method in METHODS
    }

    criteria = _evaluate_criteria(summary, seeds, reproducibility)

    report_path = output_dir / "gate_report.json"
    report = {
        "deterministic": args.deterministic,
        "seeds": seeds,
        "methods": summary,
        "reproducibility": reproducibility,
        "criteria": criteria,
        "pass": all(criteria.values()),
        "report_path": str(report_path),
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_summary(report)


if __name__ == "__main__":
    main()
