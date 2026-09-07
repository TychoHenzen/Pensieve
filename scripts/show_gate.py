"""Cycling experiment: does re-exposure to the task sequence help retention?

Runs each baseline through class-incremental Split-MNIST under multiple
cycling configurations. Each config presents the same total number of
examples (2000 per task) split across different numbers of cycles.
Subjects keep their state between cycles. Three seeds per config give
mean and standard deviation.

Joint training (retrain on everything each boundary) is the ceiling.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import torch

from eval.baselines.ewc import EWCBaseline
from eval.baselines.joint import JointBaseline
from eval.baselines.naive import NaiveBaseline
from eval.baselines.replay import ReplayBaseline
from eval.stream.config import StreamConfig
from eval.stream.events import Boundary, BoundaryKind, Observe, Probe
from eval.stream.generators.split_classify import SplitClassifyGenerator
from eval.subject import Subject

INPUT_DIM = 784
OUTPUT_DIM = 10
NUM_TASKS = 5
CLASSES_PER_TASK = 2
PROBES_PER_TASK = 200
TOTAL_EXAMPLES_PER_TASK = 2000
NUM_SEEDS = 3
METHOD_ORDER = ["naive", "ewc", "replay", "joint"]

TASK_LABELS: dict[str, dict[int, str]] = {
    "mnist": {
        0: "0,1",
        1: "2,3",
        2: "4,5",
        3: "6,7",
        4: "8,9",
    },
    "fashion-mnist": {
        0: "top,trouser",
        1: "pullover,dress",
        2: "coat,sandal",
        3: "shirt,sneaker",
        4: "bag,boot",
    },
}


@dataclass
class ExperimentConfig:
    name: str
    examples_per_task: int
    num_cycles: int
    train_iterations: int
    dataset: str = "mnist"


def _cycling_configs(dataset: str) -> list[ExperimentConfig]:
    return [
        ExperimentConfig("seq", 2000, 1, 2000, dataset),
        ExperimentConfig("2-cycle", 1000, 2, 1000, dataset),
        ExperimentConfig("10-cycle", 200, 10, 200, dataset),
        ExperimentConfig("50-cycle", 40, 50, 50, dataset),
        ExperimentConfig("200-cycle", 10, 200, 50, dataset),
    ]


DATASETS = ["mnist", "fashion-mnist"]


def _stream_config(examples_per_task: int, dataset: str) -> StreamConfig:
    return StreamConfig(
        generator="split-classify",
        params={
            "num_tasks": NUM_TASKS,
            "classes_per_task": CLASSES_PER_TASK,
            "examples_per_task": examples_per_task,
            "probes_per_task": PROBES_PER_TASK,
            "data_source": dataset,
            "data_dir": "./data",
        },
    )


def _build_subjects(device: str, train_iterations: int) -> dict[str, Subject]:
    return {
        "naive": NaiveBaseline(
            INPUT_DIM,
            OUTPUT_DIM,
            2,
            400,
            0.001,
            train_iterations=train_iterations,
            device=device,
        ),
        "ewc": EWCBaseline(
            INPUT_DIM,
            OUTPUT_DIM,
            2,
            400,
            0.001,
            ewc_lambda=1e7,
            fisher_samples=1000,
            train_iterations=train_iterations,
            device=device,
        ),
        "replay": ReplayBaseline(
            INPUT_DIM,
            OUTPUT_DIM,
            2,
            400,
            0.001,
            latent_dim=100,
            train_iterations=train_iterations,
            device=device,
            pixel_mode=True,
        ),
        "joint": JointBaseline(
            INPUT_DIM,
            OUTPUT_DIM,
            2,
            400,
            0.001,
            train_iterations=train_iterations,
            device=device,
        ),
    }


def _checkpoint_cycles(num_cycles: int) -> set[int]:
    """Which cycle indices (0-based) get printed as checkpoints."""
    if num_cycles <= 10:
        return set(range(num_cycles))
    points = {0, num_cycles - 1}
    for frac in (0.25, 0.5, 0.75):
        points.add(round(num_cycles * frac) - 1)
    return points


def _print_header(phase: int, cycle: int, num_cycles: int, dataset: str) -> None:
    labels = TASK_LABELS[dataset]
    learned = ", ".join(labels[t] for t in range(phase + 1))
    _log(f"{'=' * 60}")
    _log(f"  Cycle {cycle + 1}/{num_cycles} - After task {phase} ({learned})")
    _log(f"{'=' * 60}")
    header = f"  {'method':8s}"
    for t in range(phase + 1):
        header += f"  task{t}({labels[t]})"
    header += "   avg"
    _log(header)
    _log(f"  {'-' * (len(header) - 2)}")


def _print_row(method: str, accuracies: dict[int, float], phase: int) -> None:
    row = f"  {method:8s}"
    values = []
    for t in range(phase + 1):
        acc = accuracies.get(t, 0.0)
        values.append(acc)
        if acc >= 80:
            marker = ""
        elif acc <= 25:
            marker = " !"
        else:
            marker = " ~"
        row += f"  {acc:11.1f}%{marker}"
    avg = sum(values) / len(values) if values else 0.0
    row += f"  {avg:5.1f}%"
    _log(row)


def _compute_results(
    subjects: dict[str, Subject],
    phase_probes: dict[str, list[tuple[str, str, str]]],
) -> dict[str, dict[int, float]]:
    results: dict[str, dict[int, float]] = {}
    for name in subjects:
        task_correct: dict[int, int] = defaultdict(int)
        task_total: dict[int, int] = defaultdict(int)
        for task_id_str, truth_answer, answer in phase_probes[name]:
            t = int(task_id_str.replace("task", ""))
            task_total[t] += 1
            if answer == truth_answer:
                task_correct[t] += 1
        results[name] = {t: task_correct[t] / task_total[t] * 100 for t in task_total}
    return results


def _method_avg(results: dict[str, dict[int, float]], name: str) -> float:
    values = list(results[name].values())
    return sum(values) / len(values) if values else 0.0


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


def _run_cycle(
    subjects: dict[str, Subject],
    stream: list,
    cycle: int,
    cfg: ExperimentConfig,
    verbose: bool,
    seed_idx: int = 0,
    seed_start: float = 0.0,
) -> dict[str, float]:
    """Run one cycle. Returns per-method final avg accuracy."""
    phase = 0
    phase_probes: dict[str, list[tuple[str, str, str]]] = {m: [] for m in subjects}
    last_results: dict[str, dict[int, float]] = {}
    cycle_t0 = time.monotonic()
    tag = f"[{cfg.name} s{seed_idx + 1}] cycle {cycle + 1}/{cfg.num_cycles}"

    for item in stream:
        event = item.event

        if isinstance(event, Observe):
            for subject in subjects.values():
                subject.observe(event)

        elif isinstance(event, Boundary):
            if event.kind == BoundaryKind.TASK_TRAINED:
                _log(f"  {tag} task {phase} training...")
                for name, subject in subjects.items():
                    t0 = time.monotonic()
                    subject.observe(event)
                    dt = time.monotonic() - t0
                    _log(f"    {name}: {dt:.1f}s")
                _log(f"  {tag} task {phase} probing...")
            else:
                for subject in subjects.values():
                    subject.observe(event)

                last_results = _compute_results(subjects, phase_probes)
                if verbose:
                    _print_header(phase, cycle, cfg.num_cycles, cfg.dataset)
                    for name in METHOD_ORDER:
                        if name in subjects:
                            _print_row(name, last_results[name], phase)

                phase += 1
                for m in phase_probes:
                    phase_probes[m] = []

        elif isinstance(event, Probe):
            for name, subject in subjects.items():
                answer = subject.answer(event)
                truth = str(item.truth.answer) if item.truth else ""
                phase_probes[name].append((event.task_id, truth, answer))

    last_results = _compute_results(subjects, phase_probes)
    if verbose:
        _print_header(phase, cycle, cfg.num_cycles, cfg.dataset)
        for name in METHOD_ORDER:
            if name in subjects:
                _print_row(name, last_results[name], phase)

    cycle_dt = time.monotonic() - cycle_t0
    avgs = {name: _method_avg(last_results, name) for name in subjects}

    seed_elapsed = time.monotonic() - seed_start
    if cycle > 0:
        per_cycle = seed_elapsed / (cycle + 1)
        remaining = per_cycle * (cfg.num_cycles - cycle - 1)
        eta = f"ETA {_format_duration(remaining)}"
    else:
        eta = ""

    parts = " ".join(f"{n}={avgs[n]:.1f}%" for n in METHOD_ORDER if n in avgs)
    _log(f"  {tag} done ({_format_duration(cycle_dt)}) {parts} {eta}")

    return avgs


def _run_config_seed(
    cfg: ExperimentConfig,
    device: str,
    seed_idx: int,
    verbose: bool,
) -> list[dict[str, float]]:
    """Run all cycles for one config+seed. Returns per-cycle avg accuracies."""
    seed = random.getrandbits(32)
    random.seed(seed)
    torch.manual_seed(seed)

    subjects = _build_subjects(device, cfg.train_iterations)
    stream_config = _stream_config(cfg.examples_per_task, cfg.dataset)
    generator = SplitClassifyGenerator()
    checkpoints = _checkpoint_cycles(cfg.num_cycles)

    cycle_avgs: list[dict[str, float]] = []
    seed_start = time.monotonic()

    for cycle in range(cfg.num_cycles):
        is_checkpoint = cycle in checkpoints
        show_full = verbose and is_checkpoint

        if show_full and cfg.num_cycles > 10:
            _log(f"  --- checkpoint cycle {cycle + 1}/{cfg.num_cycles} ---")

        stream = list(
            generator.generate(
                stream_config,
                seed=random.getrandbits(32),
            )
        )
        avgs = _run_cycle(
            subjects,
            stream,
            cycle,
            cfg,
            show_full,
            seed_idx=seed_idx,
            seed_start=seed_start,
        )
        cycle_avgs.append(avgs)

        if is_checkpoint and not show_full:
            elapsed = time.monotonic() - seed_start
            parts = " ".join(f"{n}={avgs[n]:.1f}%" for n in METHOD_ORDER if n in avgs)
            _log(f"  CHECKPOINT {cycle + 1}/{cfg.num_cycles} ({_format_duration(elapsed)}): {parts}")

    return cycle_avgs


def _print_cycle_comparison(
    cfg: ExperimentConfig,
    cycle_avgs: list[dict[str, float]],
    subjects: list[str],
) -> None:
    """Print first vs last cycle comparison for one seed."""
    checkpoints = sorted(_checkpoint_cycles(cfg.num_cycles))
    _log(f"  Cycle trajectory ({cfg.name}):")
    header = f"  {'method':8s}"
    for cp in checkpoints:
        header += f"  {'c' + str(cp + 1):>7s}"
    header += f"  {'delta':>7s}"
    _log(header)
    _log(f"  {'-' * (len(header) - 2)}")

    for name in subjects:
        row = f"  {name:8s}"
        for cp in checkpoints:
            row += f"  {cycle_avgs[cp][name]:6.1f}%"
        delta = cycle_avgs[-1][name] - cycle_avgs[0][name]
        sign = "+" if delta >= 0 else ""
        row += f"  {sign}{delta:5.1f}%"
        _log(row)


def _print_grand_summary(
    all_results: dict[str, dict[str, list[float]]],
    configs: list[ExperimentConfig],
    dataset: str,
) -> None:
    """Print the final comparison table across all configs and seeds."""
    _log(f"{'#' * 60}")
    _log(f"  GRAND SUMMARY - {dataset}")
    _log(f"{'#' * 60}")

    header = f"  {'config':>10s}  {'method':8s}"
    for i in range(NUM_SEEDS):
        header += f"  {'s' + str(i + 1):>6s}"
    header += f"  {'mean':>6s}  {'std':>5s}"
    _log(header)
    _log(f"  {'-' * (len(header) - 2)}")

    for cfg in configs:
        for method in METHOD_ORDER:
            vals = all_results[cfg.name].get(method, [])
            if not vals:
                continue
            row = f"  {cfg.name:>10s}  {method:8s}"
            for v in vals:
                row += f"  {v:5.1f}%"
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            row += f"  {mean:5.1f}%  {std:4.1f}%"
            _log(row)
        _log("")


def _run_dataset(dataset: str, device: str) -> None:
    configs = _cycling_configs(dataset)

    _log(f"{'=' * 60}")
    _log(f"  DATASET: {dataset}")
    _log(f"  {TOTAL_EXAMPLES_PER_TASK} total examples per task, {NUM_SEEDS} seeds per config.")
    _log(f"  {len(configs)} configs: {', '.join(c.name for c in configs)}")
    _log(f"{'=' * 60}")

    all_results: dict[str, dict[str, list[float]]] = {c.name: {m: [] for m in METHOD_ORDER} for c in configs}

    total_runs = len(configs) * NUM_SEEDS
    run_count = 0
    dataset_start = time.monotonic()

    for cfg_idx, cfg in enumerate(configs):
        _log(f"{'#' * 60}")
        _log(f"  [{dataset}] CONFIG {cfg_idx + 1}/{len(configs)}: {cfg.name}")
        _log(f"  {cfg.examples_per_task} examples/cycle x {cfg.num_cycles} cycles")
        _log(f"  {cfg.train_iterations} train iterations per boundary")
        if run_count > 0:
            elapsed = time.monotonic() - dataset_start
            per_run = elapsed / run_count
            runs_left = total_runs - run_count
            _log(
                f"  dataset progress: {run_count}/{total_runs} seeds done, ETA {_format_duration(per_run * runs_left)}"
            )
        _log(f"{'#' * 60}")

        for seed_idx in range(NUM_SEEDS):
            run_count += 1
            _log(f"  --- seed {seed_idx + 1}/{NUM_SEEDS} (run {run_count}/{total_runs}) ---")

            verbose = (seed_idx == 0) and (cfg.num_cycles <= 10)
            cycle_avgs = _run_config_seed(cfg, device, seed_idx, verbose)

            if cfg.num_cycles > 1:
                _print_cycle_comparison(
                    cfg,
                    cycle_avgs,
                    METHOD_ORDER,
                )

            final = cycle_avgs[-1]
            for method in METHOD_ORDER:
                all_results[cfg.name][method].append(final[method])

            _log(f"  seed {seed_idx + 1} done ({_format_duration(time.monotonic() - dataset_start)} into {dataset})")

    _print_grand_summary(all_results, configs, dataset)

    dataset_time = time.monotonic() - dataset_start
    _log(f"  {dataset} done in {_format_duration(dataset_time)}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"device={device}")
    _log("Cycling experiment: 5 tasks, 2 classes each.")
    _log(f"Datasets: {', '.join(DATASETS)}")
    _log(f"{TOTAL_EXAMPLES_PER_TASK} total examples per task, {NUM_SEEDS} seeds per config.")

    experiment_start = time.monotonic()

    for dataset in DATASETS:
        _run_dataset(dataset, device)

    total_time = time.monotonic() - experiment_start
    _log(f"  Total experiment time: {_format_duration(total_time)}")
    _log("  Reading guide:")
    _log("  - 'seq' is the original single-pass baseline.")
    _log("  - Higher cycle counts approach interleaved training.")
    _log("  - 'joint' retrains on all data each boundary (the ceiling).")
    _log("  - If replay's mean rises with cycle count, cycling helps.")
    _log("  - If EWC's mean drops, Fisher over-consolidation hurts it.")
    _log("  - fashion-mnist is harder: fuzzier class boundaries test replay quality.")
    _log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
