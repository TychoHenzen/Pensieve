"""Gate 8.4: cycling sweep on FashionMNIST alongside MNIST.

Presents class-incremental Split-classify streams under several
"cycling" configurations: the same total number of examples per task,
split across different numbers of re-exposure cycles (1 cycle is the
original single-pass baseline; more cycles interleave smaller slices of
each task more often). Runs each configuration on both `mnist` and
`fashion-mnist` so the comparison is not an artifact of one dataset's
class boundaries. Final-phase accuracy, averaged across tasks, is
recorded per method per cycle per seed, so retention across cycles can
be read directly from the saved JSON.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from eval.baselines.naive import NaiveBaseline
from eval.baselines.replay import ReplayBaseline
from eval.stream.config import StreamConfig
from eval.stream.events import Boundary, BoundaryKind, Observe, Probe
from eval.stream.generators.split_classify import SplitClassifyGenerator
from eval.subject import Subject

DEFAULT_OUTPUT = Path("gate_results/cycling_sweep.json")
DEFAULT_DATASETS = ["mnist", "fashion-mnist"]
DEFAULT_METHODS = ["naive", "replay"]

INPUT_DIM = 784
OUTPUT_DIM = 10
NUM_TASKS = 5
CLASSES_PER_TASK = 2
PROBES_PER_TASK = 200
TOTAL_EXAMPLES_PER_TASK = 2000
DEFAULT_SEEDS = [0, 1, 2]


@dataclass(frozen=True)
class CycleConfig:
    name: str
    examples_per_task: int
    num_cycles: int
    train_iterations: int


DEFAULT_CYCLE_CONFIGS = [
    CycleConfig("seq", TOTAL_EXAMPLES_PER_TASK, 1, 2000),
    CycleConfig("5-cycle", TOTAL_EXAMPLES_PER_TASK // 5, 5, 400),
    CycleConfig("20-cycle", TOTAL_EXAMPLES_PER_TASK // 20, 20, 100),
]


def _build_subject(method: str, device: str, train_iterations: int) -> Subject:
    if method == "naive":
        return NaiveBaseline(
            INPUT_DIM,
            OUTPUT_DIM,
            2,
            400,
            0.001,
            train_iterations=train_iterations,
            device=device,
        )
    if method == "replay":
        return ReplayBaseline(
            INPUT_DIM,
            OUTPUT_DIM,
            2,
            400,
            0.001,
            latent_dim=100,
            train_iterations=train_iterations,
            device=device,
            pixel_mode=True,
        )
    raise ValueError(f"unknown method: {method}")


def _stream_config(examples_per_task: int, dataset: str, data_dir: str) -> StreamConfig:
    return StreamConfig(
        generator="split-classify",
        params={
            "num_tasks": NUM_TASKS,
            "classes_per_task": CLASSES_PER_TASK,
            "examples_per_task": examples_per_task,
            "probes_per_task": PROBES_PER_TASK,
            "data_source": dataset,
            "data_dir": data_dir,
        },
    )


def _run_one_cycle(subject: Subject, stream: list) -> dict[int, float]:
    """Drive `subject` through one cycle's stream and return per-task accuracy for the final phase."""
    phase = 0
    phase_probes: list[tuple[str, str, str]] = []

    for item in stream:
        event = item.event
        if isinstance(event, Observe):
            subject.observe(event)
        elif isinstance(event, Boundary):
            subject.observe(event)
            if event.kind != BoundaryKind.TASK_TRAINED:
                phase += 1
                phase_probes = []
        elif isinstance(event, Probe):
            answer = subject.answer(event)
            truth = str(item.truth.answer) if item.truth else ""
            phase_probes.append((event.task_id, truth, answer))

    task_correct: dict[int, int] = defaultdict(int)
    task_total: dict[int, int] = defaultdict(int)
    for task_id_str, truth, answer in phase_probes:
        task_index = int(task_id_str.replace("task", ""))
        task_total[task_index] += 1
        if answer == truth:
            task_correct[task_index] += 1
    return {task_index: task_correct[task_index] / task_total[task_index] for task_index in task_total}


def _phase_avg(per_task: dict[int, float]) -> float:
    values = list(per_task.values())
    return sum(values) / len(values) if values else 0.0


def _run_config_seed(
    cfg: CycleConfig,
    dataset: str,
    method: str,
    seed: int,
    device: str,
    data_dir: str,
) -> list[float]:
    """Run all of `cfg`'s cycles for one (method, dataset, seed); returns per-cycle final accuracy."""
    random.seed(seed)
    torch.manual_seed(seed)

    subject = _build_subject(method, device, cfg.train_iterations)
    stream_config = _stream_config(cfg.examples_per_task, dataset, data_dir)
    generator = SplitClassifyGenerator()

    cycle_accuracies: list[float] = []
    for _ in range(cfg.num_cycles):
        stream = list(generator.generate(stream_config, seed=random.getrandbits(32)))
        per_task = _run_one_cycle(subject, stream)
        cycle_accuracies.append(_phase_avg(per_task))
    return cycle_accuracies


def run_sweep(
    datasets: list[str],
    methods: list[str],
    cycle_configs: list[CycleConfig],
    seeds: list[int],
    device: str,
    data_dir: str,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for dataset in datasets:
        results[dataset] = {}
        for cfg in cycle_configs:
            results[dataset][cfg.name] = {}
            for method in methods:
                final_accuracies: list[float] = []
                trajectories: list[list[float]] = []
                for seed in seeds:
                    cycle_accuracies = _run_config_seed(
                        cfg,
                        dataset,
                        method,
                        seed,
                        device,
                        data_dir,
                    )
                    trajectories.append(cycle_accuracies)
                    final_accuracies.append(cycle_accuracies[-1])
                    print(
                        f"dataset={dataset} config={cfg.name} method={method} seed={seed} "
                        f"final_accuracy={cycle_accuracies[-1]:.4f}"
                    )
                mean = statistics.mean(final_accuracies) if final_accuracies else 0.0
                std = statistics.stdev(final_accuracies) if len(final_accuracies) > 1 else 0.0
                results[dataset][cfg.name][method] = {
                    "num_cycles": cfg.num_cycles,
                    "examples_per_task": cfg.examples_per_task,
                    "seeds": seeds,
                    "trajectories": trajectories,
                    "final_accuracies": final_accuracies,
                    "mean": mean,
                    "std": std,
                }
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate 8.4: cycling sweep on FashionMNIST alongside MNIST.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    datasets = [token for token in args.datasets.split(",") if token.strip() != ""]
    methods = [token for token in args.methods.split(",") if token.strip() != ""]
    seeds = [int(token) for token in args.seeds.split(",") if token.strip() != ""]

    results = run_sweep(datasets, methods, DEFAULT_CYCLE_CONFIGS, seeds, args.device, args.data_dir)

    report = {
        "datasets": datasets,
        "methods": methods,
        "cycle_configs": [cfg.__dict__ for cfg in DEFAULT_CYCLE_CONFIGS],
        "seeds": seeds,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
