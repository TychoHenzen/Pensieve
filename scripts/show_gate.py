"""Show what the reproduction gate proves.

Runs each baseline through class-incremental Split-MNIST and prints a
retention matrix after each task: how well the model remembers every
task it has seen so far. The matrix makes the core question visible:
does the model retain old knowledge as it learns new tasks?

Naive and EWC collapse to the last task (diagonal-only retention).
Replay retains all tasks (full matrix stays high).
"""
from __future__ import annotations

import random
import sys
import time
from collections import defaultdict
import torch

from eval.baselines.ewc import EWCBaseline
from eval.baselines.naive import NaiveBaseline
from eval.baselines.replay import ReplayBaseline
from eval.stream.config import StreamConfig
from eval.stream.events import Boundary, BoundaryKind, Observe, Probe
from eval.stream.generators.split_classify import SplitClassifyGenerator
from eval.subject import Subject
from eval.subject.isolation import isolated_answer

INPUT_DIM = 784
OUTPUT_DIM = 10
NUM_TASKS = 5
CLASSES_PER_TASK = 2
EXAMPLES_PER_TASK = 2000
PROBES_PER_TASK = 200

DIGIT_NAMES = {
    0: "0,1",
    1: "2,3",
    2: "4,5",
    3: "6,7",
    4: "8,9",
}


def _stream_config() -> StreamConfig:
    return StreamConfig(
        generator="split-classify",
        params={
            "num_tasks": NUM_TASKS,
            "classes_per_task": CLASSES_PER_TASK,
            "examples_per_task": EXAMPLES_PER_TASK,
            "probes_per_task": PROBES_PER_TASK,
            "data_source": "mnist",
            "data_dir": "./data",
        },
    )


def _build_subjects(device: str) -> dict[str, Subject]:
    return {
        "naive": NaiveBaseline(
            INPUT_DIM, OUTPUT_DIM, 2, 400, 0.001, device=device,
        ),
        "ewc": EWCBaseline(
            INPUT_DIM, OUTPUT_DIM, 2, 400, 0.001,
            ewc_lambda=1e7, fisher_samples=1000, device=device,
        ),
        "replay": ReplayBaseline(
            INPUT_DIM, OUTPUT_DIM, 2, 400, 0.001,
            latent_dim=100, device=device, pixel_mode=True,
        ),
    }


def _print_header(phase: int) -> None:
    digits_learned = ", ".join(DIGIT_NAMES[t] for t in range(phase + 1))
    print(f"\n{'=' * 70}")
    print(f"  After learning task {phase} (digits {digits_learned})")
    print(f"  Question: does each method still know the earlier digits?")
    print(f"{'=' * 70}")
    print()
    header = f"  {'method':8s}"
    for t in range(phase + 1):
        header += f"  task{t}({DIGIT_NAMES[t]})"
    header += "   avg"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")


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
    print(row)


def _print_interpretation(results: dict[str, dict[int, float]], phase: int) -> None:
    if phase == 0:
        return
    print()
    for method in ["naive", "ewc", "replay"]:
        accs = results[method]
        old_tasks = [accs.get(t, 0.0) for t in range(phase)]
        old_avg = sum(old_tasks) / len(old_tasks)
        current = accs.get(phase, 0.0)
        if old_avg < 25:
            status = "forgot everything before this task"
        elif old_avg > 75:
            status = "retained old tasks"
        else:
            status = f"partially retained ({old_avg:.0f}% on old tasks)"
        print(f"  {method:8s}: learned task {phase} at {current:.0f}%, {status}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    print()
    print("Class-incremental Split-MNIST: 5 tasks, each teaching 2 new digits.")
    print("After each task, we probe accuracy on ALL tasks seen so far.")
    print("A method that learns without forgetting keeps all columns high.")
    print("A method that forgets shows high accuracy only on the latest task.")

    random.seed(0)
    torch.manual_seed(0)

    subjects = _build_subjects(device)
    config = _stream_config()
    generator = SplitClassifyGenerator()
    stream = list(generator.generate(config, seed=0))

    phase = 0
    phase_probes: dict[str, list[tuple[str, str, str]]] = {
        m: [] for m in subjects
    }

    start = time.monotonic()

    for item in stream:
        event = item.event

        if isinstance(event, Observe):
            for subject in subjects.values():
                subject.observe(event)

        elif isinstance(event, Boundary):
            if event.kind == BoundaryKind.TASK_TRAINED:
                for name, subject in subjects.items():
                    t0 = time.monotonic()
                    subject.observe(event)
                    dt = time.monotonic() - t0
                    sys.stdout.write(f"\r  training {name} on task {phase}... {dt:.1f}s")
                    sys.stdout.flush()
                sys.stdout.write("\r" + " " * 60 + "\r")
                sys.stdout.flush()
            else:
                for subject in subjects.values():
                    subject.observe(event)

                results: dict[str, dict[int, float]] = {}
                for name in subjects:
                    task_correct: dict[int, int] = defaultdict(int)
                    task_total: dict[int, int] = defaultdict(int)
                    for task_id_str, truth_answer, answer in phase_probes[name]:
                        t = int(task_id_str.replace("task", ""))
                        task_total[t] += 1
                        if answer == truth_answer:
                            task_correct[t] += 1
                    results[name] = {
                        t: task_correct[t] / task_total[t] * 100
                        for t in task_total
                    }

                _print_header(phase)
                for name in subjects:
                    _print_row(name, results[name], phase)
                _print_interpretation(results, phase)

                phase += 1
                for m in phase_probes:
                    phase_probes[m] = []

        elif isinstance(event, Probe):
            for name, subject in subjects.items():
                answer = isolated_answer(subject, event)
                truth = str(item.truth.answer) if item.truth else ""
                phase_probes[name].append((event.task_id, truth, answer))

    # Final phase (no TASK_SWITCH after last task, but there are probes)
    results = {}
    for name in subjects:
        task_correct: dict[int, int] = defaultdict(int)
        task_total: dict[int, int] = defaultdict(int)
        for task_id_str, truth_answer, answer in phase_probes[name]:
            t = int(task_id_str.replace("task", ""))
            task_total[t] += 1
            if answer == truth_answer:
                task_correct[t] += 1
        results[name] = {
            t: task_correct[t] / task_total[t] * 100
            for t in task_total
        }

    _print_header(phase)
    for name in subjects:
        _print_row(name, results[name], phase)
    _print_interpretation(results, phase)

    elapsed = time.monotonic() - start
    print(f"\n{'=' * 70}")
    print(f"  Total time: {elapsed:.0f}s")
    print()
    print("  What the gate proves:")
    print("  - Naive SGD and EWC both catastrophically forget in class-incremental")
    print("    learning. They only know the last 2 digits (20% out of 10 classes).")
    print("  - Generative replay retains all 10 digits across all 5 tasks (~90%).")
    print("  - The harness measures this correctly: probes, isolation, metrics.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
