"""Gate machinery smoke test: runner, probe log, and accuracy on a tiny stream.

Drives each of the three gate baselines (naive, EWC, generative replay)
through a tiny synthetic `split-classify` stream (2 tasks, 2 classes per
task, 10 taught examples per task, 2 probes per task, no MNIST data) using
`eval.run.runner.run`, the same entry point `scripts/run_gate.py` calls.
The point is not that any baseline passes the real gate, only that the
run produces a non-empty probe log, that log parses back into
`ProbeLogEntry` objects, and `eval.metrics.accuracy.task_accuracy` reduces
that log to a pooled accuracy in `[0.0, 1.0]`. Small `hidden_units` and
few examples/iterations keep the whole module under ten seconds with no
network access and no GPU.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

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

NUM_TASKS = 2
CLASSES_PER_TASK = 2
INPUT_DIM = CLASSES_PER_TASK
OUTPUT_DIM = NUM_TASKS * CLASSES_PER_TASK
HIDDEN_UNITS = 16
SEED = 0

_STREAM_CONFIG = StreamConfig(
    generator="split-classify",
    params={
        "num_tasks": NUM_TASKS,
        "classes_per_task": CLASSES_PER_TASK,
        "examples_per_task": 10,
        "probes_per_task": 2,
    },
)

BASELINE_FACTORIES: dict[str, Callable[[], Subject]] = {
    "naive": lambda: NaiveBaseline(
        input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_units=HIDDEN_UNITS
    ),
    "ewc": lambda: EWCBaseline(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_units=HIDDEN_UNITS,
        fisher_samples=5,
    ),
    "replay": lambda: ReplayBaseline(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_units=HIDDEN_UNITS,
        train_iterations=10,
    ),
}

BASELINE_NAMES = sorted(BASELINE_FACTORIES)


def _run_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        subject="gate-smoke",
        stream=_STREAM_CONFIG,
        seed=SEED,
        checkpoint_interval_events=10_000,
        checkpoint_interval_seconds=10_000.0,
        retention_keep_every=1,
        output_dir=tmp_path,
    )


def _run_baseline(name: str, run_dir: Path) -> Path:
    """Build baseline `name`, drive it through the tiny stream, return its run dir."""
    baseline = BASELINE_FACTORIES[name]()
    stream = SplitClassifyGenerator().generate(_STREAM_CONFIG, seed=SEED)
    config = _run_config(run_dir)
    return run(baseline, stream, config, run_dir=run_dir)


def _load_probe_log(run_dir: Path) -> list[ProbeLogEntry]:
    """Read `run_dir`'s newline-delimited probe log back into `ProbeLogEntry` objects."""
    probe_log_path = run_dir / PROBE_LOG_FILENAME
    entries: list[ProbeLogEntry] = []
    for line in probe_log_path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        entries.append(
            ProbeLogEntry(
                position=raw["position"],
                probe_id=raw["probe_id"],
                task_id=raw["task_id"],
                teaching_position=raw["teaching_position"],
                correct=raw["correct"],
                cost_counters=CostCounters(**raw["cost_counters"]),
            )
        )
    return entries


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_gate_run_produces_nonempty_probe_log(tmp_path: Path, name: str) -> None:
    run_dir = _run_baseline(name, tmp_path / name)

    probe_log_path = run_dir / PROBE_LOG_FILENAME
    assert probe_log_path.exists()
    assert probe_log_path.stat().st_size > 0


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_gate_probe_log_parses_into_entries(tmp_path: Path, name: str) -> None:
    run_dir = _run_baseline(name, tmp_path / name)

    entries = _load_probe_log(run_dir)

    expected_probe_count = sum(
        1
        for item in SplitClassifyGenerator().generate(_STREAM_CONFIG, seed=SEED)
        if item.truth is not None
    )
    assert len(entries) == expected_probe_count
    assert len(entries) > 0
    for entry in entries:
        assert isinstance(entry, ProbeLogEntry)
        assert isinstance(entry.correct, bool)


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_gate_task_accuracy_pooled_value_is_in_unit_range(tmp_path: Path, name: str) -> None:
    run_dir = _run_baseline(name, tmp_path / name)
    entries = _load_probe_log(run_dir)

    result = task_accuracy(entries)

    assert isinstance(result.pooled, float)
    assert 0.0 <= result.pooled <= 1.0
