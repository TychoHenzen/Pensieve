from __future__ import annotations

import json

from eval.run.config import RunConfig
from eval.run.runner import run
from eval.run import PROBE_LOG_FILENAME
from eval.stream.config import StreamConfig
from eval.stream.events import Observe, Probe
from eval.stream.truth import ProbeTruth, StreamItem
from eval.subject.oracles.perfect_memory import PerfectMemoryOracle


def _make_items(count: int) -> list[StreamItem]:
    items: list[StreamItem] = []
    position = 0
    for i in range(count):
        key = f"k{i}"
        value = f"v{i}"
        items.append(
            StreamItem(
                event=Observe(position=position, payload={"key": key, "value": value}),
                truth=None,
            )
        )
        position += 1
        items.append(
            StreamItem(
                event=Probe(
                    position=position,
                    probe_id=f"p{i}",
                    task_id="t",
                    query=key,
                    teaching_position=position - 1,
                ),
                truth=ProbeTruth(answer=value),
            )
        )
        position += 1
    return items


def _make_config(tmp_path, checkpoint_interval_events=100, checkpoint_interval_seconds=3600.0):
    return RunConfig(
        subject="perfect_memory",
        stream=StreamConfig(generator="fake"),
        seed=0,
        checkpoint_interval_events=checkpoint_interval_events,
        checkpoint_interval_seconds=checkpoint_interval_seconds,
        retention_keep_every=1,
        output_dir=tmp_path,
    )


def _read_log(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# covers: eval/run::run processes a stream through a subject and logs probes
def test_run_scores_probes_and_writes_log(tmp_path):
    subject = PerfectMemoryOracle()
    config = _make_config(tmp_path)
    items = _make_items(5)

    run_dir = run(subject, items, config, run_dir=tmp_path / "run1")

    log_path = run_dir / PROBE_LOG_FILENAME
    assert log_path.exists()
    entries = _read_log(log_path)
    assert len(entries) == 5
    assert all(entry["correct"] for entry in entries)
    assert entries[0]["teaching_position"] == 0


# covers: eval/run::run writes checkpoints at every N stream positions
def test_run_writes_checkpoints_at_position_intervals(tmp_path):
    subject = PerfectMemoryOracle()
    config = _make_config(tmp_path, checkpoint_interval_events=100)
    items = _make_items(125)  # 250 events total

    run_dir = run(subject, items, config, run_dir=tmp_path / "run2")

    checkpoint_dir = run_dir / "checkpoints"
    files = sorted(p.name for p in checkpoint_dir.glob("checkpoint_*.pkl"))
    assert "checkpoint_000100.pkl" in files
    assert "checkpoint_000200.pkl" in files


# covers: eval/run::run resumes from a checkpoint and produces the same probe log
def test_run_resume_produces_same_probe_log(tmp_path):
    items = _make_items(60)  # 120 events total

    unbroken_subject = PerfectMemoryOracle()
    unbroken_config = _make_config(tmp_path, checkpoint_interval_events=1000)
    unbroken_dir = run(unbroken_subject, items, unbroken_config, run_dir=tmp_path / "unbroken")
    unbroken_log = _read_log(unbroken_dir / PROBE_LOG_FILENAME)

    killed_subject = PerfectMemoryOracle()
    killed_config = _make_config(tmp_path, checkpoint_interval_events=50)
    killed_dir = tmp_path / "resumed"
    run(killed_subject, items[:50], killed_config, run_dir=killed_dir)

    resumed_subject = PerfectMemoryOracle()
    run(resumed_subject, items, killed_config, run_dir=killed_dir)
    resumed_log = _read_log(killed_dir / PROBE_LOG_FILENAME)

    assert resumed_log == unbroken_log
