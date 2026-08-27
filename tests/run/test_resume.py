"""Thorough kill-and-resume coverage beyond the basic case in test_runner.py.

`test_run_resume_produces_same_probe_log` in test_runner.py covers the
single-kill case with a deterministic subject. This file adds: a subject
whose answers depend on internal random state (ChanceOracle), a run
killed and resumed twice, plain seeded reproducibility across two
independent unbroken runs, and resume against a directory that holds no
checkpoint at all.
"""

from __future__ import annotations

import json

from eval.run import PROBE_LOG_FILENAME
from eval.run.config import RunConfig
from eval.run.runner import run
from eval.stream.config import StreamConfig
from eval.stream.events import Observe, Probe
from eval.stream.truth import ProbeTruth, StreamItem
from eval.subject.oracles.chance import ChanceOracle
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


def _make_config(
    tmp_path,
    subject="perfect_memory",
    checkpoint_interval_events=100,
    checkpoint_interval_seconds=3600.0,
    seed=0,
):
    return RunConfig(
        subject=subject,
        stream=StreamConfig(generator="fake"),
        seed=seed,
        checkpoint_interval_events=checkpoint_interval_events,
        checkpoint_interval_seconds=checkpoint_interval_seconds,
        retention_keep_every=1,
        output_dir=tmp_path,
    )


def _read_log(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# covers: eval/run::run resume with a subject whose answers depend on
# internal random state (ChanceOracle) still reproduces the unbroken log
def test_resume_with_chance_oracle_matches_unbroken_run(tmp_path):
    items = _make_items(30)  # 60 events total

    unbroken_subject = ChanceOracle(seed=7)
    unbroken_config = _make_config(
        tmp_path, subject="chance", checkpoint_interval_events=1000, seed=7
    )
    unbroken_dir = run(unbroken_subject, items, unbroken_config, run_dir=tmp_path / "unbroken")
    unbroken_log = _read_log(unbroken_dir / PROBE_LOG_FILENAME)

    killed_dir = tmp_path / "resumed"
    killed_subject = ChanceOracle(seed=7)
    killed_config = _make_config(
        tmp_path, subject="chance", checkpoint_interval_events=25, seed=7
    )
    run(killed_subject, items[:25], killed_config, run_dir=killed_dir)

    resumed_subject = ChanceOracle(seed=7)
    run(resumed_subject, items, killed_config, run_dir=killed_dir)
    resumed_log = _read_log(killed_dir / PROBE_LOG_FILENAME)

    assert resumed_log == unbroken_log


# covers: eval/run::run a run killed and resumed twice produces the same
# probe log as an unbroken run
def test_run_killed_and_resumed_twice_matches_unbroken_run(tmp_path):
    items = _make_items(50)  # 100 events total

    unbroken_subject = PerfectMemoryOracle()
    unbroken_config = _make_config(tmp_path, checkpoint_interval_events=1000)
    unbroken_dir = run(unbroken_subject, items, unbroken_config, run_dir=tmp_path / "unbroken")
    unbroken_log = _read_log(unbroken_dir / PROBE_LOG_FILENAME)

    resumed_dir = tmp_path / "resumed"

    # Leg 1: run to position 30, checkpointing there.
    leg1_subject = PerfectMemoryOracle()
    leg1_config = _make_config(tmp_path, checkpoint_interval_events=30)
    run(leg1_subject, items[:30], leg1_config, run_dir=resumed_dir)

    # Leg 2: resume, run to position 70, checkpointing there.
    leg2_subject = PerfectMemoryOracle()
    leg2_config = _make_config(tmp_path, checkpoint_interval_events=70)
    run(leg2_subject, items[:70], leg2_config, run_dir=resumed_dir)

    # Leg 3: resume from position 70, run to completion.
    leg3_subject = PerfectMemoryOracle()
    leg3_config = _make_config(tmp_path, checkpoint_interval_events=1000)
    run(leg3_subject, items, leg3_config, run_dir=resumed_dir)

    resumed_log = _read_log(resumed_dir / PROBE_LOG_FILENAME)
    assert resumed_log == unbroken_log


# covers: eval/run::run two independent runs with the same config and
# seed produce identical probe logs
def test_two_runs_with_same_seed_produce_same_probe_log(tmp_path):
    items = _make_items(20)  # 40 events total

    subject_a = PerfectMemoryOracle()
    config_a = _make_config(tmp_path, checkpoint_interval_events=1000, seed=42)
    dir_a = run(subject_a, items, config_a, run_dir=tmp_path / "run_a")

    subject_b = PerfectMemoryOracle()
    config_b = _make_config(tmp_path, checkpoint_interval_events=1000, seed=42)
    dir_b = run(subject_b, items, config_b, run_dir=tmp_path / "run_b")

    assert _read_log(dir_a / PROBE_LOG_FILENAME) == _read_log(dir_b / PROBE_LOG_FILENAME)


# covers: eval/run::run resuming against a directory with no checkpoint
# behaves like a normal run from the start
def test_resume_with_no_checkpoint_behaves_like_fresh_run(tmp_path):
    items = _make_items(10)  # 20 events total

    baseline_subject = PerfectMemoryOracle()
    baseline_config = _make_config(tmp_path, checkpoint_interval_events=1000)
    baseline_dir = run(baseline_subject, items, baseline_config, run_dir=tmp_path / "baseline")
    baseline_log = _read_log(baseline_dir / PROBE_LOG_FILENAME)

    # A run directory that exists but has never been checkpointed.
    fresh_dir = tmp_path / "fresh"
    fresh_dir.mkdir()
    fresh_subject = PerfectMemoryOracle()
    fresh_config = _make_config(tmp_path, checkpoint_interval_events=1000)
    run(fresh_subject, items, fresh_config, run_dir=fresh_dir)
    fresh_log = _read_log(fresh_dir / PROBE_LOG_FILENAME)

    assert fresh_log == baseline_log
    assert len(fresh_log) == 10
