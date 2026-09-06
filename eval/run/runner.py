"""The main run loop: drive a subject through a stream, log probes, checkpoint.

`run` is the one entry point. It processes a `StreamItem` at a time, routes
each event to the subject by kind, scores probes against the truth side
channel, and appends one `ProbeLogEntry` per probe to the run's probe log.
Checkpointing happens on two independent schedules, a stream-position count
and a wall-clock interval, either of which can trigger a save. On start, an
existing checkpoint (if any) restores the subject and RNG state and skips
the stream forward to the same position, so a killed and resumed run
produces the same probe log as an unbroken one.

No metric computation and no retention policy live here; those are
separate concerns layered on top of the probe log and the checkpoint
directory this module writes.
"""

from __future__ import annotations

import dataclasses
import json
import random
import time
from collections.abc import Iterable
from pathlib import Path

from eval.metrics import ProbeLogEntry
from eval.run import CHECKPOINT_DIRNAME, PROBE_LOG_FILENAME
from eval.run.checkpoint import (
    capture_rng_state,
    latest_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from eval.run.config import RunConfig, config_hash
from eval.stream.events import Boundary, Idle, Observe, Probe
from eval.stream.truth import StreamItem
from eval.subject import Subject
from eval.subject.isolation import isolated_answer


def _run_directory(config: RunConfig, run_dir: Path | None) -> Path:
    """Return the directory this run writes to, creating it if needed."""
    directory = run_dir if run_dir is not None else config.output_dir / config_hash(config)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _log_line(entry: ProbeLogEntry) -> str:
    """Return one newline-delimited JSON line for `entry`."""
    return json.dumps(dataclasses.asdict(entry))


def run(
    subject: Subject,
    stream: Iterable[StreamItem],
    config: RunConfig,
    run_dir: Path | None = None,
) -> Path:
    """Drive `subject` through `stream` under `config`, and return the run directory.

    Resumes from the latest checkpoint under the run directory if one
    exists: the subject and RNG state are restored, and the stream is
    replayed forward without re-processing events already checkpointed.
    """
    directory = _run_directory(config, run_dir)
    checkpoint_dir = directory / CHECKPOINT_DIRNAME

    random.seed(config.seed)

    start_position = 0
    checkpoint = latest_checkpoint(checkpoint_dir)
    if checkpoint is not None:
        subject.restore(checkpoint.subject_state)
        restore_rng_state(checkpoint.rng_state)
        start_position = checkpoint.position

    probe_log_path = directory / PROBE_LOG_FILENAME
    last_checkpoint_time = time.time()

    with open(probe_log_path, "a", encoding="utf-8") as probe_log:
        position = 0
        for item in stream:
            position += 1
            if position <= start_position:
                continue

            event = item.event
            if isinstance(event, Probe):
                answer = isolated_answer(subject, event)
                assert item.truth is not None
                correct = str(item.truth.answer) == answer
                entry = ProbeLogEntry(
                    position=position,
                    probe_id=event.probe_id,
                    task_id=event.task_id,
                    teaching_position=event.teaching_position or 0,
                    correct=correct,
                    cost_counters=subject.cost(),
                )
                probe_log.write(_log_line(entry) + "\n")
            elif isinstance(event, Observe):
                subject.observe(event)
            elif isinstance(event, Idle):
                subject.idle(event.budget)
            elif isinstance(event, Boundary):
                subject.observe(event)

            checkpoint_due = position > 0 and position % config.checkpoint_interval_events == 0
            time_due = time.time() - last_checkpoint_time >= config.checkpoint_interval_seconds
            if checkpoint_due or time_due:
                save_checkpoint(
                    checkpoint_dir,
                    position,
                    subject.snapshot(),
                    capture_rng_state(),
                )
                last_checkpoint_time = time.time()

    return directory
