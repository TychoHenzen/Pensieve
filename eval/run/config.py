"""Typed run configuration: resolved from a file plus overrides, and hashed.

A `RunConfig` names the subject, the stream it draws from, its seed, and
its checkpoint and retention schedule. It reads nothing from the
environment at runtime except the filesystem path it was loaded from, so
the same file plus the same overrides always resolves to the same config,
and `config_hash` can turn that config into a stable run id.

`output_dir` is the one field that names a location on this machine, not a
property of the run itself. Two runs with identical config but different
output directories are the same run repeated, not two different runs, so
`output_dir` is excluded from the hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.stream.config import StreamConfig
from eval.stream.hashing import canonical_json


def _stream_config_from(value: StreamConfig | dict[str, Any]) -> StreamConfig:
    """Return `value` as a `StreamConfig`, building one if it is a plain dict."""
    if isinstance(value, StreamConfig):
        return value
    return StreamConfig(**value)


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    """The fully resolved configuration of one eval run.

    Frozen for the same reason `StreamConfig` and `RunRecord` freeze their
    fields: a `RunConfig` should not change out from under whoever holds a
    reference to it.
    """

    subject: str
    stream: StreamConfig
    seed: int
    checkpoint_interval_events: int
    checkpoint_interval_seconds: float
    retention_keep_every: int
    output_dir: Path


def resolve_config(path: Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    """Resolve a `RunConfig` from the JSON file at `path` plus `overrides`.

    `overrides` is applied on top of the file's contents with a shallow
    merge at the top level: a key present in `overrides` replaces the
    file's value for that key entirely, including a nested `stream` dict.
    Nothing is read from the environment; `path` itself is the only
    filesystem input.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    merged = dict(raw)
    if overrides:
        merged.update(overrides)

    stream = _stream_config_from(merged["stream"])
    output_dir = Path(merged["output_dir"])

    return RunConfig(
        subject=merged["subject"],
        stream=stream,
        seed=merged["seed"],
        checkpoint_interval_events=merged["checkpoint_interval_events"],
        checkpoint_interval_seconds=merged["checkpoint_interval_seconds"],
        retention_keep_every=merged["retention_keep_every"],
        output_dir=output_dir,
    )


def config_hash(config: RunConfig) -> str:
    """Return a hex digest identifying `config`, to be used as the run id.

    `output_dir` is excluded: it names where this run's outputs live on
    one machine, not a property of the run itself, so two runs with
    identical config but different output directories share a run id.
    """
    payload = {
        "subject": config.subject,
        "stream": {
            "generator": config.stream.generator,
            "params": config.stream.params,
        },
        "seed": config.seed,
        "checkpoint_interval_events": config.checkpoint_interval_events,
        "checkpoint_interval_seconds": config.checkpoint_interval_seconds,
        "retention_keep_every": config.retention_keep_every,
    }
    canonical = canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
