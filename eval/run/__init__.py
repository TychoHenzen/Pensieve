"""Run records: what one completed eval run produced, and where.

A run directory is a plain folder of JSON files plus one append-only
newline-delimited JSON probe log. `RunRecord` names those files and holds
the run's identity (resolved config, code revision, stream hash,
environment) alongside the paths to its outputs. It is a data holder, not
a manager: it does not read or write files itself.

Per design.md: run records are JSON files in a directory, the probe log
is newline-delimited JSON, and there is no SQLite, no HDF5.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

#: Resolved run configuration, as JSON.
CONFIG_FILENAME = "config.json"
#: Code revision (e.g. a git commit hash), as plain text.
REVISION_FILENAME = "revision.txt"
#: Stream hash identifying the stream this run drew from, as plain text.
STREAM_HASH_FILENAME = "stream_hash.txt"
#: Environment (interpreter, platform, package versions, ...), as JSON.
ENVIRONMENT_FILENAME = "environment.json"
#: Probe log, newline-delimited JSON, one `ProbeLogEntry` per line, append-only.
PROBE_LOG_FILENAME = "probes.ndjson"
#: Metric summary, as JSON.
METRIC_SUMMARY_FILENAME = "metrics.json"
#: Directory holding subject checkpoint files.
CHECKPOINT_DIRNAME = "checkpoints"


def _freeze_mapping(value: Mapping[str, Any]) -> MappingProxyType[str, Any]:
    """Return a read-only copy of `value`."""
    return MappingProxyType(dict(value))


@dataclass(frozen=True, kw_only=True)
class RunRecord:
    """The identity and output paths of one completed eval run.

    `config` is the fully resolved run configuration, not a partial or
    templated one, so a `RunRecord` on its own tells you exactly what ran.
    `environment` and `checkpoint_paths` are frozen in `__post_init__` for
    the same reason `StreamConfig` freezes its params: a `RunRecord` should
    not change out from under whoever holds a reference to it.
    """

    config: Mapping[str, Any]
    code_revision: str
    stream_hash: str
    environment: Mapping[str, str] = field(default_factory=dict)
    probe_log_path: Path
    metric_summary_path: Path
    checkpoint_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", _freeze_mapping(self.config))
        object.__setattr__(self, "environment", _freeze_mapping(self.environment))
        object.__setattr__(self, "checkpoint_paths", tuple(self.checkpoint_paths))
