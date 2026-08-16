"""Checkpoint I/O: atomic writes, loading, and RNG state save/restore.

A checkpoint captures everything a run needs to resume at the same stream
position with the same RNG state: the subject's own snapshot (from
`Subject.snapshot()`) and the state of the process's random number
generators. Checkpoints are written to a directory as one file per
position, named so the position sorts lexicographically with the file
name (`checkpoint_000100.pkl`).

Writes are atomic: the payload is written to a temporary file in the same
directory, then moved into place with `os.replace`, which is an atomic
rename on both POSIX and Windows for a same-filesystem destination. A
crash mid-write leaves the temporary file orphaned and the previous
checkpoint at that position, if any, untouched; it never leaves a
half-written file at the checkpoint's final name.

The on-disk format is pickle, not JSON, because a subject's snapshot is
an arbitrary Python object, not necessarily JSON-serializable. Pickle is
an implementation detail behind `save_checkpoint`/`load_checkpoint`, and
callers should treat checkpoint files as opaque.
"""

from __future__ import annotations

import os
import pickle
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as _numpy
except ImportError:  # pragma: no cover - numpy is an optional dependency here
    _numpy = None

#: Checkpoint file name template, position zero-padded so names sort
#: lexicographically in position order.
_FILENAME_TEMPLATE = "checkpoint_{position:06d}.pkl"
_FILENAME_GLOB = "checkpoint_*.pkl"


@dataclass(frozen=True, kw_only=True)
class CheckpointData:
    """The full contents of one checkpoint.

    `subject_state` is whatever `Subject.snapshot()` returned; this module
    does not interpret it. `rng_state` is whatever `capture_rng_state()`
    returned, and is passed back to `restore_rng_state()` unmodified.
    """

    position: int
    subject_state: object
    rng_state: object
    timestamp: float


def capture_rng_state() -> object:
    """Return the current state of the process's random number generators.

    Captures Python's `random` module state, and numpy's global RNG state
    if numpy is importable. The returned object is opaque to callers and
    must be passed to `restore_rng_state` unmodified.
    """
    state: dict[str, object] = {"python": random.getstate()}
    if _numpy is not None:
        state["numpy"] = _numpy.random.get_state()
    return state


def restore_rng_state(state: object) -> None:
    """Restore random number generator state previously captured.

    `state` must be an object returned by `capture_rng_state`. Restores
    Python's `random` module state, and numpy's global RNG state if it was
    captured and numpy is importable.
    """
    assert isinstance(state, dict)
    random.setstate(state["python"])
    if "numpy" in state and _numpy is not None:
        _numpy.random.set_state(state["numpy"])


def _checkpoint_path(directory: Path, position: int) -> Path:
    return directory / _FILENAME_TEMPLATE.format(position=position)


def save_checkpoint(
    directory: Path, position: int, subject_state: object, rng_state: object
) -> Path:
    """Atomically write a checkpoint for `position` into `directory`.

    Writes the payload to a temporary file in `directory`, then moves it
    into place with `os.replace`. If the process is interrupted before the
    replace completes, the checkpoint's final path is left untouched: it
    either does not exist yet, or still holds whatever a prior write left
    there. Returns the path of the written checkpoint file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    data = CheckpointData(
        position=position,
        subject_state=subject_state,
        rng_state=rng_state,
        timestamp=time.time(),
    )
    final_path = _checkpoint_path(directory, position)

    fd, tmp_name = tempfile.mkstemp(
        dir=directory, prefix=f"{final_path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(data, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return final_path


def load_checkpoint(path: Path) -> CheckpointData:
    """Load and return the `CheckpointData` stored at `path`."""
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    assert isinstance(data, CheckpointData)
    return data


def latest_checkpoint(directory: Path) -> CheckpointData | None:
    """Return the highest-position checkpoint in `directory`, or `None`.

    Returns `None` if `directory` does not exist or contains no checkpoint
    files. Position is read from each file's name, not its contents, so
    finding the latest checkpoint does not require loading every file.
    """
    if not directory.exists():
        return None

    best_position: int | None = None
    best_path: Path | None = None
    for candidate in directory.glob(_FILENAME_GLOB):
        stem = candidate.stem  # "checkpoint_000100"
        try:
            position = int(stem.rsplit("_", maxsplit=1)[-1])
        except ValueError:
            continue
        if best_position is None or position > best_position:
            best_position = position
            best_path = candidate

    if best_path is None:
        return None
    return load_checkpoint(best_path)
