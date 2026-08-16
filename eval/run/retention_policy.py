"""Checkpoint retention policy: keep the last checkpoint plus every Nth.

A run may write many checkpoints over its lifetime. Keeping all of them
wastes disk space once a run has advanced past them, so the retention
policy prunes checkpoint files down to a fixed pattern: the highest
position ("the last checkpoint") is always kept, and every checkpoint
whose position is a multiple of `keep_every` is kept. Everything else is
deleted.

This module only inspects and deletes files by name; it never opens a
checkpoint file's contents, matching `checkpoint.latest_checkpoint`'s
approach of reading position from the file name.
"""

from __future__ import annotations

from pathlib import Path

from eval.run.checkpoint import _FILENAME_GLOB


def _position_of(path: Path) -> int | None:
    stem = path.stem  # "checkpoint_000100"
    try:
        return int(stem.rsplit("_", maxsplit=1)[-1])
    except ValueError:
        return None


def apply_retention(directory: Path, keep_every: int) -> list[Path]:
    """Delete checkpoints in `directory` outside the retention policy.

    Keeps the checkpoint with the highest position (the "last" one) and
    every checkpoint whose position is a multiple of `keep_every`.
    Deletes every other checkpoint file matching the checkpoint naming
    pattern and returns the list of deleted paths.

    Returns an empty list if `directory` does not exist or contains no
    checkpoint files. If `keep_every` is not positive, only the last
    checkpoint is kept.
    """
    if not directory.exists():
        return []

    positioned: list[tuple[int, Path]] = []
    for candidate in directory.glob(_FILENAME_GLOB):
        position = _position_of(candidate)
        if position is None:
            continue
        positioned.append((position, candidate))

    if not positioned:
        return []

    last_position = max(position for position, _ in positioned)

    deleted: list[Path] = []
    for position, path in positioned:
        keep = position == last_position
        if not keep and keep_every > 0 and position % keep_every == 0:
            keep = True
        if not keep:
            path.unlink()
            deleted.append(path)

    return deleted
