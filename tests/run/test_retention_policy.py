from __future__ import annotations

from pathlib import Path

from eval.run.checkpoint import save_checkpoint
from eval.run.retention_policy import apply_retention


def _positions_present(directory: Path) -> set[int]:
    positions = set()
    for path in directory.glob("checkpoint_*.pkl"):
        positions.add(int(path.stem.rsplit("_", maxsplit=1)[-1]))
    return positions


def test_apply_retention_missing_directory_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    assert apply_retention(missing, keep_every=10) == []


def test_apply_retention_empty_directory_returns_empty(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    assert apply_retention(tmp_path, keep_every=10) == []


def test_apply_retention_keeps_last_and_every_nth(tmp_path: Path) -> None:
    # Checkpoints every 5 events for a 55-event run: 5, 10, ..., 55.
    for position in range(5, 56, 5):
        save_checkpoint(tmp_path, position, subject_state={}, rng_state={})

    deleted = apply_retention(tmp_path, keep_every=10)

    remaining = _positions_present(tmp_path)
    assert remaining == {10, 20, 30, 40, 50, 55}
    assert {int(p.stem.rsplit("_", maxsplit=1)[-1]) for p in deleted} == {
        5,
        15,
        25,
        35,
        45,
    }


def test_apply_retention_keep_every_non_positive_keeps_only_last(
    tmp_path: Path,
) -> None:
    for position in (10, 20, 30):
        save_checkpoint(tmp_path, position, subject_state={}, rng_state={})

    apply_retention(tmp_path, keep_every=0)

    assert _positions_present(tmp_path) == {30}


def test_apply_retention_zero_position_always_kept(tmp_path: Path) -> None:
    save_checkpoint(tmp_path, 0, subject_state={}, rng_state={})
    save_checkpoint(tmp_path, 7, subject_state={}, rng_state={})
    save_checkpoint(tmp_path, 20, subject_state={}, rng_state={})

    apply_retention(tmp_path, keep_every=10)

    assert _positions_present(tmp_path) == {0, 20}
