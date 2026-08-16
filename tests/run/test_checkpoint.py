from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from unittest.mock import patch

import pytest

from eval.run.checkpoint import (
    CheckpointData,
    capture_rng_state,
    latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)


def _rng_states_equal(left: object, right: object) -> bool:
    """Compare two `capture_rng_state()` results, element by element.

    Needed because a numpy array's `==` is elementwise, not a bool, so
    `dict.__eq__` (and thus a plain `==` on the states) raises when a
    numpy array is present.
    """
    assert isinstance(left, dict) and isinstance(right, dict)
    if left.keys() != right.keys():
        return False
    if left["python"] != right["python"]:
        return False
    if "numpy" in left:
        for left_part, right_part in zip(left["numpy"], right["numpy"], strict=True):
            equal = left_part == right_part
            if hasattr(equal, "all"):
                if not equal.all():
                    return False
            elif not equal:
                return False
    return True


def test_save_load_round_trip(tmp_path: Path) -> None:
    subject_state = {"foo": "bar", "n": 3}
    rng_state = capture_rng_state()

    path = save_checkpoint(
        tmp_path, position=100, subject_state=subject_state, rng_state=rng_state
    )
    loaded = load_checkpoint(path)

    assert loaded.position == 100
    assert loaded.subject_state == subject_state
    assert _rng_states_equal(loaded.rng_state, rng_state)


def test_latest_checkpoint_returns_highest_position(tmp_path: Path) -> None:
    save_checkpoint(tmp_path, position=100, subject_state={}, rng_state={})
    save_checkpoint(tmp_path, position=300, subject_state={}, rng_state={})
    save_checkpoint(tmp_path, position=200, subject_state={}, rng_state={})

    latest = latest_checkpoint(tmp_path)

    assert latest is not None
    assert latest.position == 300


def test_latest_checkpoint_missing_directory_returns_none(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    assert latest_checkpoint(missing) is None


def test_latest_checkpoint_empty_directory_returns_none(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    assert latest_checkpoint(tmp_path) is None


def test_interrupted_write_leaves_prior_checkpoint_intact(tmp_path: Path) -> None:
    save_checkpoint(tmp_path, position=100, subject_state={"n": 1}, rng_state={})

    with patch("eval.run.checkpoint.os.replace", side_effect=OSError("simulated crash")):
        with pytest.raises(OSError, match="simulated crash"):
            save_checkpoint(tmp_path, position=200, subject_state={"n": 2}, rng_state={})

    # The prior checkpoint at position 100 is still intact and loadable.
    latest = latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.position == 100
    assert latest.subject_state == {"n": 1}

    # No half-written file remains at position 200's final name.
    final_path_200 = tmp_path / "checkpoint_000200.pkl"
    assert not final_path_200.exists()

    # No leftover temp files either: the failed write cleans up after itself.
    remaining_tmp_files = list(tmp_path.glob("*.tmp"))
    assert remaining_tmp_files == []


def test_rng_state_round_trip(tmp_path: Path) -> None:
    random.seed(12345)
    state = capture_rng_state()
    expected_next_values = [random.random() for _ in range(5)]

    # Perturb the RNG state.
    for _ in range(50):
        random.random()

    restore_rng_state(state)
    actual_next_values = [random.random() for _ in range(5)]

    assert actual_next_values == expected_next_values


def test_checkpoint_data_is_frozen() -> None:
    data = CheckpointData(
        position=1, subject_state=None, rng_state=None, timestamp=0.0
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        data.position = 2  # type: ignore[misc]
