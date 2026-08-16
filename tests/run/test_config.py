from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.run.config import RunConfig, config_hash, resolve_config
from eval.stream.config import StreamConfig

_BASE_CONFIG: dict[str, Any] = {
    "subject": "perfect_memory",
    "stream": {"generator": "assoc", "params": {}},
    "seed": 42,
    "checkpoint_interval_events": 100,
    "checkpoint_interval_seconds": 300.0,
    "retention_keep_every": 10,
    "output_dir": "/tmp/runs",
}


def _write_config(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_same_config_produces_same_run_id(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _BASE_CONFIG)

    config_a = resolve_config(path)
    config_b = resolve_config(path)

    assert config_hash(config_a) == config_hash(config_b)


def test_different_seed_produces_different_run_id(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _BASE_CONFIG)

    base = resolve_config(path)
    changed = resolve_config(path, overrides={"seed": 43})

    assert config_hash(base) != config_hash(changed)


def test_different_output_dir_produces_same_run_id(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _BASE_CONFIG)

    base = resolve_config(path)
    changed = resolve_config(path, overrides={"output_dir": "/tmp/other-runs"})

    assert base.output_dir != changed.output_dir
    assert config_hash(base) == config_hash(changed)


def test_resolve_config_reads_json_file(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _BASE_CONFIG)

    config = resolve_config(path)

    assert config.subject == "perfect_memory"
    assert config.stream == StreamConfig(generator="assoc", params={})
    assert config.seed == 42
    assert config.checkpoint_interval_events == 100
    assert config.checkpoint_interval_seconds == 300.0
    assert config.retention_keep_every == 10
    assert config.output_dir == Path("/tmp/runs")


def test_resolve_config_applies_overrides(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _BASE_CONFIG)

    config = resolve_config(
        path,
        overrides={"subject": "other_subject", "seed": 99},
    )

    assert config.subject == "other_subject"
    assert config.seed == 99
    # Untouched fields still come from the file.
    assert config.retention_keep_every == 10


def test_resolve_config_override_replaces_nested_stream_entirely(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path, _BASE_CONFIG)

    config = resolve_config(
        path,
        overrides={"stream": {"generator": "other", "params": {"k": "v"}}},
    )

    assert config.stream == StreamConfig(generator="other", params={"k": "v"})


def test_resolve_config_accepts_env_var_style_path(tmp_path: Path) -> None:
    payload = dict(_BASE_CONFIG)
    payload["output_dir"] = "$HOME/runs"
    path = _write_config(tmp_path, payload)

    config = resolve_config(path)

    assert config.output_dir == Path("$HOME/runs")


def test_resolve_config_accepts_regular_path_string(tmp_path: Path) -> None:
    payload = dict(_BASE_CONFIG)
    payload["output_dir"] = str(tmp_path / "runs")
    path = _write_config(tmp_path, payload)

    config = resolve_config(path)

    assert config.output_dir == tmp_path / "runs"


def test_run_config_is_frozen(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _BASE_CONFIG)
    config = resolve_config(path)

    with pytest.raises(AttributeError):
        config.seed = 100  # type: ignore[misc]
