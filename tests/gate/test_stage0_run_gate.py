"""No-download integration tests for strict Stage 0 gate orchestration."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eval.gate import gate_report, latent_eval, result_cache, run_gate, token_cot_baseline
from tests.gate.test_stage0_result_cache import (
    CHECKPOINT_DIGEST,
    SEEDS,
)
from tests.gate.test_stage0_result_cache import (
    _latent_result as _cached_latent_result,
)
from tests.gate.test_stage0_result_cache import (
    _records as _cache_records,
)
from tests.gate.test_stage0_result_cache import (
    _token_result as _cached_token_result,
)
from tests.gate.test_stage0_token_baseline import (
    _backbone_loader,
    _FakeModel,
    _FakeTokenizer,
    _records,
)


def _args(results_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        device="cpu",
        development_limit=1,
        results_dir=results_dir,
        checkpoint=results_dir / "stage0.ckpt",
        slot_count=16,
        num_steps=2,
    )


def _prepared() -> tuple[Any, _FakeModel]:
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    prepared = token_cot_baseline.prepare_baseline_request(
        device="cpu",
        development_limit=1,
        records=_records(),
        backbone_loader=_backbone_loader(tokenizer, model),
    )
    return prepared, model


def test_default_gate_paths_are_asdiv_a_qwen_scoped() -> None:
    assert Path("gate_results/asdiv_a_qwen") == run_gate.DEFAULT_RESULTS_DIR
    assert token_cot_baseline.DEFAULT_OUTPUT == (
        run_gate.DEFAULT_RESULTS_DIR / "token_cot.json"
    )


def test_valid_token_cache_reuse_recomputes_current_identity_without_generation(
    tmp_path: Path,
) -> None:
    prepared, model = _prepared()
    result = token_cot_baseline.run_baseline(
        device="cpu", prepared_request=prepared
    )
    path = tmp_path / "token_cot.json"
    result_cache.write_gate_result(path, result)
    model.generate_calls.clear()

    loaded = run_gate._run_baseline(_args(tmp_path), _records(), prepared)

    assert loaded == result
    assert model.generate_calls == []


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("rendered_inputs_sha256", "0" * 64),
        ("runtime.python_version", "changed-runtime"),
    ],
)
def test_stale_rendered_or_runtime_identity_is_rejected_before_reuse(
    tmp_path: Path, field: str, replacement: str
) -> None:
    prepared, model = _prepared()
    result = token_cot_baseline.run_baseline(
        device="cpu", prepared_request=prepared
    )
    stale = copy.deepcopy(result)
    if field.startswith("runtime."):
        stale["identity"]["runtime"][field.split(".", 1)[1]] = replacement
    else:
        stale["identity"][field] = replacement
    result_cache.write_gate_result(tmp_path / "token_cot.json", stale)
    model.generate_calls.clear()

    with pytest.raises(result_cache.GateResultError, match=r"identity|incompatible"):
        run_gate._run_baseline(_args(tmp_path), _records(), prepared)

    assert model.generate_calls == []


def test_generated_baseline_uses_atomic_cache_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, _model = _prepared()
    calls: list[tuple[Path, dict[str, Any]]] = []

    monkeypatch.setattr(
        run_gate,
        "write_gate_result",
        lambda path, result: calls.append((path, result)),
    )

    result = run_gate._run_baseline(_args(tmp_path), _records(), prepared)

    assert calls == [(tmp_path / "token_cot.json", result)]


def test_latent_phase_logs_per_seed_and_global_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, _model = _prepared()
    args = _args(tmp_path)
    args.checkpoint.write_bytes(b"checkpoint")
    token_result = token_cot_baseline.run_baseline(
        device="cpu", prepared_request=prepared
    )
    messages: list[str] = []

    def fake_run_eval(**kwargs: Any) -> dict[str, Any]:
        callback = kwargs["on_item"]
        callback(
            latent_eval.LatentProgress(
                seed=0,
                seed_index=1,
                seed_count=5,
                item_index=10,
                item_count=520,
                seed_correct=4,
                global_done=10,
                global_count=2600,
                global_correct=4,
            )
        )
        return {"runs": []}

    monkeypatch.setattr(run_gate, "run_eval", fake_run_eval)
    monkeypatch.setattr(run_gate, "write_gate_result", lambda *_args: None)
    monkeypatch.setattr(run_gate, "_log", messages.append)

    run_gate._run_latent_eval(
        args,
        list(run_gate.DEFAULT_SEEDS),
        token_result,
        _records(),
        prepared,
    )

    progress = next(message for message in messages if "latent seed" in message)
    assert "seed 1/5 (0) [10/520]" in progress
    assert "seed_acc=0.4000 (4/10)" in progress
    assert "global=[10/2600] acc=0.4000 (4/10)" in progress
    assert "elapsed " in progress
    assert "ETA " in progress


def test_token_standalone_main_uses_atomic_cache_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "token.json"
    result = {"correct": 1, "total": 1}
    calls: list[tuple[Path, dict[str, Any]]] = []
    monkeypatch.setattr(
        token_cot_baseline,
        "_parse_args",
        lambda: SimpleNamespace(device="cpu", development_limit=1, output=output),
    )
    monkeypatch.setattr(token_cot_baseline, "configure_deterministic_runtime", lambda: None)
    monkeypatch.setattr(token_cot_baseline, "run_baseline", lambda **_kwargs: result)
    monkeypatch.setattr(
        result_cache,
        "write_gate_result",
        lambda path, value: calls.append((path, value)),
    )

    token_cot_baseline.main()

    assert calls == [(output, result)]


def test_latent_standalone_main_uses_strict_reader_and_atomic_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records()
    token_path = tmp_path / "token.json"
    output = tmp_path / "latent.json"
    args = SimpleNamespace(
        checkpoint=tmp_path / "stage0.ckpt",
        token_result=token_path,
        seeds="0,1,2,3,4",
        development_limit=1,
        slot_count=16,
        num_steps=2,
        device="cpu",
        output=output,
    )
    prepared = SimpleNamespace(
        identity={"current": "identity"},
        selection=SimpleNamespace(records=records[:1]),
        backbone=object(),
    )
    token_result = {"schema_version": 2}
    latent_result = {"runs": [{"seed": 0, "correct": 1, "total": 1}]}
    reads: list[tuple[Path, Any, Any]] = []
    writes: list[tuple[Path, Any]] = []
    monkeypatch.setattr(latent_eval, "_parse_args", lambda: args)
    monkeypatch.setattr(latent_eval, "configure_deterministic_runtime", lambda: None)
    monkeypatch.setattr(latent_eval, "load_asdiv_a_record_split", lambda _split: records)
    monkeypatch.setattr(
        token_cot_baseline,
        "prepare_baseline_request",
        lambda **_kwargs: prepared,
    )
    monkeypatch.setattr(
        result_cache,
        "read_token_result",
        lambda path, *, expected_identity, records: (
            reads.append((path, expected_identity, records)) or token_result
        ),
    )
    monkeypatch.setattr(
        result_cache,
        "write_gate_result",
        lambda path, value: writes.append((path, value)),
    )
    monkeypatch.setattr(latent_eval, "run_eval", lambda **_kwargs: latent_result)

    latent_eval.main()

    assert reads == [(token_path, prepared.identity, prepared.selection.records)]
    assert writes == [(output, latent_result)]


@pytest.mark.parametrize("field", ["rendered_inputs_sha256", "runtime"])
def test_build_report_rejects_stale_token_request_identity(
    tmp_path: Path, field: str
) -> None:
    records = _cache_records()
    token = _cached_token_result(records)
    latent = _cached_latent_result(records)
    expected_identity = copy.deepcopy(token["identity"])
    if field == "rendered_inputs_sha256":
        token["identity"][field] = "0" * 64
    else:
        token["identity"][field]["python_version"] = "changed-runtime"
    result_cache.write_gate_result(tmp_path / "token_cot.json", token)
    result_cache.write_gate_result(tmp_path / "latent_eval.json", latent)

    with pytest.raises(result_cache.GateResultError, match=r"identity|incompatible"):
        gate_report.build_report(
            tmp_path,
            records=records,
            expected_token_identity=expected_identity,
            expected_checkpoint_sha256=CHECKPOINT_DIGEST,
            expected_seeds=SEEDS,
        )


@pytest.mark.parametrize("field", ["checkpoint", "seeds"])
def test_build_report_rejects_stale_latent_checkpoint_or_seeds(
    tmp_path: Path, field: str
) -> None:
    records = _cache_records()
    token = _cached_token_result(records)
    latent = _cached_latent_result(records)
    if field == "checkpoint":
        latent["checkpoint_sha256"] = "d" * 64
        latent["identity"]["checkpoint_sha256"] = "d" * 64
    else:
        changed = [0, 1, 2, 3, 5]
        latent["seeds"] = changed
        latent["identity"]["seeds"] = changed
        latent["runs"][-1]["seed"] = 5
    result_cache.write_gate_result(tmp_path / "token_cot.json", token)
    result_cache.write_gate_result(tmp_path / "latent_eval.json", latent)

    with pytest.raises(
        result_cache.GateResultError,
        match=r"identity|incompatible|checkpoint|seed|mismatch",
    ):
        gate_report.build_report(
            tmp_path,
            records=records,
            expected_token_identity=token["identity"],
            expected_checkpoint_sha256=CHECKPOINT_DIGEST,
            expected_seeds=SEEDS,
        )
