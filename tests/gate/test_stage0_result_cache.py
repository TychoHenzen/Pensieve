"""Contract tests for safe Stage 0 gate result reuse.

Public API assumptions for task 5.7:

``eval.gate.result_cache`` exposes ``GateResultError``,
``checkpoint_sha256(path)``, ``rendered_inputs_sha256(rendered_inputs)``,
``read_token_result(path, expected_identity=..., records=...)``,
``read_latent_result(path, expected_identity=..., expected_checkpoint_sha256=...,
expected_seeds=..., records=...)``, and ``write_gate_result(path, result)``.

Readers validate the bounded version-2 JSON and return a plain mapping only
after schema, identity, coverage, targets, item scores, and aggregates match.
The writer accepts either validated token or latent root and replaces the
destination atomically.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from eval.gate.answer_scoring import _NUMBER_PATTERN
from eval.stage0_identity import (
    ASDIV_CONFIGURATION,
    ASDIV_DATASET,
    ASDIV_REVISION,
    PROMPT_CONTRACT_VERSION,
    QWEN_ANSWER_PREFILL,
    QWEN_MANIFEST,
    QWEN_MATH_PROMPT,
    QWEN_MODEL,
    QWEN_REVISION,
    canonical_json_bytes,
)
from eval.stream.generators.asdiv_a import (
    AsdivRecord,
    asdiv_a_selection_identity,
)

MAX_RESULT_BYTES = 16 * 1024 * 1024
SEEDS = [0, 1, 2, 3, 4]
CHECKPOINT_DIGEST = hashlib.sha256(b"stage-0-checkpoint").hexdigest()


@pytest.fixture
def cache() -> Any:
    """Import lazily so the red suite collects before task 5.7 adds the module."""
    return importlib.import_module("eval.gate.result_cache")


def _records() -> tuple[AsdivRecord, ...]:
    return (
        AsdivRecord(
            id="asdiv_a__test_alpha",
            split="test",
            question="What is one half?",
            target="0.5",
        ),
        AsdivRecord(
            id="asdiv_a__test_beta",
            split="test",
            question="What is seven minus two?",
            target="5",
        ),
    )


def _rendered_inputs(item_ids: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {"item_id": item_id, "input_ids": [11, index + 20, 31]}
        for index, item_id in enumerate(item_ids)
    ]


def _rendered_digest(item_ids: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(_rendered_inputs(item_ids))).hexdigest()


def _asset_manifest() -> dict[str, dict[str, str]]:
    return {path: dict(digest) for path, digest in QWEN_MANIFEST.items()}


def _identity(item_ids: Sequence[str], *, latent: bool = False) -> dict[str, Any]:
    records_by_id = {record.id: record for record in _records()}
    selected_records = tuple(records_by_id[item_id] for item_id in item_ids)
    identity: dict[str, Any] = {
        "schema_version": 1,
        "selection": {
            "schema_version": 1,
            "dataset": ASDIV_DATASET,
            "configuration": ASDIV_CONFIGURATION,
            "revision": ASDIV_REVISION,
            "split": "test",
            "seed": 0,
            "problem_count": len(item_ids),
            "ordered_item_ids": list(item_ids),
            "identity_sha256": asdiv_a_selection_identity(
                records=selected_records,
                split="test",
                seed=0,
                problem_count=len(selected_records),
                revision=ASDIV_REVISION,
            ),
        },
        "model_assets": {
            "repository": QWEN_MODEL,
            "revision": QWEN_REVISION,
            "manifest": _asset_manifest(),
        },
        "tokenizer_assets": {
            "repository": QWEN_MODEL,
            "revision": QWEN_REVISION,
            "manifest": _asset_manifest(),
        },
        "prompt": {
            "contract_version": PROMPT_CONTRACT_VERSION,
            "template": QWEN_MATH_PROMPT,
            "assistant_prefill": QWEN_ANSWER_PREFILL,
            "context_token_limit": 512,
        },
        "rendered_inputs_sha256": _rendered_digest(item_ids),
        "scorer": {
            "contract_version": 1,
            "grammar": _NUMBER_PATTERN.pattern,
        },
        "token_generation": {
            "batch_size": 1,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 16,
            "use_cache": True,
            "eos_token_id": 151_645,
            "pad_token_id": 151_645,
        },
        "latent_generation": {
            "method": "greedy_argmax",
            "max_new_tokens": 64,
            "eos_token_id": 151_645,
        },
        "runtime": {
            "initialization_seed": 0,
            "python_version": "3.13.5",
            "numpy_version": "2.3.2",
            "pytorch_version": "2.7.1+cu128",
            "cuda_version": "12.8",
            "transformers_version": "4.55.2",
            "datasets_version": "4.0.0",
            "sentence_transformers_version": "5.1.0",
            "device_topology": ["cuda:0"],
            "dtype": "float32",
            "attention_implementation": "eager",
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "tf32_enabled": False,
            "cudnn_benchmark": False,
            "device": "cuda",
        },
        "dtype": "float32",
        "attention_implementation": "eager",
        "slot_count": 16,
        "latent_step_count": 2,
        "tap_tuple_index": 12,
        "development_only": True,
    }
    if latent:
        identity["checkpoint_sha256"] = CHECKPOINT_DIGEST
        identity["seeds"] = list(SEEDS)
    return identity


def _items(
    records: Sequence[AsdivRecord], *, diagnostics: bool = False
) -> list[dict[str, Any]]:
    items = [
        {
            "item_id": records[0].id,
            "prediction": "1/2",
            "target": records[0].target,
            "correct": True,
        },
        {
            "item_id": records[1].id,
            "prediction": "4",
            "target": records[1].target,
            "correct": False,
        },
    ]
    if diagnostics:
        for item in items:
            item.update(
                {
                    "completion": item["prediction"],
                    "generated_token_count": 1,
                    "hit_token_limit": False,
                }
            )
    return items


def _token_result(records: Sequence[AsdivRecord] | None = None) -> dict[str, Any]:
    source = tuple(records or _records())
    ids = [record.id for record in source]
    return {
        "schema_version": 2,
        "identity": _identity(ids),
        "items": _items(source, diagnostics=True),
        "correct": 1,
        "total": len(source),
    }


def _latent_result(records: Sequence[AsdivRecord] | None = None) -> dict[str, Any]:
    source = tuple(records or _records())
    ids = [record.id for record in source]
    return {
        "schema_version": 2,
        "identity": _identity(ids, latent=True),
        "checkpoint_sha256": CHECKPOINT_DIGEST,
        "seeds": list(SEEDS),
        "runs": [
            {
                "seed": seed,
                "items": _items(source),
                "correct": 1,
                "total": len(source),
            }
            for seed in SEEDS
        ],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _set_path(root: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    target = root
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value


def _delete_path(root: dict[str, Any], path: tuple[str, ...]) -> None:
    target = root
    for name in path[:-1]:
        target = target[name]
    del target[path[-1]]


def _read_token(cache: Any, path: Path, records: Sequence[AsdivRecord]) -> Any:
    return cache.read_token_result(
        path,
        expected_identity=_identity([record.id for record in records]),
        records=records,
    )


def _read_latent(cache: Any, path: Path, records: Sequence[AsdivRecord]) -> Any:
    return cache.read_latent_result(
        path,
        expected_identity=_identity([record.id for record in records], latent=True),
        expected_checkpoint_sha256=CHECKPOINT_DIGEST,
        expected_seeds=SEEDS,
        records=records,
    )


# covers: eval/stage0-gate::Result compatibility identity::Matching cached baseline
def test_matching_token_cache_is_reused_only_after_complete_revalidation(
    cache: Any, tmp_path: Path
) -> None:
    records = _records()
    expected = _token_result(records)
    path = tmp_path / "token_cot.json"
    _write_json(path, expected)

    loaded = _read_token(cache, path, records)

    assert loaded == expected
    assert loaded is not expected
    assert loaded["correct"] == 1
    assert loaded["total"] == 2


@pytest.mark.parametrize(
    ("identity_path", "replacement"),
    [
        (("selection", "revision"), "changed-revision"),
        (("model_assets", "revision"), "changed-model"),
        (("tokenizer_assets", "revision"), "changed-tokenizer"),
        (("prompt", "template"), "changed prompt"),
        (("rendered_inputs_sha256",), "0" * 64),
        (("scorer", "grammar"), "changed grammar"),
        (("token_generation", "max_new_tokens"), 63),
        (("latent_generation", "max_new_tokens"), 63),
        (("runtime", "python_version"), "0.0"),
        (("runtime", "cublas_workspace_config"), ":16:8"),
        (("dtype",), "float16"),
        (("attention_implementation",), "sdpa"),
        (("slot_count",), 15),
        (("latent_step_count",), 3),
        (("tap_tuple_index",), 11),
    ],
)
# covers: eval/stage0-gate::Result compatibility identity::Stale cached baseline
def test_changed_identity_field_rejects_stale_token_cache(
    cache: Any,
    tmp_path: Path,
    identity_path: tuple[str, ...],
    replacement: object,
) -> None:
    records = _records()
    result = _token_result(records)
    target: dict[str, Any] = result["identity"]
    for name in identity_path[:-1]:
        target = target[name]
    target[identity_path[-1]] = replacement
    path = tmp_path / "token_cot.json"
    _write_json(path, result)

    with pytest.raises(cache.GateResultError, match=r"identity|incompatible|mismatch"):
        _read_token(cache, path, records)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "selection",
        "model_assets",
        "tokenizer_assets",
        "prompt",
        "rendered_inputs_sha256",
        "scorer",
        "token_generation",
        "latent_generation",
        "runtime",
        "dtype",
        "attention_implementation",
        "slot_count",
        "latent_step_count",
        "tap_tuple_index",
        "development_only",
    ],
)
def test_absent_identity_field_is_never_compatible(
    cache: Any, tmp_path: Path, field: str
) -> None:
    records = _records()
    result = _token_result(records)
    del result["identity"][field]
    path = tmp_path / "token_cot.json"
    _write_json(path, result)

    with pytest.raises(cache.GateResultError, match=r"identity|missing|required"):
        _read_token(cache, path, records)


@pytest.mark.parametrize(
    ("identity_path", "invalid_value"),
    [
        (("schema_version",), True),
        (("selection", "seed"), False),
        (("selection", "problem_count"), "2"),
        (("selection", "ordered_item_ids"), "asdiv_a__test_alpha"),
        (("model_assets", "manifest", "config.json", "algorithm"), "sha1"),
        (("prompt", "contract_version"), False),
        (("prompt", "context_token_limit"), "512"),
        (("rendered_inputs_sha256",), "not-a-sha256"),
        (("scorer", "contract_version"), 1.0),
        (("token_generation", "do_sample"), 0),
        (("token_generation", "max_new_tokens"), "64"),
        (("latent_generation", "eos_token_id"), True),
        (("runtime", "deterministic_algorithms"), 1),
        (("runtime", "device_topology"), "cuda:0"),
        (("runtime", "initialization_seed"), False),
        (("slot_count",), False),
        (("development_only",), 0),
    ],
)
def test_identity_schema_remains_typed_when_cached_and_requested_values_match(
    cache: Any,
    tmp_path: Path,
    identity_path: tuple[str, ...],
    invalid_value: object,
) -> None:
    records = _records()
    result = _token_result(records)
    expected_identity = copy.deepcopy(result["identity"])
    _set_path(result["identity"], identity_path, invalid_value)
    _set_path(expected_identity, identity_path, invalid_value)
    path = tmp_path / "token_cot.json"
    _write_json(path, result)

    with pytest.raises(cache.GateResultError, match=r"type|integer|boolean|schema|digest"):
        cache.read_token_result(
            path,
            expected_identity=expected_identity,
            records=records,
        )


@pytest.mark.parametrize(
    "identity_path",
    [
        ("selection", "revision"),
        ("model_assets", "manifest"),
        ("tokenizer_assets", "repository"),
        ("prompt", "template"),
        ("scorer", "grammar"),
        ("token_generation", "use_cache"),
        ("latent_generation", "method"),
        ("runtime", "python_version"),
        ("runtime", "cublas_workspace_config"),
    ],
)
def test_nested_identity_fields_are_required_even_if_request_also_omits_them(
    cache: Any, tmp_path: Path, identity_path: tuple[str, ...]
) -> None:
    records = _records()
    result = _token_result(records)
    expected_identity = copy.deepcopy(result["identity"])
    _delete_path(result["identity"], identity_path)
    _delete_path(expected_identity, identity_path)
    path = tmp_path / "token_cot.json"
    _write_json(path, result)

    with pytest.raises(cache.GateResultError, match=r"missing|required|schema|field"):
        cache.read_token_result(
            path,
            expected_identity=expected_identity,
            records=records,
        )


def test_rendered_inputs_digest_uses_canonical_ordered_item_and_token_ids(
    cache: Any,
) -> None:
    rendered = _rendered_inputs(["alpha", "beta"])
    expected = hashlib.sha256(canonical_json_bytes(rendered)).hexdigest()

    assert cache.rendered_inputs_sha256(rendered) == expected
    assert cache.rendered_inputs_sha256(list(reversed(rendered))) != expected
    changed = copy.deepcopy(rendered)
    changed[0]["input_ids"][0] += 1
    assert cache.rendered_inputs_sha256(changed) != expected


def test_checkpoint_digest_hashes_exact_file_bytes(cache: Any, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"stage-0-checkpoint")

    assert cache.checkpoint_sha256(checkpoint) == CHECKPOINT_DIGEST

    checkpoint.write_bytes(b"stage-0-checkpoint-changed")
    assert cache.checkpoint_sha256(checkpoint) != CHECKPOINT_DIGEST


def test_matching_latent_cache_requires_checkpoint_seed_and_identity_match(
    cache: Any, tmp_path: Path
) -> None:
    records = _records()
    expected = _latent_result(records)
    path = tmp_path / "latent_eval.json"
    _write_json(path, expected)

    assert _read_latent(cache, path, records) == expected


@pytest.mark.parametrize("location", ["root", "identity"])
def test_changed_checkpoint_digest_rejects_latent_cache(
    cache: Any, tmp_path: Path, location: str
) -> None:
    records = _records()
    result = _latent_result(records)
    if location == "root":
        result["checkpoint_sha256"] = "d" * 64
    else:
        result["identity"]["checkpoint_sha256"] = "d" * 64
    path = tmp_path / "latent_eval.json"
    _write_json(path, result)

    with pytest.raises(cache.GateResultError, match=r"checkpoint|identity|mismatch"):
        _read_latent(cache, path, records)


@pytest.mark.parametrize("location", ["root", "identity"])
def test_changed_or_non_exact_seeds_reject_latent_cache(
    cache: Any, tmp_path: Path, location: str
) -> None:
    records = _records()
    result = _latent_result(records)
    if location == "root":
        result["seeds"] = [0, 1, 2, 3, 5]
    else:
        result["identity"]["seeds"] = [0, 1, 2, 3, 5]
    path = tmp_path / "latent_eval.json"
    _write_json(path, result)

    with pytest.raises(cache.GateResultError, match=r"seed|identity|mismatch"):
        _read_latent(cache, path, records)


@pytest.mark.parametrize(
    ("kind", "mutate"),
    [
        ("token", lambda result: result["items"].append(copy.deepcopy(result["items"][0]))),
        ("token", lambda result: result["items"].pop()),
        ("token", lambda result: result["items"].reverse()),
        ("token", lambda result: result["items"][0].update(item_id="unknown")),
        ("latent", lambda result: result["runs"][0]["items"].append(copy.deepcopy(result["runs"][0]["items"][0]))),
        ("latent", lambda result: result["runs"][0]["items"].pop()),
        ("latent", lambda result: result["runs"][0]["items"].reverse()),
        ("latent", lambda result: result["runs"].pop()),
        ("latent", lambda result: result["runs"][1].update(seed=0)),
    ],
)
def test_item_and_run_coverage_must_be_unique_complete_and_ordered(
    cache: Any, tmp_path: Path, kind: str, mutate: Any
) -> None:
    records = _records()
    result = _token_result(records) if kind == "token" else _latent_result(records)
    mutate(result)
    path = tmp_path / f"{kind}.json"
    _write_json(path, result)

    with pytest.raises(
        cache.GateResultError,
        match=r"item|coverage|duplicate|order|seed|run|unknown",
    ):
        (_read_token if kind == "token" else _read_latent)(cache, path, records)


@pytest.mark.parametrize("kind", ["token", "latent"])
def test_targets_are_reloaded_from_validated_records(
    cache: Any, tmp_path: Path, kind: str
) -> None:
    records = _records()
    result = _token_result(records) if kind == "token" else _latent_result(records)
    item = result["items"][0] if kind == "token" else result["runs"][0]["items"][0]
    item.update(prediction="999", target="999", correct=True)
    if kind == "token":
        item["completion"] = "999"
    path = tmp_path / f"{kind}.json"
    _write_json(path, result)

    with pytest.raises(cache.GateResultError, match=r"target|correct|score"):
        (_read_token if kind == "token" else _read_latent)(cache, path, records)


@pytest.mark.parametrize("kind", ["token", "latent"])
@pytest.mark.parametrize("field", ["correct", "total"])
def test_stored_item_counts_and_aggregates_must_recompute(
    cache: Any, tmp_path: Path, kind: str, field: str
) -> None:
    records = _records()
    result = _token_result(records) if kind == "token" else _latent_result(records)
    container = result if kind == "token" else result["runs"][0]
    if field == "correct":
        container["items"][0]["correct"] = False
    else:
        container["total"] += 1
    path = tmp_path / f"{kind}.json"
    _write_json(path, result)

    with pytest.raises(cache.GateResultError, match=r"correct|total|aggregate|score"):
        (_read_token if kind == "token" else _read_latent)(cache, path, records)


def test_reader_recomputes_with_shared_numerical_scorer(
    cache: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records()
    result = _token_result(records)
    path = tmp_path / "token.json"
    _write_json(path, result)
    calls: list[tuple[str, str]] = []

    def always_incorrect(prediction: str, target: str) -> bool:
        calls.append((prediction, target))
        return False

    monkeypatch.setattr(cache, "score_numerical_answer", always_incorrect)

    with pytest.raises(cache.GateResultError, match=r"correct|score|aggregate"):
        _read_token(cache, path, records)

    assert calls[0] == (result["items"][0]["prediction"], records[0].target)


@pytest.mark.parametrize("kind", ["token", "latent"])
@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result.update(schema_version=True),
        lambda result: result.update(schema_version=3),
        lambda result: result.update(unexpected=True),
        lambda result: result.pop("identity"),
    ],
)
def test_token_and_latent_roots_are_exact_typed_version_two_schemas(
    cache: Any, tmp_path: Path, kind: str, mutation: Any
) -> None:
    records = _records()
    result = _token_result(records) if kind == "token" else _latent_result(records)
    mutation(result)
    path = tmp_path / f"{kind}.json"
    _write_json(path, result)

    with pytest.raises(cache.GateResultError, match=r"schema|version|field|identity"):
        (_read_token if kind == "token" else _read_latent)(cache, path, records)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b'{"schema_version":2,"schema_version":2}', r"duplicate"),
        (b'{"schema_version":2,"identity":NaN}', r"finite|constant|NaN"),
        (b"\xff", r"UTF-8|decode|JSON"),
        (b"{", r"JSON|parse"),
    ],
)
def test_unsafe_json_is_rejected_before_schema_use(
    cache: Any, tmp_path: Path, raw: bytes, expected: str
) -> None:
    path = tmp_path / "token.json"
    path.write_bytes(raw)

    with pytest.raises(cache.GateResultError, match=expected):
        _read_token(cache, path, _records())


def test_result_size_is_bounded_before_json_parse(cache: Any, tmp_path: Path) -> None:
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (MAX_RESULT_BYTES + 1))

    with pytest.raises(cache.GateResultError, match=r"16 MiB|size|large"):
        _read_token(cache, path, _records())


def test_nesting_depth_over_eight_is_rejected(cache: Any, tmp_path: Path) -> None:
    result = _token_result()
    nested: object = "leaf"
    for _ in range(9):
        nested = [nested]
    result["identity"]["runtime"]["device_topology"] = nested
    path = tmp_path / "deep.json"
    _write_json(path, result)

    with pytest.raises(cache.GateResultError, match=r"depth|nesting|8"):
        _read_token(cache, path, _records())


def test_collection_over_three_thousand_items_is_rejected(
    cache: Any, tmp_path: Path
) -> None:
    result = _token_result()
    result["items"] = [copy.deepcopy(result["items"][0]) for _ in range(3_001)]
    path = tmp_path / "wide.json"
    _write_json(path, result)

    with pytest.raises(cache.GateResultError, match=r"3.?000|collection|items"):
        _read_token(cache, path, _records())


@pytest.mark.parametrize(
    ("location", "value"),
    [
        ("prediction", "x" * 4_097),
        ("item_id", "x" * 1_025),
        ("runtime", "x" * 1_025),
    ],
)
def test_strings_are_bounded_by_field(cache: Any, tmp_path: Path, location: str, value: str) -> None:
    result = _token_result()
    if location == "runtime":
        result["identity"]["runtime"]["python_version"] = value
    else:
        result["items"][0][location] = value
    path = tmp_path / "strings.json"
    _write_json(path, result)

    with pytest.raises(cache.GateResultError, match=r"length|long|string|limit"):
        _read_token(cache, path, _records())


def test_writer_fsyncs_same_directory_temp_then_atomically_replaces(
    cache: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "token_cot.json"
    result = _token_result()
    fsync_calls: list[int] = []
    replace_calls: list[tuple[Path, Path]] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def record_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        real_fsync(fd)

    def record_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        source_path = Path(source)
        target_path = Path(target)
        replace_calls.append((source_path, target_path))
        real_replace(source, target)

    monkeypatch.setattr(cache.os, "fsync", record_fsync)
    monkeypatch.setattr(cache.os, "replace", record_replace)

    cache.write_gate_result(destination, result)

    assert json.loads(destination.read_text(encoding="utf-8")) == result
    assert fsync_calls
    assert len(replace_calls) == 1
    temporary, target = replace_calls[0]
    assert temporary.parent == destination.parent
    assert target == destination
    assert not temporary.exists()


def test_failed_atomic_replace_preserves_old_result_and_removes_temp(
    cache: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "token_cot.json"
    old_content = b'{"old":"cache"}'
    destination.write_bytes(old_content)
    seen_temp: Path | None = None

    def fail_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        del target
        nonlocal seen_temp
        seen_temp = Path(source)
        assert seen_temp.parent == destination.parent
        assert seen_temp.exists()
        raise OSError("simulated replace interruption")

    monkeypatch.setattr(cache.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace interruption"):
        cache.write_gate_result(destination, _token_result())

    assert destination.read_bytes() == old_content
    assert seen_temp is not None
    assert not seen_temp.exists()
