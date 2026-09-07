"""Bounded, schema-checked persistence for Stage 0 gate results."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eval.gate.answer_scoring import extract_predicted_number, score_numerical_answer
from eval.stage0_identity import (
    ASDIV_CONFIGURATION,
    ASDIV_DATASET,
    ASDIV_REVISION,
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
MAX_DEPTH = 8
MAX_COLLECTION_ITEMS = 3_000
MAX_STRING_LENGTH = 1_024
MAX_PREDICTION_LENGTH = 4_096
_HEX = frozenset("0123456789abcdef")


class GateResultError(ValueError):
    """A persisted gate result failed safe parsing or contract validation."""


def checkpoint_sha256(path: Path) -> str:
    _require_local_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rendered_inputs_sha256(rendered_inputs: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(rendered_inputs))).hexdigest()


def _require_local_path(path: Path) -> None:
    raw = str(path)
    if "://" in raw or raw.startswith(("\\\\", "//")):
        raise GateResultError("gate commands accept only local result paths")


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateResultError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise GateResultError(f"non-finite JSON constant {value!r} is forbidden")


def _check_bounds(value: Any, *, depth: int = 0, path: str = "$") -> None:
    if depth > MAX_DEPTH:
        raise GateResultError(f"JSON nesting depth exceeds {MAX_DEPTH} at {path}")
    if isinstance(value, str):
        limit = MAX_PREDICTION_LENGTH if path.endswith((".prediction", ".completion")) else MAX_STRING_LENGTH
        if len(value) > limit:
            raise GateResultError(f"string length exceeds the {limit} character limit at {path}")
        return
    if isinstance(value, Mapping):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise GateResultError(f"collection exceeds 3,000 items at {path}")
        for key, item in value.items():
            if not isinstance(key, str):
                raise GateResultError(f"JSON object field is not a string at {path}")
            if len(key) > MAX_STRING_LENGTH:
                raise GateResultError(f"field string length exceeds limit at {path}")
            _check_bounds(item, depth=depth + 1, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise GateResultError(f"collection exceeds 3,000 items at {path}")
        for index, item in enumerate(value):
            _check_bounds(item, depth=depth + 1, path=f"{path}[{index}]")
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    raise GateResultError(f"unsupported JSON value type at {path}")


def _read_json(path: Path) -> dict[str, Any]:
    _require_local_path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise GateResultError(f"cannot read gate result {path}: {exc}") from exc
    if size > MAX_RESULT_BYTES:
        raise GateResultError("gate result size exceeds 16 MiB")
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GateResultError("gate result is not valid UTF-8 JSON") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_constant,
        )
    except GateResultError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise GateResultError(f"gate result JSON parse failed: {exc}") from exc
    if not isinstance(value, dict):
        raise GateResultError("gate result JSON root must be an object")
    _check_bounds(value)
    return value


def _require_exact_fields(value: Any, required: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GateResultError(f"{path} must be an object")
    fields = set(value)
    missing = required - fields
    unexpected = fields - required
    if missing or unexpected:
        raise GateResultError(
            f"{path} schema fields mismatch: missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}"
        )
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateResultError(f"{path} must be a non-boolean integer")
    if minimum is not None and value < minimum:
        raise GateResultError(f"{path} integer must be at least {minimum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise GateResultError(f"{path} must be a boolean")
    return value


def _string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise GateResultError(f"{path} must be a non-empty string")
    return value


def _digest(value: Any, path: str) -> str:
    text = _string(value, path)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise GateResultError(f"{path} must be a lowercase SHA-256 digest")
    return text


def _validate_manifest(value: Any, path: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise GateResultError(f"{path} manifest must be a non-empty object")
    for filename, entry in value.items():
        _string(filename, f"{path}.field")
        item = _require_exact_fields(entry, {"algorithm", "digest"}, f"{path}.{filename}")
        algorithm = _string(item["algorithm"], f"{path}.{filename}.algorithm")
        digest = _string(item["digest"], f"{path}.{filename}.digest")
        if algorithm not in {"sha256", "git_blob_sha1"}:
            raise GateResultError(f"{path}.{filename}.algorithm has unsupported digest type")
        expected_length = 64 if algorithm == "sha256" else 40
        if len(digest) != expected_length or any(character not in _HEX for character in digest):
            raise GateResultError(f"{path}.{filename}.digest has invalid digest syntax")


def _validate_identity(value: Any, *, latent: bool) -> Mapping[str, Any]:
    fields = {
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
    }
    if latent:
        fields |= {"checkpoint_sha256", "seeds"}
    identity = _require_exact_fields(value, fields, "identity")
    if _integer(identity["schema_version"], "identity.schema_version") != 1:
        raise GateResultError("identity schema_version must be 1")

    selection = _require_exact_fields(
        identity["selection"],
        {
            "schema_version",
            "dataset",
            "configuration",
            "revision",
            "split",
            "seed",
            "problem_count",
            "ordered_item_ids",
            "identity_sha256",
        },
        "identity.selection",
    )
    if _integer(selection["schema_version"], "identity.selection.schema_version") != 1:
        raise GateResultError("selection schema_version must be 1")
    for name in ("dataset", "configuration", "revision", "split"):
        _string(selection[name], f"identity.selection.{name}")
    _integer(selection["seed"], "identity.selection.seed", minimum=0)
    count = _integer(selection["problem_count"], "identity.selection.problem_count", minimum=1)
    ordered = selection["ordered_item_ids"]
    if not isinstance(ordered, list):
        raise GateResultError("identity.selection.ordered_item_ids type must be a list")
    if len(ordered) != count:
        raise GateResultError("selection problem_count does not match ordered item coverage")
    ids = [_string(item, "identity.selection.ordered_item_ids[]") for item in ordered]
    if len(set(ids)) != len(ids):
        raise GateResultError("selection contains duplicate item identifiers")
    _digest(selection["identity_sha256"], "identity.selection.identity_sha256")

    for asset_name in ("model_assets", "tokenizer_assets"):
        asset = _require_exact_fields(
            identity[asset_name], {"repository", "revision", "manifest"}, f"identity.{asset_name}"
        )
        _string(asset["repository"], f"identity.{asset_name}.repository")
        _string(asset["revision"], f"identity.{asset_name}.revision")
        _validate_manifest(asset["manifest"], f"identity.{asset_name}.manifest")
        if (
            asset["repository"] != QWEN_MODEL
            or asset["revision"] != QWEN_REVISION
            or asset["manifest"] != QWEN_MANIFEST
        ):
            raise GateResultError(f"identity.{asset_name} does not match pinned Qwen assets")

    if (
        selection["dataset"] != ASDIV_DATASET
        or selection["configuration"] != ASDIV_CONFIGURATION
        or selection["revision"] != ASDIV_REVISION
        or selection["split"] != "test"
        or selection["seed"] != 0
    ):
        raise GateResultError("identity selection does not match pinned Calc-ASDiv_A test identity")

    prompt = _require_exact_fields(
        identity["prompt"],
        {
            "contract_version",
            "template",
            "assistant_prefill",
            "context_token_limit",
        },
        "identity.prompt",
    )
    _integer(prompt["contract_version"], "identity.prompt.contract_version", minimum=1)
    _string(prompt["template"], "identity.prompt.template")
    _string(prompt["assistant_prefill"], "identity.prompt.assistant_prefill")
    _integer(prompt["context_token_limit"], "identity.prompt.context_token_limit", minimum=1)
    if prompt["template"] != QWEN_MATH_PROMPT:
        raise GateResultError("identity prompt does not match the exact Qwen math prompt")
    if prompt["assistant_prefill"] != QWEN_ANSWER_PREFILL:
        raise GateResultError("identity prompt does not match the exact Qwen answer prefill")
    _digest(identity["rendered_inputs_sha256"], "identity.rendered_inputs_sha256")

    scorer = _require_exact_fields(identity["scorer"], {"contract_version", "grammar"}, "identity.scorer")
    _integer(scorer["contract_version"], "identity.scorer.contract_version", minimum=1)
    _string(scorer["grammar"], "identity.scorer.grammar")

    token = _require_exact_fields(
        identity["token_generation"],
        {"batch_size", "do_sample", "num_beams", "max_new_tokens", "use_cache", "eos_token_id", "pad_token_id"},
        "identity.token_generation",
    )
    for name in ("batch_size", "num_beams", "max_new_tokens", "eos_token_id", "pad_token_id"):
        _integer(token[name], f"identity.token_generation.{name}", minimum=0)
    _boolean(token["do_sample"], "identity.token_generation.do_sample")
    _boolean(token["use_cache"], "identity.token_generation.use_cache")

    latent_generation = _require_exact_fields(
        identity["latent_generation"], {"method", "max_new_tokens", "eos_token_id"}, "identity.latent_generation"
    )
    _string(latent_generation["method"], "identity.latent_generation.method")
    _integer(latent_generation["max_new_tokens"], "identity.latent_generation.max_new_tokens", minimum=1)
    _integer(latent_generation["eos_token_id"], "identity.latent_generation.eos_token_id", minimum=0)

    runtime = _require_exact_fields(
        identity["runtime"],
        {
            "initialization_seed",
            "python_version",
            "numpy_version",
            "pytorch_version",
            "cuda_version",
            "transformers_version",
            "datasets_version",
            "sentence_transformers_version",
            "device_topology",
            "dtype",
            "attention_implementation",
            "cublas_workspace_config",
            "deterministic_algorithms",
            "tf32_enabled",
            "cudnn_benchmark",
            "device",
        },
        "identity.runtime",
    )
    _integer(runtime["initialization_seed"], "identity.runtime.initialization_seed", minimum=0)
    for name in (
        "python_version",
        "numpy_version",
        "pytorch_version",
        "cuda_version",
        "transformers_version",
        "datasets_version",
        "sentence_transformers_version",
        "dtype",
        "attention_implementation",
        "cublas_workspace_config",
        "device",
    ):
        _string(runtime[name], f"identity.runtime.{name}")
    topology = runtime["device_topology"]
    if not isinstance(topology, list):
        raise GateResultError("identity.runtime.device_topology type must be a list")
    for item in topology:
        _string(item, "identity.runtime.device_topology[]")
    for name in ("deterministic_algorithms", "tf32_enabled", "cudnn_benchmark"):
        _boolean(runtime[name], f"identity.runtime.{name}")

    _string(identity["dtype"], "identity.dtype")
    _string(identity["attention_implementation"], "identity.attention_implementation")
    for name in ("slot_count", "latent_step_count", "tap_tuple_index"):
        _integer(identity[name], f"identity.{name}", minimum=0)
    _boolean(identity["development_only"], "identity.development_only")
    if latent:
        _digest(identity["checkpoint_sha256"], "identity.checkpoint_sha256")
        _validate_seeds(identity["seeds"], "identity.seeds")
    return identity


def _validate_seeds(value: Any, path: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise GateResultError(f"{path} must be a non-empty seed list")
    seeds = [_integer(seed, f"{path}[]", minimum=0) for seed in value]
    if len(set(seeds)) != len(seeds):
        raise GateResultError(f"{path} contains duplicate seeds")
    return seeds


def _validate_root(value: Any, *, latent: bool) -> Mapping[str, Any]:
    fields = (
        {"schema_version", "identity", "checkpoint_sha256", "seeds", "runs"}
        if latent
        else {"schema_version", "identity", "items", "correct", "total"}
    )
    root = _require_exact_fields(value, fields, "gate result root")
    if _integer(root["schema_version"], "schema_version") != 2:
        raise GateResultError("gate result schema_version must be 2")
    _validate_identity(root["identity"], latent=latent)
    if latent:
        checkpoint = _digest(root["checkpoint_sha256"], "checkpoint_sha256")
        seeds = _validate_seeds(root["seeds"], "seeds")
        if root["identity"]["checkpoint_sha256"] != checkpoint:
            raise GateResultError("latent checkpoint identity mismatch")
        if root["identity"]["seeds"] != seeds:
            raise GateResultError("latent seed identity mismatch")
        runs = root["runs"]
        if not isinstance(runs, list) or len(runs) != len(seeds):
            raise GateResultError("latent run coverage must match seeds")
        for index, (run, seed) in enumerate(zip(runs, seeds, strict=True)):
            run_map = _require_exact_fields(run, {"seed", "items", "correct", "total"}, f"runs[{index}]")
            if _integer(run_map["seed"], f"runs[{index}].seed") != seed:
                raise GateResultError("latent run seed order mismatch")
            _validate_item_container(run_map, records=None, expected_ids=None, path=f"runs[{index}]")
    else:
        _validate_item_container(
            root,
            records=None,
            expected_ids=None,
            path="token result",
            require_generation_diagnostics=True,
        )
    return root


def _validate_item_container(
    container: Mapping[str, Any],
    *,
    records: Sequence[AsdivRecord] | None,
    expected_ids: Sequence[str] | None,
    path: str,
    require_generation_diagnostics: bool = False,
) -> None:
    items = container.get("items")
    if not isinstance(items, list):
        raise GateResultError(f"{path} items must be a list")
    if expected_ids is not None and [
        item.get("item_id") if isinstance(item, Mapping) else None for item in items
    ] != list(expected_ids):
        raise GateResultError(f"{path} item coverage or order mismatch")
    targets = {record.id: record.target for record in records} if records is not None else {}
    recomputed = 0
    for index, item_value in enumerate(items):
        item_fields = {"item_id", "prediction", "target", "correct"}
        if require_generation_diagnostics:
            item_fields |= {
                "completion",
                "generated_token_count",
                "hit_token_limit",
            }
        item = _require_exact_fields(
            item_value,
            item_fields,
            f"{path}.items[{index}]",
        )
        item_id = _string(item["item_id"], f"{path}.items[{index}].item_id")
        prediction = _string(item["prediction"], f"{path}.items[{index}].prediction", nonempty=False)
        if require_generation_diagnostics:
            completion = _string(
                item["completion"],
                f"{path}.items[{index}].completion",
                nonempty=False,
            )
            _integer(
                item["generated_token_count"],
                f"{path}.items[{index}].generated_token_count",
                minimum=0,
            )
            _boolean(
                item["hit_token_limit"],
                f"{path}.items[{index}].hit_token_limit",
            )
            extracted = extract_predicted_number(QWEN_ANSWER_PREFILL + completion) or ""
            if prediction != extracted:
                raise GateResultError(f"{path} extracted prediction mismatch for {item_id!r}")
        target = _string(item["target"], f"{path}.items[{index}].target")
        stored = _boolean(item["correct"], f"{path}.items[{index}].correct")
        if records is not None:
            if item_id not in targets:
                raise GateResultError(f"{path} contains unknown item {item_id!r}")
            canonical_target = targets[item_id]
            if target != canonical_target:
                raise GateResultError(f"{path} target mismatch for {item_id!r}")
            scored = score_numerical_answer(prediction, canonical_target)
            if stored is not scored:
                raise GateResultError(f"{path} correct score mismatch for {item_id!r}")
            recomputed += int(scored)
        else:
            recomputed += int(stored)
    correct = _integer(container.get("correct"), f"{path}.correct", minimum=0)
    total = _integer(container.get("total"), f"{path}.total", minimum=0)
    if total != len(items):
        raise GateResultError(f"{path} total aggregate does not match items")
    if correct != recomputed:
        raise GateResultError(f"{path} correct aggregate does not match item scores")


def _plain_equal(actual: Any, expected: Any) -> bool:
    return canonical_json_bytes(actual) == canonical_json_bytes(expected)


def _validate_record_selection(identity: Mapping[str, Any], records: Sequence[AsdivRecord]) -> list[str]:
    selection = identity["selection"]
    expected_ids = selection["ordered_item_ids"]
    actual_ids = [record.id for record in records]
    if actual_ids != expected_ids:
        raise GateResultError("canonical record coverage or order does not match selection")
    recomputed = asdiv_a_selection_identity(
        records=records,
        split=selection["split"],
        seed=selection["seed"],
        problem_count=selection["problem_count"],
        revision=selection["revision"],
    )
    if recomputed != selection["identity_sha256"]:
        raise GateResultError("dataset selection identity digest mismatch")
    return expected_ids


def read_token_result(
    path: Path, *, expected_identity: Mapping[str, Any], records: Sequence[AsdivRecord]
) -> dict[str, Any]:
    value = _read_json(path)
    root = _validate_root(value, latent=False)
    _validate_identity(expected_identity, latent=False)
    if not _plain_equal(root["identity"], expected_identity):
        raise GateResultError("cached token identity is incompatible with requested identity")
    expected_ids = _validate_record_selection(root["identity"], records)
    _validate_item_container(
        root,
        records=records,
        expected_ids=expected_ids,
        path="token result",
        require_generation_diagnostics=True,
    )
    return value


def read_latent_result(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_checkpoint_sha256: str,
    expected_seeds: Sequence[int],
    records: Sequence[AsdivRecord],
) -> dict[str, Any]:
    value = _read_json(path)
    root = _validate_root(value, latent=True)
    _validate_identity(expected_identity, latent=True)
    expected_digest = _digest(expected_checkpoint_sha256, "expected checkpoint_sha256")
    expected_seed_list = [_integer(seed, "expected seeds[]", minimum=0) for seed in expected_seeds]
    if len(set(expected_seed_list)) != len(expected_seed_list):
        raise GateResultError("expected seeds contain duplicates")
    if not _plain_equal(root["identity"], expected_identity):
        raise GateResultError("cached latent identity is incompatible with requested identity")
    if root["checkpoint_sha256"] != expected_digest or root["identity"]["checkpoint_sha256"] != expected_digest:
        raise GateResultError("cached checkpoint digest mismatch")
    if root["seeds"] != expected_seed_list or root["identity"]["seeds"] != expected_seed_list:
        raise GateResultError("cached seed identity mismatch")
    expected_ids = _validate_record_selection(root["identity"], records)
    for index, (run, seed) in enumerate(zip(root["runs"], expected_seed_list, strict=True)):
        run_map = _require_exact_fields(run, {"seed", "items", "correct", "total"}, f"runs[{index}]")
        if _integer(run_map["seed"], f"runs[{index}].seed") != seed:
            raise GateResultError("latent run seed order mismatch")
        _validate_item_container(run_map, records=records, expected_ids=expected_ids, path=f"runs[{index}]")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Serialize one bounded JSON mapping with same-directory atomic replacement."""
    _require_local_path(path)
    _check_bounds(value)
    payload = canonical_json_bytes(value)
    if len(payload) > MAX_RESULT_BYTES:
        raise GateResultError("gate result size exceeds 16 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_gate_result(path: Path, result: Mapping[str, Any]) -> None:
    value = dict(result)
    _validate_root(value, latent="runs" in value)
    write_json_atomic(path, value)
