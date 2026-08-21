"""Shared immutable coordinates and prompt contract for Stage 0.

This module defines the new Calc-MAWPS and Qwen identity without changing any
runtime entry point. Loaders accept injected callables so contract tests do not
need network access.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch


CALC_MAWPS_DATASET = "MU-NLPC/Calc-mawps"
CALC_MAWPS_CONFIGURATION = "default"
CALC_MAWPS_REVISION = "38c10053efeafd20ab6ff4e08c3ec17de26c19b7"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

WORKSPACE_DIMENSION = 896
LATENT_TAP_LAYER = 12
PROMPT_CONTRACT_VERSION = 1
NUMERICAL_SCORER_CONTRACT_VERSION = 1

CALC_MAWPS_RAW_COUNTS = MappingProxyType(
    {"train": 1089, "validation": 1040, "test": 520}
)
CALC_MAWPS_VALIDATION_EXCLUDED_ID = "mawps__qA0gWJatQEeMzvOw"

QWEN_MATH_PROMPT = (
    "Solve the following math word problem. Show your reasoning, then end your "
    'response with exactly "#### <answer>", where <answer> is a finite integer, '
    "decimal, or fraction.\n\nProblem:\n{question}"
)


def _immutable_manifest(
    entries: Mapping[str, Mapping[str, str]],
) -> MappingProxyType[str, MappingProxyType[str, str]]:
    return MappingProxyType(
        {path: MappingProxyType(dict(digest)) for path, digest in entries.items()}
    )


STAGE0_IDENTITY = MappingProxyType(
    {
        "schema_version": 1,
        "dataset": CALC_MAWPS_DATASET,
        "dataset_configuration": CALC_MAWPS_CONFIGURATION,
        "dataset_revision": CALC_MAWPS_REVISION,
        "model": QWEN_MODEL,
        "model_revision": QWEN_REVISION,
        "sentence_encoder": MINILM_MODEL,
        "sentence_encoder_revision": MINILM_REVISION,
        "workspace_dimension": WORKSPACE_DIMENSION,
        "latent_tap_layer": LATENT_TAP_LAYER,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "numerical_scorer_contract_version": NUMERICAL_SCORER_CONTRACT_VERSION,
    }
)

CALC_MAWPS_MANIFEST = _immutable_manifest(
    {
        "data/train-00000-of-00001-4bb1451333aad61c.parquet": {
            "algorithm": "sha256",
            "digest": "7a8dd8f7680b5e5ddb908cdfa33867c298d9225d1d7595a17d4771c09e482140",
        },
        "data/validation-00000-of-00001-2ce28573971ca59f.parquet": {
            "algorithm": "sha256",
            "digest": "9d2ca9e33d8efdbc2527a84f1685d6dd5ec7defb201a290a836660c071f5b607",
        },
        "data/test-00000-of-00001-5a59f3fc4b0d9c98.parquet": {
            "algorithm": "sha256",
            "digest": "2ef6313cb811d5c5ebef422a909015f7db4750020f835fd95c2ac2ccbf343d82",
        },
    }
)

QWEN_MANIFEST = _immutable_manifest(
    {
        "config.json": {
            "algorithm": "git_blob_sha1",
            "digest": "0dbb161213629a23f0fc00ef286e6b1e366d180f",
        },
        "generation_config.json": {
            "algorithm": "git_blob_sha1",
            "digest": "dfc11073787daf1b0f9c0f1499487ab5f4c93738",
        },
        "merges.txt": {
            "algorithm": "git_blob_sha1",
            "digest": "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0",
        },
        "tokenizer.json": {
            "algorithm": "git_blob_sha1",
            "digest": "443909a61d429dff23010e5bddd28ff530edda00",
        },
        "tokenizer_config.json": {
            "algorithm": "git_blob_sha1",
            "digest": "07bfe0640cb5a0037f9322287fbfc682806cf672",
        },
        "vocab.json": {
            "algorithm": "git_blob_sha1",
            "digest": "4783fe10ac3adce15ac8f358ef5462739852c569",
        },
        "model.safetensors": {
            "algorithm": "sha256",
            "digest": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
        },
    }
)

MINILM_MANIFEST = _immutable_manifest(
    {
        "1_Pooling/config.json": {
            "algorithm": "git_blob_sha1",
            "digest": "d1514c3162bbe87b343f565fadc62e6c06f04f03",
        },
        "config.json": {
            "algorithm": "git_blob_sha1",
            "digest": "72b987fd805cfa2b58c4c8c952b274a11bfd5a00",
        },
        "config_sentence_transformers.json": {
            "algorithm": "git_blob_sha1",
            "digest": "fd1b291129c607e5d49799f87cb219b27f98acdf",
        },
        "modules.json": {
            "algorithm": "git_blob_sha1",
            "digest": "952a9b81c0bfd99800fabf352f69c7ccd46c5e43",
        },
        "sentence_bert_config.json": {
            "algorithm": "git_blob_sha1",
            "digest": "59d594003bf59880a884c574bf88ef7555bb0202",
        },
        "tokenizer.json": {
            "algorithm": "git_blob_sha1",
            "digest": "cb202bfe2e3c98645018a6d12f182a434c9d3e02",
        },
        "tokenizer_config.json": {
            "algorithm": "git_blob_sha1",
            "digest": "c79f2b6a0cea6f4b564fed1938984bace9d30ff0",
        },
        "vocab.txt": {
            "algorithm": "git_blob_sha1",
            "digest": "fb140275c155a9c7c5a3b3e0e77a9e839594a938",
        },
        "model.safetensors": {
            "algorithm": "sha256",
            "digest": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
        },
    }
)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact, sorted UTF-8 JSON and reject non-finite numbers."""
    return json.dumps(
        _json_compatible(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def identity_sha256(identity: Mapping[str, Any] = STAGE0_IDENTITY) -> str:
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def digest_bytes(content: bytes, algorithm: str) -> str:
    """Digest raw bytes with the declared Hugging Face manifest semantics."""
    if algorithm == "sha256":
        return hashlib.sha256(content).hexdigest()
    if algorithm == "git_blob_sha1":
        framed = b"blob " + str(len(content)).encode("ascii") + b"\0" + content
        return hashlib.sha1(framed).hexdigest()
    raise ValueError(f"Unsupported digest algorithm: {algorithm}")


def qwen_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": QWEN_MATH_PROMPT.format(question=question)}
    ]


def apply_qwen_chat_template(tokenizer: Any, question: str) -> Any:
    return tokenizer.apply_chat_template(
        qwen_messages(question),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )


def _load_failure(repository: str, revision: str, error: Exception) -> RuntimeError:
    return RuntimeError(
        f"Could not load repository {repository} at pinned revision {revision}: {error}"
    )


@dataclass(frozen=True)
class FrozenQwenBackbone:
    """Pinned Qwen assets shared by one Stage 0 runtime."""

    model: Any
    tokenizer: Any
    config: Any
    generation_config: Any


def verify_huggingface_manifest(
    repository: str,
    revision: str,
    manifest: Mapping[str, Mapping[str, str]],
) -> None:
    """Download and verify every asset allowed by a pinned manifest."""
    try:
        from huggingface_hub import hf_hub_download

        for path, expected in manifest.items():
            local_path = hf_hub_download(
                repository,
                filename=path,
                revision=revision,
            )
            actual = digest_bytes(Path(local_path).read_bytes(), expected["algorithm"])
            if actual != expected["digest"]:
                raise ValueError(
                    f"{path}: expected {expected['digest']}, actual {actual}"
                )
    except Exception as error:
        raise _load_failure(repository, revision, error) from error


def load_frozen_qwen_backbone(
    *,
    device: str | torch.device = "cpu",
    model_loader: Callable[..., Any] | None = None,
    tokenizer_loader: Callable[..., Any] | None = None,
    config_loader: Callable[..., Any] | None = None,
    generation_config_loader: Callable[..., Any] | None = None,
    manifest_verifier: Callable[[str, str, Mapping[str, Mapping[str, str]]], None]
    | None = None,
) -> FrozenQwenBackbone:
    """Load one pinned, verified Qwen model and freeze it for Stage 0."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if manifest_verifier is None:
        manifest_verifier = verify_huggingface_manifest
    if model_loader is None or tokenizer_loader is None or config_loader is None or generation_config_loader is None:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            GenerationConfig,
        )

        model_loader = model_loader or AutoModelForCausalLM.from_pretrained
        tokenizer_loader = tokenizer_loader or AutoTokenizer.from_pretrained
        config_loader = config_loader or AutoConfig.from_pretrained
        generation_config_loader = (
            generation_config_loader or GenerationConfig.from_pretrained
        )

    try:
        manifest_verifier(QWEN_MODEL, QWEN_REVISION, QWEN_MANIFEST)
        config = config_loader(
            QWEN_MODEL,
            revision=QWEN_REVISION,
            trust_remote_code=False,
        )
        generation_config = generation_config_loader(
            QWEN_MODEL,
            revision=QWEN_REVISION,
            trust_remote_code=False,
        )
        tokenizer = tokenizer_loader(
            QWEN_MODEL,
            revision=QWEN_REVISION,
            trust_remote_code=False,
        )
        model = model_loader(
            QWEN_MODEL,
            revision=QWEN_REVISION,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            config=config,
        )
    except Exception as error:
        raise _load_failure(QWEN_MODEL, QWEN_REVISION, error) from error

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.generation_config = generation_config
    return FrozenQwenBackbone(
        model=model,
        tokenizer=tokenizer,
        config=config,
        generation_config=generation_config,
    )


def load_qwen_model(model_loader: Callable[..., Any] | None = None) -> Any:
    if model_loader is None:
        from transformers import AutoModelForCausalLM

        model_loader = AutoModelForCausalLM.from_pretrained
    try:
        return model_loader(
            QWEN_MODEL,
            revision=QWEN_REVISION,
            trust_remote_code=False,
            use_safetensors=True,
        )
    except Exception as error:
        raise _load_failure(QWEN_MODEL, QWEN_REVISION, error) from error


def load_qwen_tokenizer(tokenizer_loader: Callable[..., Any] | None = None) -> Any:
    if tokenizer_loader is None:
        from transformers import AutoTokenizer

        tokenizer_loader = AutoTokenizer.from_pretrained
    try:
        return tokenizer_loader(
            QWEN_MODEL,
            revision=QWEN_REVISION,
            trust_remote_code=False,
        )
    except Exception as error:
        raise _load_failure(QWEN_MODEL, QWEN_REVISION, error) from error


def _sentence_transformer_loader(
    repository: str,
    *,
    revision: str,
    trust_remote_code: bool,
    use_safetensors: bool,
) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        repository,
        revision=revision,
        trust_remote_code=trust_remote_code,
        model_kwargs={"use_safetensors": use_safetensors},
    )


def load_minilm_model(model_loader: Callable[..., Any] | None = None) -> Any:
    if model_loader is None:
        model_loader = _sentence_transformer_loader
    try:
        return model_loader(
            MINILM_MODEL,
            revision=MINILM_REVISION,
            trust_remote_code=False,
            use_safetensors=True,
        )
    except Exception as error:
        raise _load_failure(MINILM_MODEL, MINILM_REVISION, error) from error


def load_calc_mawps_split(
    split: str,
    dataset_loader: Callable[..., Any] | None = None,
) -> list[Mapping[str, Any]]:
    if split not in CALC_MAWPS_RAW_COUNTS:
        allowed = ", ".join(CALC_MAWPS_RAW_COUNTS)
        raise ValueError(f"Unknown Calc-MAWPS split {split!r}; expected one of {allowed}")
    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset

    try:
        loaded = dataset_loader(
            CALC_MAWPS_DATASET,
            CALC_MAWPS_CONFIGURATION,
            split=split,
            revision=CALC_MAWPS_REVISION,
            trust_remote_code=False,
        )
    except Exception as error:
        raise _load_failure(
            CALC_MAWPS_DATASET, CALC_MAWPS_REVISION, error
        ) from error

    expected_count = CALC_MAWPS_RAW_COUNTS[split]
    actual_count = len(loaded)
    if actual_count != expected_count:
        raise ValueError(
            f"Calc-MAWPS {split} raw count mismatch: expected "
            f"{expected_count}, found {actual_count}"
        )

    rows = list(loaded)
    if split != "validation":
        return rows

    excluded_count = sum(
        row.get("id") == CALC_MAWPS_VALIDATION_EXCLUDED_ID for row in rows
    )
    if excluded_count != 1:
        raise ValueError(
            "Calc-MAWPS validation pinned exclusion mismatch: expected one "
            f"{CALC_MAWPS_VALIDATION_EXCLUDED_ID}, found {excluded_count}"
        )
    return [
        row
        for row in rows
        if row.get("id") != CALC_MAWPS_VALIDATION_EXCLUDED_ID
    ]


def training_identity(
    *, runtime: Mapping[str, Any], held_out_item_ids: Sequence[str]
) -> dict[str, Any]:
    return {
        **STAGE0_IDENTITY,
        "runtime": dict(runtime),
        "held_out_item_ids": list(held_out_item_ids),
    }
