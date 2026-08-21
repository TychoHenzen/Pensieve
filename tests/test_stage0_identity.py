import importlib
from unittest.mock import Mock

import pytest


DATASET = "MU-NLPC/Calc-mawps"
DATASET_CONFIGURATION = "default"
DATASET_REVISION = "38c10053efeafd20ab6ff4e08c3ec17de26c19b7"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

EXPECTED_STAGE0_IDENTITY = {
    "schema_version": 1,
    "dataset": DATASET,
    "dataset_configuration": DATASET_CONFIGURATION,
    "dataset_revision": DATASET_REVISION,
    "model": QWEN_MODEL,
    "model_revision": QWEN_REVISION,
    "sentence_encoder": MINILM_MODEL,
    "sentence_encoder_revision": MINILM_REVISION,
    "workspace_dimension": 896,
    "latent_tap_layer": 12,
    "prompt_contract_version": 1,
    "numerical_scorer_contract_version": 1,
}

EXPECTED_CANONICAL_IDENTITY = (
    b'{"dataset":"MU-NLPC/Calc-mawps","dataset_configuration":"default",'
    b'"dataset_revision":"38c10053efeafd20ab6ff4e08c3ec17de26c19b7",'
    b'"latent_tap_layer":12,"model":"Qwen/Qwen2.5-0.5B-Instruct",'
    b'"model_revision":"7ae557604adf67be50417f59c2c2f167def9a775",'
    b'"numerical_scorer_contract_version":1,"prompt_contract_version":1,'
    b'"schema_version":1,'
    b'"sentence_encoder":"sentence-transformers/all-MiniLM-L6-v2",'
    b'"sentence_encoder_revision":"1110a243fdf4706b3f48f1d95db1a4f5529b4d41",'
    b'"workspace_dimension":896}'
)
EXPECTED_IDENTITY_SHA256 = (
    "3f2cc635e1ed53cbda5b9c986ebb80893afdf6e7ff3711322cffc736a8938083"
)

EXPECTED_DATASET_MANIFEST = {
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

EXPECTED_QWEN_MANIFEST = {
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

EXPECTED_MINILM_MANIFEST = {
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

PROMPT = (
    "Solve the following math word problem. Show your reasoning, then end your "
    'response with exactly "#### <answer>", where <answer> is a finite integer, '
    "decimal, or fraction.\n\nProblem:\n{question}"
)


def _subject():
    return importlib.import_module("eval.stage0_identity")


def _row(split: str, index: int) -> dict[str, object]:
    return {
        "id": f"mawps__{split}_{index}",
        "question": f"What is {index} plus 1?",
        "result": str(index + 1),
        "result_float": float(index + 1),
    }


def _raw_rows(split: str, count: int) -> list[dict[str, object]]:
    rows = [_row(split, index) for index in range(count)]
    if split == "validation":
        rows[-1]["id"] = "mawps__qA0gWJatQEeMzvOw"
    return rows


def test_stage0_identity_has_pinned_coordinates_and_canonical_sha256():
    stage0 = _subject()

    assert dict(stage0.STAGE0_IDENTITY) == EXPECTED_STAGE0_IDENTITY
    assert stage0.canonical_json_bytes(stage0.STAGE0_IDENTITY) == EXPECTED_CANONICAL_IDENTITY
    assert stage0.identity_sha256(stage0.STAGE0_IDENTITY) == EXPECTED_IDENTITY_SHA256


def test_canonical_json_is_compact_sorted_utf8_and_rejects_non_finite_values():
    stage0 = _subject()

    assert stage0.canonical_json_bytes({"z": "café", "a": [1, 2]}) == (
        b'{"a":[1,2],"z":"caf\xc3\xa9"}'
    )
    with pytest.raises(ValueError):
        stage0.canonical_json_bytes({"not_finite": float("nan")})


def test_manifests_pin_every_declared_asset_and_digest_algorithm():
    stage0 = _subject()

    assert dict(stage0.CALC_MAWPS_MANIFEST) == EXPECTED_DATASET_MANIFEST
    assert dict(stage0.QWEN_MANIFEST) == EXPECTED_QWEN_MANIFEST
    assert dict(stage0.MINILM_MANIFEST) == EXPECTED_MINILM_MANIFEST


def test_manifest_digest_uses_git_blob_framing_or_raw_sha256_as_declared():
    stage0 = _subject()

    assert stage0.digest_bytes(b"hello", "git_blob_sha1") == (
        "b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0"
    )
    assert stage0.digest_bytes(b"hello", "sha256") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_qwen_prompt_is_one_exact_user_message():
    stage0 = _subject()
    question = "A cafe has 12 cups and buys 3 more. How many cups?"

    assert stage0.qwen_messages(question) == [
        {"role": "user", "content": PROMPT.format(question=question)}
    ]


def test_qwen_prompt_uses_exact_chat_template_arguments():
    stage0 = _subject()
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = {"input_ids": [[7, 8, 9]]}
    question = "What is 2 + 2?"

    rendered = stage0.apply_qwen_chat_template(tokenizer, question)

    assert rendered == {"input_ids": [[7, 8, 9]]}
    tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": PROMPT.format(question=question)}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )


@pytest.mark.parametrize(
    ("split", "raw_count", "usable_count"),
    (("train", 1089, 1089), ("validation", 1040, 1039), ("test", 520, 520)),
)
# covers: eval/generators/calc-mawps::Filtered Calc-MAWPS split binding::Default filtered splits load
def test_default_filtered_splits_verify_raw_count_and_return_expected_rows(
    split: str, raw_count: int, usable_count: int
):
    stage0 = _subject()
    dataset_loader = Mock(return_value=_raw_rows(split, raw_count))

    records = stage0.load_calc_mawps_split(split, dataset_loader=dataset_loader)

    assert len(records) == usable_count
    dataset_loader.assert_called_once_with(
        DATASET,
        DATASET_CONFIGURATION,
        split=split,
        revision=DATASET_REVISION,
        trust_remote_code=False,
    )


def test_dataset_raw_count_mismatch_fails_verification():
    stage0 = _subject()
    dataset_loader = Mock(return_value=_raw_rows("test", 519))

    with pytest.raises(ValueError, match=r"test.*520.*519"):
        stage0.load_calc_mawps_split("test", dataset_loader=dataset_loader)


# covers: eval/generators/calc-mawps::Filtered Calc-MAWPS split binding::Dataset revision unavailable
def test_unavailable_dataset_revision_error_names_dataset_and_revision():
    stage0 = _subject()
    dataset_loader = Mock(side_effect=OSError("revision not found"))

    with pytest.raises(RuntimeError) as error:
        stage0.load_calc_mawps_split("train", dataset_loader=dataset_loader)

    message = str(error.value)
    assert DATASET in message
    assert DATASET_REVISION in message


def test_qwen_and_minilm_loaders_pin_revisions_and_disable_unsafe_loading():
    stage0 = _subject()
    qwen_loader = Mock(return_value=object())
    tokenizer_loader = Mock(return_value=object())
    minilm_loader = Mock(return_value=object())

    stage0.load_qwen_model(model_loader=qwen_loader)
    stage0.load_qwen_tokenizer(tokenizer_loader=tokenizer_loader)
    stage0.load_minilm_model(model_loader=minilm_loader)

    qwen_loader.assert_called_once_with(
        QWEN_MODEL,
        revision=QWEN_REVISION,
        trust_remote_code=False,
        use_safetensors=True,
    )
    tokenizer_loader.assert_called_once_with(
        QWEN_MODEL,
        revision=QWEN_REVISION,
        trust_remote_code=False,
    )
    minilm_loader.assert_called_once_with(
        MINILM_MODEL,
        revision=MINILM_REVISION,
        trust_remote_code=False,
        use_safetensors=True,
    )


@pytest.mark.parametrize(
    ("loader_name", "repository", "revision"),
    (
        ("load_qwen_model", QWEN_MODEL, QWEN_REVISION),
        ("load_minilm_model", MINILM_MODEL, MINILM_REVISION),
    ),
)
def test_unavailable_model_revision_error_names_model_and_revision(
    loader_name: str, repository: str, revision: str
):
    stage0 = _subject()
    model_loader = Mock(side_effect=OSError("revision not found"))

    with pytest.raises(RuntimeError) as error:
        getattr(stage0, loader_name)(model_loader=model_loader)

    message = str(error.value)
    assert repository in message
    assert revision in message


# covers: train/stage0-training::Stage 0 identity accompanies training artifacts::New checkpoint identity
def test_training_identity_contains_runtime_fields_and_fixed_held_out_identifiers():
    stage0 = _subject()
    runtime = {
        "initialization_seed": 0,
        "python_version": "3.13.5",
        "numpy_version": "2.3.2",
        "pytorch_version": "2.8.0",
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
    }
    held_out_item_ids = ["mawps__validation_17", "mawps__validation_904"]

    identity = stage0.training_identity(
        runtime=runtime, held_out_item_ids=held_out_item_ids
    )

    assert identity == {
        **EXPECTED_STAGE0_IDENTITY,
        "runtime": runtime,
        "held_out_item_ids": held_out_item_ids,
    }
