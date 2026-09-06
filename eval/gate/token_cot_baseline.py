"""Frozen-Qwen token baseline for the Stage 0 Calc-ASDiv_A gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from codecs_module.decoder import DEFAULT_MAX_TOKENS
from eval.gate import result_cache
from eval.gate.answer_scoring import (
    _NUMBER_PATTERN,
    MIN_MEANINGFUL_ACCURACY,
    extract_predicted_number,
    score_numerical_answer,
)
from eval.stage0_identity import (
    ASDIV_CONFIGURATION,
    ASDIV_DATASET,
    ASDIV_REVISION,
    LATENT_TAP_LAYER,
    NUMERICAL_SCORER_CONTRACT_VERSION,
    PROMPT_CONTRACT_VERSION,
    QWEN_ANSWER_PREFILL,
    QWEN_MANIFEST,
    QWEN_MATH_PROMPT,
    QWEN_MODEL,
    QWEN_REVISION,
    apply_qwen_chat_template,
    canonical_json_bytes,
    load_frozen_qwen_backbone,
)
from eval.stream.generators.asdiv_a import (
    AsdivRecord,
    AsdivSelection,
    load_asdiv_a_record_split,
    select_asdiv_a_records,
)
from eval.subjects.latent_core import DEFAULT_NUM_STEPS
from train.answer_objective import QWEN_PROMPT_TOKEN_LIMIT
from train.standalone_checkpoint import (
    configure_deterministic_runtime,
    runtime_identity,
)
from workspace.concept_slots import DEFAULT_SLOT_COUNT

DEFAULT_OUTPUT = Path("gate_results/asdiv_a_qwen/token_cot.json")
LOW_ACCURACY_WARNING_THRESHOLD = MIN_MEANINGFUL_ACCURACY
DEFAULT_MAX_NEW_TOKENS = 16
TEST_SEED = 0
FULL_TEST_COUNT = 520


_extract_predicted_number = extract_predicted_number


@dataclass(frozen=True)
class PreparedBaselineRequest:
    """Current request identity and rendered inputs, prepared without generation."""

    selection: AsdivSelection
    backbone: Any
    rendered: tuple[Mapping[str, Any], ...]
    identity: dict[str, Any]
    eos_token_id: int


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _plain_mapping(item) if isinstance(item, Mapping) else item for key, item in value.items()}


def _selection_identity(selection: AsdivSelection) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": ASDIV_DATASET,
        "configuration": ASDIV_CONFIGURATION,
        "revision": ASDIV_REVISION,
        "split": selection.split,
        "seed": selection.seed,
        "problem_count": selection.problem_count,
        "ordered_item_ids": list(selection.ordered_item_ids),
        "identity_sha256": selection.identity,
    }


def _rendered_ids(value: Any) -> list[int]:
    rows = value.tolist()
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
        raise ValueError("Qwen chat input_ids must have shape (1, tokens)")
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in rows[0]):
        raise ValueError("Qwen chat input_ids must contain integers")
    return rows[0]


def _result_identity(
    *,
    selection: AsdivSelection,
    rendered_inputs: Sequence[Mapping[str, Any]],
    device: str,
    runtime: Mapping[str, Any],
    eos_token_id: int,
    development_only: bool,
) -> dict[str, Any]:
    generation = {
        "batch_size": 1,
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "use_cache": True,
        "eos_token_id": eos_token_id,
        "pad_token_id": eos_token_id,
    }
    return {
        "schema_version": 1,
        "selection": _selection_identity(selection),
        "model_assets": {
            "repository": QWEN_MODEL,
            "revision": QWEN_REVISION,
            "manifest": _plain_mapping(QWEN_MANIFEST),
        },
        "tokenizer_assets": {
            "repository": QWEN_MODEL,
            "revision": QWEN_REVISION,
            "manifest": _plain_mapping(QWEN_MANIFEST),
        },
        "prompt": {
            "contract_version": PROMPT_CONTRACT_VERSION,
            "template": QWEN_MATH_PROMPT,
            "assistant_prefill": QWEN_ANSWER_PREFILL,
            "context_token_limit": QWEN_PROMPT_TOKEN_LIMIT,
        },
        "rendered_inputs_sha256": hashlib.sha256(canonical_json_bytes(list(rendered_inputs))).hexdigest(),
        "scorer": {
            "contract_version": NUMERICAL_SCORER_CONTRACT_VERSION,
            "grammar": _NUMBER_PATTERN.pattern,
        },
        "token_generation": generation,
        "latent_generation": {
            "method": "greedy_argmax",
            "max_new_tokens": DEFAULT_MAX_TOKENS,
            "eos_token_id": eos_token_id,
        },
        "runtime": {**dict(runtime), "device": device},
        "dtype": "float32",
        "attention_implementation": "eager",
        "slot_count": DEFAULT_SLOT_COUNT,
        "latent_step_count": DEFAULT_NUM_STEPS,
        "tap_tuple_index": LATENT_TAP_LAYER,
        "development_only": development_only,
    }


def _select_records(records: Sequence[AsdivRecord], development_limit: int | None) -> AsdivSelection:
    if len(records) != FULL_TEST_COUNT:
        raise ValueError(f"Calc-ASDiv_A test records must contain exactly {FULL_TEST_COUNT} items")
    if any(record.split != "test" for record in records):
        raise ValueError("Calc-ASDiv_A token baseline accepts only test records")
    if development_limit is not None:
        if isinstance(development_limit, bool) or not isinstance(development_limit, int):
            raise TypeError(f"development_limit must be a non-boolean integer from 1 through {FULL_TEST_COUNT}")
        if not 1 <= development_limit <= FULL_TEST_COUNT:
            raise ValueError(f"development_limit must be from 1 through {FULL_TEST_COUNT}")
    return select_asdiv_a_records(
        {"test": tuple(records)},
        split="test",
        seed=TEST_SEED,
        problem_count=development_limit,
    )


def _validated_eos_token_id(tokenizer: Any, generation_config: Any) -> int:
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tokenizer_eos, bool) or not isinstance(tokenizer_eos, int):
        raise TypeError("Qwen tokenizer must define an integer EOS token")
    configured = getattr(generation_config, "eos_token_id", None)
    configured_ids = configured if isinstance(configured, list) else [configured]
    if (
        not configured_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in configured_ids)
        or len(set(configured_ids)) != len(configured_ids)
    ):
        raise ValueError("Qwen generation config must define unique integer EOS tokens")
    if tokenizer_eos not in configured_ids:
        raise ValueError("Qwen tokenizer and generation config EOS tokens must match")
    return tokenizer_eos


def prepare_baseline_request(
    *,
    device: str,
    development_limit: int | None = None,
    records: Sequence[AsdivRecord] | None = None,
    backbone_loader: Callable[..., Any] = load_frozen_qwen_backbone,
    runtime_configurer: Callable[[], None] = configure_deterministic_runtime,
) -> PreparedBaselineRequest:
    """Build the complete current identity and render prompts without generating."""
    runtime_configurer()
    run_identity = runtime_identity()
    source_records = tuple(records) if records is not None else load_asdiv_a_record_split("test")
    selection = _select_records(source_records, development_limit)
    backbone = backbone_loader(device=device)
    if getattr(backbone.config, "hidden_size", None) != 896:
        raise ValueError("Qwen backbone hidden_size must be 896")
    tokenizer = backbone.tokenizer
    eos_token_id = _validated_eos_token_id(tokenizer, backbone.generation_config)

    rendered_values: list[Mapping[str, Any]] = []
    rendered_inputs: list[dict[str, Any]] = []
    for record in selection.records:
        rendered = apply_qwen_chat_template(tokenizer, record.question)
        if not isinstance(rendered, Mapping) or "input_ids" not in rendered:
            raise ValueError("Qwen chat template must return input_ids")
        token_ids = _rendered_ids(rendered["input_ids"])
        if len(token_ids) > QWEN_PROMPT_TOKEN_LIMIT:
            raise ValueError(f"Qwen chat prompt exceeds the 512-token Stage 0 limit: actual {len(token_ids)}")
        rendered_values.append(rendered)
        rendered_inputs.append({"item_id": record.id, "input_ids": token_ids})
    identity = _result_identity(
        selection=selection,
        rendered_inputs=rendered_inputs,
        device=device,
        runtime=run_identity,
        eos_token_id=eos_token_id,
        development_only=development_limit is not None,
    )
    return PreparedBaselineRequest(
        selection=selection,
        backbone=backbone,
        rendered=tuple(rendered_values),
        identity=identity,
        eos_token_id=eos_token_id,
    )


def run_baseline(
    *,
    device: str,
    development_limit: int | None = None,
    records: Sequence[AsdivRecord] | None = None,
    backbone_loader: Callable[..., Any] = load_frozen_qwen_backbone,
    runtime_configurer: Callable[[], None] = configure_deterministic_runtime,
    on_problem: Callable[[int, int, bool, int, int], None] | None = None,
    prepared_request: PreparedBaselineRequest | None = None,
) -> dict[str, Any]:
    """Evaluate the persisted seed-0 Calc-ASDiv_A test selection."""
    prepared = prepared_request or prepare_baseline_request(
        device=device,
        development_limit=development_limit,
        records=records,
        backbone_loader=backbone_loader,
        runtime_configurer=runtime_configurer,
    )
    selection = prepared.selection
    backbone = prepared.backbone
    tokenizer = backbone.tokenizer
    model = backbone.model
    eos_token_id = prepared.eos_token_id

    correct = 0
    items: list[dict[str, Any]] = []
    for index, (record, rendered) in enumerate(zip(selection.records, prepared.rendered, strict=True)):
        token_ids = _rendered_ids(rendered["input_ids"])
        input_ids = rendered["input_ids"].to(device)
        generation_kwargs: dict[str, Any] = {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "use_cache": True,
            "eos_token_id": eos_token_id,
            "pad_token_id": eos_token_id,
        }
        if "attention_mask" in rendered:
            generation_kwargs["attention_mask"] = rendered["attention_mask"].to(device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, **generation_kwargs)
        completion_ids = output_ids[0][len(token_ids) :]
        generated_ids = completion_ids.tolist() if hasattr(completion_ids, "tolist") else list(completion_ids)
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        prediction = extract_predicted_number(QWEN_ANSWER_PREFILL + completion) or ""
        is_correct = score_numerical_answer(prediction, record.target)
        correct += int(is_correct)
        items.append(
            {
                "item_id": record.id,
                "completion": completion,
                "generated_token_count": len(generated_ids),
                "hit_token_limit": (
                    len(generated_ids) == DEFAULT_MAX_NEW_TOKENS
                    and (not generated_ids or generated_ids[-1] != eos_token_id)
                ),
                "prediction": prediction,
                "target": record.target,
                "correct": is_correct,
            }
        )
        if on_problem is not None:
            done = index + 1
            on_problem(index, len(selection.records), is_correct, correct, done)

    return {
        "schema_version": 2,
        "identity": copy.deepcopy(prepared.identity),
        "items": items,
        "correct": correct,
        "total": len(items),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0 frozen-Qwen token baseline on Calc-ASDiv_A.")
    parser.add_argument(
        "--development-limit",
        type=int,
        default=None,
        help=("Development-only prefix size from 1 through 520. The default evaluates the full gate selection."),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    configure_deterministic_runtime()
    args = _parse_args()
    result = run_baseline(
        device=args.device,
        development_limit=args.development_limit,
    )

    result_cache.write_gate_result(args.output, result)

    accuracy = result["correct"] / result["total"] if result["total"] else 0.0
    print(f"token-CoT baseline accuracy: {accuracy:.4f} ({result['correct']}/{result['total']})")
    if accuracy < LOW_ACCURACY_WARNING_THRESHOLD:
        print(
            "WARNING: token-CoT baseline accuracy is below "
            f"{LOW_ACCURACY_WARNING_THRESHOLD:.0%}; this result cannot establish "
            "the Stage 0 accuracy floor."
        )
    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
