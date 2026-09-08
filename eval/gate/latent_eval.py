"""Latent Stage 0 evaluation on the persisted Calc-ASDiv_A test selection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.gate import result_cache, token_cot_baseline
from eval.gate.answer_scoring import extract_predicted_number, score_numerical_answer
from eval.stage0_identity import (
    ASDIV_REVISION,
    LATENT_TAP_LAYER,
    apply_qwen_chat_template,
    canonical_json_bytes,
    load_frozen_qwen_backbone,
)
from eval.stream.generators.asdiv_a import (
    AsdivRecord,
    asdiv_a_selection_identity,
    load_asdiv_a_record_split,
    select_asdiv_a_records,
)
from eval.subjects.latent_core import DEFAULT_NUM_STEPS, LatentCoreSubject
from train.alternating_checkpoint import load_checkpoint as load_alternating_checkpoint
from train.answer_objective import QWEN_PROMPT_TOKEN_LIMIT, SUBJECT_LATENT_RUNS_PER_ANSWER
from train.standalone_checkpoint import configure_deterministic_runtime
from workspace.concept_slots import DEFAULT_SLOT_COUNT

DEFAULT_OUTPUT = Path("gate_results/asdiv_a_qwen/latent_eval.json")
DEFAULT_TOKEN_RESULT = Path("gate_results/asdiv_a_qwen/token_cot.json")
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
FULL_TEST_COUNT = 520


@dataclass(frozen=True)
class LatentProgress:
    """Progress after one latent-evaluation item has completed."""

    seed: int
    seed_index: int
    seed_count: int
    item_index: int
    item_count: int
    seed_correct: int
    global_done: int
    global_count: int
    global_correct: int


def _checkpoint_digest(checkpoint_path: Path | None) -> str | None:
    if checkpoint_path is None:
        return None
    return hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()


def load_subject(
    checkpoint_path: Path | None,
    slot_count: int,
    num_steps: int,
    device: str,
    *,
    backbone: object | None = None,
) -> LatentCoreSubject:
    """Build a frozen-Qwen subject and apply only a safe version-2 checkpoint."""
    checkpoint = None
    if checkpoint_path is not None:
        if checkpoint_path.suffix != ".ckpt":
            raise ValueError("Stage 0 latent evaluation requires a version-2 .ckpt file")
        checkpoint = load_alternating_checkpoint(checkpoint_path)
        slot_queries = checkpoint.tensors.get("model.encoder.slot_queries")
        if not isinstance(slot_queries, torch.Tensor) or slot_queries.ndim != 2:
            raise ValueError("checkpoint must contain model.encoder.slot_queries")
        checkpoint_slot_count = int(slot_queries.shape[0])
        if checkpoint_slot_count != slot_count:
            raise ValueError(
                "checkpoint slot count does not match the requested slot count: "
                f"checkpoint={checkpoint_slot_count}, requested={slot_count}"
            )

    shared_backbone = backbone or load_frozen_qwen_backbone(device=device)
    subject = LatentCoreSubject(
        slot_count=slot_count,
        num_steps=num_steps,
        device=device,
        backbone=shared_backbone,
    )
    if checkpoint is None:
        return subject

    model_state = {
        name.removeprefix("model."): tensor.to(device)
        for name, tensor in checkpoint.tensors.items()
        if name.startswith("model.")
    }
    subject.encoder.load_state_dict(
        {name.removeprefix("encoder."): tensor for name, tensor in model_state.items() if name.startswith("encoder.")},
        strict=False,
    )
    subject.latent_loop.load_state_dict(
        {
            name.removeprefix("latent_loop."): tensor
            for name, tensor in model_state.items()
            if name.startswith("latent_loop.")
        },
        strict=False,
    )
    return subject


def _rendered_ids(value: Any) -> list[int]:
    rows = value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
        raise ValueError("Qwen chat input_ids must have shape (1, tokens)")
    ids = rows[0]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
        raise ValueError("Qwen chat input_ids must contain integers")
    if len(ids) > QWEN_PROMPT_TOKEN_LIMIT:
        raise ValueError(f"Qwen chat prompt exceeds the 512-token Stage 0 limit: actual {len(ids)}")
    return ids


def evaluate_item(subject: LatentCoreSubject, record: AsdivRecord) -> dict[str, Any]:
    """Evaluate one record with the same Qwen prompt and latent path as training."""
    rendered = apply_qwen_chat_template(subject.tokenizer, record.question)
    if not isinstance(rendered, Mapping) or "input_ids" not in rendered:
        raise ValueError("Qwen chat template must return input_ids")
    input_ids = _rendered_ids(rendered["input_ids"])
    context_ids = rendered["input_ids"].to(subject.latent_loop.device)
    with torch.no_grad():
        context_embeds = subject.latent_loop.embed_tokens(context_ids)
        subject.workspace.write_slots(subject.encoder.encode(record.question))
        for _ in range(SUBJECT_LATENT_RUNS_PER_ANSWER):
            subject.latent_loop.run(subject.workspace, context_embeds=context_embeds)
        answer = subject.decoder.decode(subject.workspace)
    return {"prediction": extract_predicted_number(answer) or "", "input_ids": input_ids}


def _validate_development_limit(development_limit: int | None, available: int) -> None:
    if development_limit is None:
        return
    if isinstance(development_limit, bool) or not isinstance(development_limit, int):
        raise TypeError(f"development_limit must be a non-boolean integer from 1 through {available}")
    if not 1 <= development_limit <= available:
        raise ValueError(f"development_limit must be from 1 through {available}")


def _validate_records(records: Sequence[AsdivRecord]) -> tuple[AsdivRecord, ...]:
    source = tuple(records)
    if len(source) != FULL_TEST_COUNT:
        raise ValueError(f"Calc-ASDiv_A test records must contain exactly {FULL_TEST_COUNT} items")
    if any(record.split != "test" for record in source):
        raise ValueError("latent evaluation accepts only Calc-ASDiv_A test records")
    ids = [record.id for record in source]
    if len(set(ids)) != len(ids):
        raise ValueError("Calc-ASDiv_A test records contain duplicate item identifiers")
    return source


def _token_order(token_result: Mapping[str, Any], records: Sequence[AsdivRecord]) -> tuple[str, ...]:
    if token_result.get("schema_version") != 2:
        raise ValueError("token result schema_version must be 2")
    identity = token_result.get("identity")
    if not isinstance(identity, Mapping):
        raise TypeError("token result identity must be an object")
    selection = identity.get("selection")
    if not isinstance(selection, Mapping):
        raise TypeError("token result selection identity must be an object")
    if selection.get("split") != "test" or selection.get("seed") != 0:
        raise ValueError("token result selection must be the seed-0 test split")
    ordered = selection.get("ordered_item_ids")
    if not isinstance(ordered, Sequence) or isinstance(ordered, (str, bytes)):
        raise TypeError("token result ordered item identifiers must be a list")
    ordered_ids = tuple(ordered)
    if any(not isinstance(item_id, str) or not item_id for item_id in ordered_ids):
        raise ValueError("token result item identifiers must be non-empty strings")
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("token result contains duplicate item identifiers")
    if not ordered_ids:
        raise ValueError("token result selection must contain item identifiers")
    if selection.get("problem_count") != len(ordered_ids):
        raise ValueError("token result selection problem_count differs from its item IDs")

    canonical_order = select_asdiv_a_records(
        {"test": tuple(records)}, split="test", seed=0, problem_count=None
    ).ordered_item_ids
    unknown = sorted(set(ordered_ids) - set(canonical_order))
    if unknown:
        raise ValueError(f"token result contains unknown item identifiers: {unknown!r}")
    if ordered_ids != canonical_order[: len(ordered_ids)]:
        raise ValueError("token result item order differs from the seed-0 selection")

    items = token_result.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise TypeError("token result items must be a list")
    item_ids: list[str] = []
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("item_id"), str):
            raise TypeError("token result items must contain item identifiers")
        item_ids.append(item["item_id"])
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("token result contains duplicate item identifiers")
    item_id_set = set(item_ids)
    ordered_set = set(ordered_ids)
    missing = [item_id for item_id in ordered_ids if item_id not in item_id_set]
    extra = [item_id for item_id in item_ids if item_id not in ordered_set]
    if missing or extra or tuple(item_ids) != ordered_ids:
        raise ValueError(
            "token result item coverage or order differs from persisted selection; "
            f"missing={missing!r}, extra={extra!r}"
        )
    return ordered_ids


def _selected_records(
    records: Sequence[AsdivRecord], ordered_ids: Sequence[str], count: int
) -> tuple[AsdivRecord, ...]:
    records_by_id = {record.id: record for record in records}
    return tuple(records_by_id[item_id] for item_id in ordered_ids[:count])


def _selection_identity(token_selection: Mapping[str, Any], records: Sequence[AsdivRecord]) -> dict[str, Any]:
    result = copy.deepcopy(dict(token_selection))
    ids = [record.id for record in records]
    result.update(
        {
            "split": "test",
            "seed": 0,
            "problem_count": len(ids),
            "ordered_item_ids": ids,
            "identity_sha256": asdiv_a_selection_identity(
                records=records,
                split="test",
                seed=0,
                problem_count=len(ids),
                revision=ASDIV_REVISION,
            ),
        }
    )
    return result


def _seed_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_eval(
    *,
    checkpoint_path: Path | None,
    seeds: list[int],
    development_limit: int | None,
    slot_count: int,
    num_steps: int,
    device: str,
    token_result: Mapping[str, Any],
    records: Sequence[AsdivRecord],
    subject_loader: Callable[..., Any] = load_subject,
    item_evaluator: Callable[[Any, AsdivRecord], Mapping[str, Any]] = evaluate_item,
    backbone_loader: Callable[..., Any] = load_frozen_qwen_backbone,
    runtime_configurer: Callable[[], None] = configure_deterministic_runtime,
    on_item: Callable[[LatentProgress], None] | None = None,
) -> dict[str, Any]:
    """Evaluate each seed against one validated, persisted item order."""
    runtime_configurer()
    source_records = _validate_records(records)
    ordered_ids = _token_order(token_result, source_records)
    _validate_development_limit(development_limit, len(ordered_ids))
    selected_count = len(ordered_ids) if development_limit is None else development_limit
    selection_records = _selected_records(source_records, ordered_ids, selected_count)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    if checkpoint_path is None and subject_loader is load_subject:
        raise ValueError("official latent evaluation requires a version-2 .ckpt checkpoint")

    checkpoint_sha256 = _checkpoint_digest(checkpoint_path)
    shared_backbone = backbone_loader(device=device) if subject_loader is load_subject else None
    runs: list[dict[str, Any]] = []
    canonical_rendered: list[dict[str, Any]] | None = None
    global_correct = 0
    global_done = 0
    global_count = len(seeds) * len(selection_records)
    for seed_index, seed in enumerate(seeds, start=1):
        _seed_runtime(seed)
        loader_kwargs = {"backbone": shared_backbone} if shared_backbone is not None else {}
        subject = subject_loader(checkpoint_path, slot_count, num_steps, device, **loader_kwargs)
        items: list[dict[str, Any]] = []
        rendered_inputs: list[dict[str, Any]] = []
        correct = 0
        for item_index, record in enumerate(selection_records, start=1):
            evaluated = item_evaluator(subject, record)
            if not isinstance(evaluated, Mapping):
                raise TypeError("item evaluator must return a mapping")
            prediction = evaluated.get("prediction")
            if not isinstance(prediction, str):
                raise TypeError("item evaluator prediction must be a string")
            input_ids = evaluated.get("input_ids")
            if not isinstance(input_ids, list) or any(
                isinstance(item, bool) or not isinstance(item, int) for item in input_ids
            ):
                raise ValueError("item evaluator input_ids must be a list of integers")
            rendered_inputs.append({"item_id": record.id, "input_ids": list(input_ids)})
            is_correct = score_numerical_answer(prediction, record.target)
            correct += int(is_correct)
            global_correct += int(is_correct)
            global_done += 1
            items.append(
                {
                    "item_id": record.id,
                    "prediction": prediction,
                    "target": record.target,
                    "correct": is_correct,
                }
            )
            if on_item is not None:
                on_item(
                    LatentProgress(
                        seed=seed,
                        seed_index=seed_index,
                        seed_count=len(seeds),
                        item_index=item_index,
                        item_count=len(selection_records),
                        seed_correct=correct,
                        global_done=global_done,
                        global_count=global_count,
                        global_correct=global_correct,
                    )
                )
        if canonical_rendered is None:
            canonical_rendered = rendered_inputs
        elif rendered_inputs != canonical_rendered:
            raise ValueError("Qwen rendered input IDs differ across evaluation seeds")
        runs.append({"seed": seed, "items": items, "correct": correct, "total": len(items)})

    assert canonical_rendered is not None
    token_identity = token_result["identity"]
    assert isinstance(token_identity, Mapping)
    token_selection = token_identity["selection"]
    assert isinstance(token_selection, Mapping)
    identity = copy.deepcopy(dict(token_identity))
    identity.update(
        {
            "selection": _selection_identity(token_selection, selection_records),
            "rendered_inputs_sha256": hashlib.sha256(canonical_json_bytes(canonical_rendered)).hexdigest(),
            "checkpoint_sha256": checkpoint_sha256,
            "seeds": list(seeds),
            "slot_count": slot_count,
            "latent_step_count": num_steps,
            "tap_tuple_index": LATENT_TAP_LAYER,
            "development_only": bool(token_identity.get("development_only"))
            or development_limit is not None
            or len(selection_records) != FULL_TEST_COUNT,
        }
    )
    return {
        "schema_version": 2,
        "identity": identity,
        "checkpoint_sha256": checkpoint_sha256,
        "seeds": list(seeds),
        "runs": runs,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0 latent evaluation on persisted Calc-ASDiv_A/Qwen inputs.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the safe version-2 .ckpt file under evaluation.",
    )
    parser.add_argument(
        "--token-result",
        type=Path,
        default=DEFAULT_TOKEN_RESULT,
        help="Local version-2 token baseline result whose persisted selection is reused.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated evaluation seeds. The gate requires exactly 0,1,2,3,4.",
    )
    parser.add_argument(
        "--development-limit",
        type=int,
        default=None,
        help="Development-only prefix length from the persisted token selection.",
    )
    parser.add_argument("--slot-count", type=int, default=DEFAULT_SLOT_COUNT)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    configure_deterministic_runtime()
    args = _parse_args()
    seeds = [int(token) for token in args.seeds.split(",") if token.strip()]
    records = tuple(load_asdiv_a_record_split("test"))
    prepared = token_cot_baseline.prepare_baseline_request(
        device=args.device,
        development_limit=args.development_limit,
        records=records,
        runtime_configurer=lambda: None,
    )
    token_result = result_cache.read_token_result(
        args.token_result,
        expected_identity=prepared.identity,
        records=prepared.selection.records,
    )
    result = run_eval(
        checkpoint_path=args.checkpoint,
        seeds=seeds,
        development_limit=args.development_limit,
        slot_count=args.slot_count,
        num_steps=args.num_steps,
        device=args.device,
        token_result=token_result,
        records=records,
        backbone_loader=lambda **_kwargs: prepared.backbone,
        runtime_configurer=lambda: None,
    )
    result_cache.write_gate_result(args.output, result)
    for run in result["runs"]:
        accuracy = run["correct"] / run["total"] if run["total"] else 0.0
        print(f"seed={run['seed']} accuracy={accuracy:.4f} ({run['correct']}/{run['total']})")
    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
