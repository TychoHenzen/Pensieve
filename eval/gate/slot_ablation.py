"""Slot-count ablation on persisted Calc-ASDiv_A/Qwen Stage 0 inputs."""

from __future__ import annotations

import argparse
import copy
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from eval.gate.latent_eval import run_eval
from eval.gate.result_cache import checkpoint_sha256, read_token_result, write_json_atomic
from eval.gate.token_cot_baseline import prepare_baseline_request
from eval.stream.generators.asdiv_a import AsdivRecord, load_asdiv_a_record_split
from eval.subjects.latent_core import DEFAULT_NUM_STEPS
from train.alternating_checkpoint import load_checkpoint
from train.standalone_checkpoint import configure_deterministic_runtime


DEFAULT_OUTPUT = Path("gate_results/asdiv_a_qwen/slot_ablation.json")
DEFAULT_TOKEN_RESULT = Path("gate_results/asdiv_a_qwen/token_cot.json")
SLOT_COUNTS = [1, 4, 8, 16, 32, 64]
DEFAULT_SEEDS = [0, 1, 2]


def parse_integer_list(raw: str, *, name: str, minimum: int) -> list[int]:
    """Parse a non-empty, duplicate-free comma-separated integer list."""
    singular = "seed" if name == "seeds" else "slot count"
    tokens = raw.split(",")
    if not raw.strip() or any(not token.strip() for token in tokens):
        raise ValueError(f"{name} must contain at least one {singular}")
    try:
        values = [int(token.strip()) for token in tokens]
    except ValueError as error:
        raise ValueError(f"{name} must contain only integers") from error
    if any(value < minimum for value in values):
        bound = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must contain only {bound} integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate values")
    return values


def _checkpoint_paths(checkpoint_dir: Path, slot_counts: Sequence[int]) -> dict[int, Path]:
    paths = {count: checkpoint_dir / f"slot-{count}.ckpt" for count in slot_counts}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"required safe checkpoint does not exist: {path}")
    return paths


def _validate_checkpoint_slot_count(
    path: Path,
    requested_slot_count: int,
    checkpoint_loader: Callable[[Path], Any],
) -> str:
    if path.suffix != ".ckpt":
        raise ValueError("slot ablation requires version-2 .ckpt checkpoints")
    checkpoint = checkpoint_loader(path)
    tensors = getattr(checkpoint, "tensors", None)
    if not isinstance(tensors, Mapping):
        raise ValueError(f"{path} does not contain a checkpoint tensor mapping")
    slot_queries = tensors.get("model.encoder.slot_queries")
    if not isinstance(slot_queries, torch.Tensor) or slot_queries.ndim != 2:
        raise ValueError(f"{path} must contain two-dimensional model.encoder.slot_queries")
    stored_slot_count = int(slot_queries.shape[0])
    if stored_slot_count != requested_slot_count:
        raise ValueError(
            f"{path} contains {stored_slot_count} slots, not requested slot count "
            f"{requested_slot_count}"
        )
    return checkpoint_sha256(path)


def _validated_runs(
    result: Mapping[str, Any],
    *,
    slot_count: int,
    seeds: Sequence[int],
    num_steps: int,
    checkpoint_digest: str,
    expected_total: int,
    expected_token_identity: Mapping[str, Any],
) -> list[dict[str, int | float]]:
    if result.get("schema_version") != 2:
        raise ValueError("latent result schema_version must be 2")
    if result.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError("latent result checkpoint digest is inconsistent")
    if result.get("seeds") != list(seeds):
        raise ValueError("latent result seed order is inconsistent")
    identity = result.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("latent result identity must be an object")
    expected_identity = copy.deepcopy(dict(expected_token_identity))
    expected_identity.update(
        {
            "slot_count": slot_count,
            "latent_step_count": num_steps,
            "checkpoint_sha256": checkpoint_digest,
            "seeds": list(seeds),
        }
    )
    if dict(identity) != expected_identity:
        raise ValueError("latent result identity is inconsistent with the token request")
    runs = result.get("runs")
    if not isinstance(runs, list) or len(runs) != len(seeds):
        raise ValueError("latent result run count is inconsistent")

    summaries: list[dict[str, int | float]] = []
    for index, (run, seed) in enumerate(zip(runs, seeds, strict=True)):
        if not isinstance(run, Mapping) or run.get("seed") != seed:
            raise ValueError("latent result seed order is inconsistent")
        correct = run.get("correct")
        total = run.get("total")
        if (
            isinstance(correct, bool)
            or not isinstance(correct, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total != expected_total
            or not 0 <= correct <= total
        ):
            raise ValueError(f"latent result run {index} counts are inconsistent")
        items = run.get("items")
        if not isinstance(items, list) or len(items) != total:
            raise ValueError(f"latent result run {index} item coverage is inconsistent")
        item_correct = [
            item.get("correct")
            for item in items
            if isinstance(item, Mapping) and isinstance(item.get("correct"), bool)
        ]
        if len(item_correct) != total or sum(item_correct) != correct:
            raise ValueError(f"latent result run {index} item scores are inconsistent")
        summaries.append(
            {
                "seed": seed,
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
            }
        )
    return summaries


def run_ablation(
    *,
    slot_counts: list[int],
    seeds: list[int],
    development_limit: int | None,
    num_steps: int,
    device: str,
    checkpoint_dir: Path,
    token_result_path: Path,
    records: Sequence[AsdivRecord] | None = None,
    record_loader: Callable[[str], Sequence[AsdivRecord]] = load_asdiv_a_record_split,
    baseline_preparer: Callable[..., Any] = prepare_baseline_request,
    token_reader: Callable[..., Mapping[str, Any]] = read_token_result,
    checkpoint_loader: Callable[[Path], Any] = load_checkpoint,
    latent_runner: Callable[..., Mapping[str, Any]] = run_eval,
) -> dict[str, Any]:
    """Evaluate validated per-width checkpoints against one persisted selection."""
    parsed_slots = parse_integer_list(
        ",".join(map(str, slot_counts)), name="slot counts", minimum=1
    )
    if parsed_slots != slot_counts:
        raise ValueError("slot counts must be an ordered list of positive integers")
    parsed_seeds = parse_integer_list(",".join(map(str, seeds)), name="seeds", minimum=0)
    if parsed_seeds != seeds:
        raise ValueError("seeds must be an ordered list of non-negative integers")
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")

    checkpoint_paths = _checkpoint_paths(checkpoint_dir, slot_counts)
    checkpoint_digests = {
        count: _validate_checkpoint_slot_count(checkpoint_paths[count], count, checkpoint_loader)
        for count in slot_counts
    }

    source_records = tuple(records) if records is not None else tuple(record_loader("test"))
    prepared = baseline_preparer(
        device=device,
        development_limit=development_limit,
        records=source_records,
        runtime_configurer=lambda: None,
    )
    token_result = token_reader(
        token_result_path,
        expected_identity=prepared.identity,
        records=prepared.selection.records,
    )
    selection = prepared.identity.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("prepared token identity selection must be an object")
    expected_total = selection.get("problem_count")
    if (
        isinstance(expected_total, bool)
        or not isinstance(expected_total, int)
        or expected_total < 1
    ):
        raise ValueError("prepared token selection problem_count must be positive")

    per_slot_count: dict[str, Any] = {}
    for slot_count in slot_counts:
        checkpoint_path = checkpoint_paths[slot_count]
        result = latent_runner(
            checkpoint_path=checkpoint_path,
            seeds=list(seeds),
            development_limit=development_limit,
            slot_count=slot_count,
            num_steps=num_steps,
            device=device,
            token_result=token_result,
            records=source_records,
            backbone_loader=lambda **_kwargs: prepared.backbone,
            runtime_configurer=lambda: None,
        )
        run_summaries = _validated_runs(
            result,
            slot_count=slot_count,
            seeds=seeds,
            num_steps=num_steps,
            checkpoint_digest=checkpoint_digests[slot_count],
            expected_total=expected_total,
            expected_token_identity=prepared.identity,
        )
        accuracies = [float(run["accuracy"]) for run in run_summaries]
        per_slot_count[str(slot_count)] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digests[slot_count],
            "runs": run_summaries,
            "mean": statistics.mean(accuracies),
            "std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        }

    single_vector = per_slot_count.get("1")
    single_vector_mean = single_vector["mean"] if single_vector is not None else None
    multi_slot_means = [per_slot_count[str(count)]["mean"] for count in slot_counts if count != 1]
    best_multi_slot_mean = max(multi_slot_means) if multi_slot_means else None
    return {
        "schema_version": 1,
        "identity": {
            "token": copy.deepcopy(prepared.identity),
            "selection": copy.deepcopy(dict(selection)),
        },
        "slot_counts": list(slot_counts),
        "seeds": list(seeds),
        "num_steps": num_steps,
        "development_only": bool(prepared.identity.get("development_only")),
        "results": per_slot_count,
        "single_vector_mean": single_vector_mean,
        "best_multi_slot_mean": best_multi_slot_mean,
        "multi_slot_advantage": (
            None
            if single_vector_mean is None or best_multi_slot_mean is None
            else best_multi_slot_mean > single_vector_mean
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 0 slot-count ablation on persisted Calc-ASDiv_A/Qwen inputs."
    )
    parser.add_argument(
        "--slot-counts",
        default=",".join(str(count) for count in SLOT_COUNTS),
        help="Comma-separated positive slot counts.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated non-negative evaluation seeds.",
    )
    parser.add_argument("--development-limit", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing exact slot-<count>.ckpt files.",
    )
    parser.add_argument(
        "--token-result",
        type=Path,
        default=DEFAULT_TOKEN_RESULT,
        help="Local version-2 token result whose persisted order is reused.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    configure_deterministic_runtime()
    args = _parse_args()
    slot_counts = parse_integer_list(args.slot_counts, name="slot counts", minimum=1)
    seeds = parse_integer_list(args.seeds, name="seeds", minimum=0)
    result = run_ablation(
        slot_counts=slot_counts,
        seeds=seeds,
        development_limit=args.development_limit,
        num_steps=args.num_steps,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        token_result_path=args.token_result,
    )
    write_json_atomic(args.output, result)
    for slot_count in slot_counts:
        stats = result["results"][str(slot_count)]
        print(f"slot_count={slot_count} mean={stats['mean']:.4f} std={stats['std']:.4f}")
    print(f"multi-slot advantage over single-vector: {result['multi_slot_advantage']}")
    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
