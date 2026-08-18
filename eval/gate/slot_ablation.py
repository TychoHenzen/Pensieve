"""Gate 8.3: slot count ablation on GSM8K.

Runs a `LatentCoreSubject` at each slot count in `SLOT_COUNTS`, from a
single workspace vector (`slot_count=1`) up to 64 slots, and records its
GSM8K test accuracy at each. The comparison this exists for is
single-vector versus multi-slot: if the workspace's structured, multi-
slot representation carries no advantage over one vector, `slot_count=1`
should score at least as well as the larger counts.

Each slot count optionally loads a matching trained checkpoint from
`--checkpoint-dir`, named `slot-<count>.pt`; a slot count with no such
file evaluates an untrained subject at that width.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from eval.gate.latent_eval import load_subject, run_seed

DEFAULT_OUTPUT = Path("gate_results/slot_ablation.json")
SLOT_COUNTS = [1, 4, 8, 16, 32, 64]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_NUM_STEPS = 8


def _checkpoint_for(checkpoint_dir: Path | None, slot_count: int) -> Path | None:
    if checkpoint_dir is None:
        return None
    candidate = checkpoint_dir / f"slot-{slot_count}.pt"
    return candidate if candidate.exists() else None


def run_ablation(
    slot_counts: list[int],
    seeds: list[int],
    problem_count: int | None,
    num_steps: int,
    device: str,
    checkpoint_dir: Path | None,
) -> dict[str, Any]:
    per_slot_count: dict[str, Any] = {}
    for slot_count in slot_counts:
        checkpoint_path = _checkpoint_for(checkpoint_dir, slot_count)
        accuracies: list[float] = []
        for seed in seeds:
            subject = load_subject(checkpoint_path, slot_count, num_steps, device)
            accuracy = run_seed(subject, seed, problem_count)
            accuracies.append(accuracy)
            print(f"slot_count={slot_count} seed={seed} accuracy={accuracy:.4f}")

        mean = statistics.mean(accuracies) if accuracies else 0.0
        std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        per_slot_count[str(slot_count)] = {
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "seeds": seeds,
            "accuracies": accuracies,
            "mean": mean,
            "std": std,
        }

    single_vector_mean = per_slot_count.get(str(SLOT_COUNTS[0]), {}).get("mean")
    multi_slot_means = {
        str(count): per_slot_count[str(count)]["mean"]
        for count in slot_counts
        if count != 1 and str(count) in per_slot_count
    }
    best_multi_slot_mean = max(multi_slot_means.values()) if multi_slot_means else None

    return {
        "slot_counts": slot_counts,
        "num_steps": num_steps,
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
    parser = argparse.ArgumentParser(description="Gate 8.3: slot count ablation on GSM8K.")
    parser.add_argument(
        "--slot-counts",
        default=",".join(str(count) for count in SLOT_COUNTS),
        help="Comma-separated list of slot counts to ablate.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument("--problem-count", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory holding per-slot-count checkpoints named slot-<count>.pt.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    slot_counts = [int(token) for token in args.slot_counts.split(",") if token.strip() != ""]
    seeds = [int(token) for token in args.seeds.split(",") if token.strip() != ""]

    result = run_ablation(
        slot_counts,
        seeds,
        args.problem_count,
        args.num_steps,
        args.device,
        args.checkpoint_dir,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for slot_count in slot_counts:
        stats = result["results"][str(slot_count)]
        print(f"slot_count={slot_count} mean={stats['mean']:.4f} std={stats['std']:.4f}")
    print(f"multi-slot advantage over single-vector: {result['multi_slot_advantage']}")
    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
