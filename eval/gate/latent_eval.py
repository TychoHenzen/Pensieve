"""Gate 8.2: latent reasoning evaluation on GSM8K.

Loads a `LatentCoreSubject` from a `train.run_training` checkpoint (or, if
none is given, an untrained subject) and evaluates it on the GSM8K test
split across multiple seeds. Multiple seeds matter here because the
GSM8K generator reshuffles problems per seed
(`eval.stream.generators.gsm8k`'s `derive(seed, "order")`), so a single
seed's accuracy could be an artifact of problem order rather than the
subject's reasoning. Accuracy is reported as a mean and standard
deviation across seeds, for comparison against the token-CoT baseline
from `eval.gate.token_cot_baseline`.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from eval.stream.config import StreamConfig
from eval.stream.events import Probe
from eval.stream.generators.gsm8k import GSM8KGenerator
from eval.subject.isolation import isolated_answer
from eval.subjects.latent_core import DEFAULT_NUM_STEPS, LatentCoreSubject
from workspace.concept_slots import DEFAULT_SLOT_COUNT

DEFAULT_OUTPUT = Path("gate_results/latent_eval.json")
DEFAULT_BASELINE_PATH = Path("gate_results/token_cot.json")
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
MIN_SEEDS = 5


def load_subject(
    checkpoint_path: Path | None,
    slot_count: int,
    num_steps: int,
    device: str,
) -> LatentCoreSubject:
    """Build a `LatentCoreSubject`, loading trained weights if `checkpoint_path` is given.

    The checkpoint format matches `train.run_training._save_checkpoint`:
    state dicts for the encoder's projection, the slot-query tensor and
    attention temperature, plus the latent loop's projection and layer norm.
    """
    subject = LatentCoreSubject(slot_count=slot_count, num_steps=num_steps, device=device)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        subject.encoder.projection.load_state_dict(state["encoder_projection"])
        with torch.no_grad():
            subject.encoder.slot_queries.copy_(state["encoder_slot_queries"])
            if "encoder_attn_log_temp" in state:
                subject.encoder.attn_log_temp.copy_(state["encoder_attn_log_temp"])
        subject.latent_loop.projection.load_state_dict(state["latent_loop_projection"])
        if "latent_loop_layer_norm" in state:
            subject.latent_loop.layer_norm.load_state_dict(state["latent_loop_layer_norm"])
    return subject


def run_seed(
    subject: LatentCoreSubject,
    seed: int,
    problem_count: int | None,
    on_problem: None | object = None,
) -> float:
    """Drive `subject` through one seed of the GSM8K test stream and return accuracy.

    `on_problem`, when provided, is called after each probe with
    (index, is_correct, running_correct, running_total).
    """
    params: dict[str, Any] = {"split": "test"}
    if problem_count is not None:
        params["problem_count"] = problem_count
    config = StreamConfig(generator="gsm8k", params=params)

    correct = 0
    total = 0
    for item in GSM8KGenerator().generate(config, seed):
        event = item.event
        if isinstance(event, Probe):
            answer = isolated_answer(subject, event)
            assert item.truth is not None
            is_correct = str(item.truth.answer) == answer
            if is_correct:
                correct += 1
            total += 1
            if on_problem is not None:
                on_problem(total - 1, is_correct, correct, total)  # type: ignore[operator]
        else:
            subject.observe(event)
    return correct / total if total else 0.0


def run_eval(
    checkpoint_path: Path | None,
    seeds: list[int],
    problem_count: int | None,
    slot_count: int,
    num_steps: int,
    device: str,
) -> dict[str, Any]:
    accuracies: list[float] = []
    for seed in seeds:
        subject = load_subject(checkpoint_path, slot_count, num_steps, device)
        accuracy = run_seed(subject, seed, problem_count)
        accuracies.append(accuracy)
        print(f"seed={seed} accuracy={accuracy:.4f}")

    mean = statistics.mean(accuracies) if accuracies else 0.0
    std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0

    return {
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "slot_count": slot_count,
        "num_steps": num_steps,
        "split": "test",
        "seeds": seeds,
        "accuracies": accuracies,
        "mean": mean,
        "std": std,
    }


def _compare_to_baseline(result: dict[str, Any], baseline_path: Path) -> None:
    if not baseline_path.exists():
        print(f"no baseline found at {baseline_path}; skipping comparison")
        return
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_accuracy = baseline["accuracy"]
    print(
        f"latent mean accuracy={result['mean']:.4f} vs "
        f"token-CoT baseline accuracy={baseline_accuracy:.4f} "
        f"({'meets or exceeds' if result['mean'] >= baseline_accuracy else 'below'} baseline)"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate 8.2: latent reasoning evaluation on GSM8K."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a train.run_training checkpoint; omit to evaluate an untrained subject.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated list of integer seeds (at least 5 required to pass the gate).",
    )
    parser.add_argument("--problem-count", type=int, default=None)
    parser.add_argument("--slot-count", type=int, default=DEFAULT_SLOT_COUNT)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = [int(token) for token in args.seeds.split(",") if token.strip() != ""]
    if len(seeds) < MIN_SEEDS:
        print(f"WARNING: only {len(seeds)} seeds given; the gate requires at least {MIN_SEEDS}.")

    result = run_eval(
        args.checkpoint,
        seeds,
        args.problem_count,
        args.slot_count,
        args.num_steps,
        args.device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"latent eval mean accuracy: {result['mean']:.4f} (std={result['std']:.4f})")
    _compare_to_baseline(result, args.baseline)
    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
