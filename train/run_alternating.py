"""CLI entry point for fixed-budget alternating training experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch

from train.alternating_config import validate_scheduler_config
from train.alternating_evaluation import DEFAULT_EVAL_PROBLEM_COUNT
from train.eggroll_trainer import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_LR as DEFAULT_EGGROLL_LR,
    DEFAULT_NUM_STEPS,
    DEFAULT_POP_SIZE,
    DEFAULT_RANK,
    DEFAULT_SIGMA,
    DEFAULT_VARIANCE_WEIGHT,
)
from train.trainer import DEFAULT_LR as DEFAULT_GRADIENT_LR
from workspace.concept_slots import DEFAULT_SLOT_COUNT


DEFAULT_EPOCHS = 5
DEFAULT_PHASE_STEPS = 500


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one shared latent core with alternating Eggroll and gradient phases."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--phase-steps", type=int, default=DEFAULT_PHASE_STEPS)
    parser.add_argument("--slot-count", type=int, default=DEFAULT_SLOT_COUNT)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--gradient-lr", type=float, default=DEFAULT_GRADIENT_LR)
    parser.add_argument("--eggroll-lr", type=float, default=DEFAULT_EGGROLL_LR)
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument(
        "--variance-weight", type=float, default=DEFAULT_VARIANCE_WEIGHT
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
        help="Perturbed candidates evaluated together.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        dest="use_amp",
        help="Use CUDA mixed precision for Eggroll candidate evaluation.",
    )
    parser.add_argument(
        "--eval-problem-count",
        type=int,
        default=DEFAULT_EVAL_PROBLEM_COUNT,
        help="Number of held-out GSM8K test problems used at each evaluation.",
    )
    parser.add_argument(
        "--problem-count",
        type=int,
        default=None,
        help="Limit GSM8K training problems. Default uses the full split.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="checkpoints/alternating/")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume from. 'latest' uses the largest global step.",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    validate_scheduler_config(phase_steps=args.phase_steps, epochs=args.epochs)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Validate command configuration before later run setup loads data or models."""
    _parse_args(argv)


if __name__ == "__main__":
    main()
