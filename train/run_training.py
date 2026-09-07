"""Gradient training on filtered Calc-ASDiv_A with the frozen Qwen backbone.

Loads all 570 records in the pinned Calc-ASDiv_A train order. The
trainer uses the frozen Qwen/Qwen2.5-0.5B-Instruct backbone and frozen MiniLM.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import torch

from eval.stage0_identity import training_identity
from train.stage0_data import load_stage0_dataset, training_examples
from train.standalone_checkpoint import (
    build_checkpoint,
    checkpoint_epoch,
    configure_deterministic_runtime,
    latest_epoch_checkpoint,
    load_checkpoint,
    restore_checkpoint,
    runtime_identity,
    save_checkpoint,
    selection_metadata,
    validate_compatibility,
)
from train.trainer import DEFAULT_LR, DEFAULT_NUM_STEPS, LatentCoreTrainer, StepResult
from workspace.concept_slots import DEFAULT_SLOT_COUNT


def _ts() -> str:
    t = time.localtime()
    return f"[{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}]"


def _log(msg: str) -> None:
    print(f"{_ts()} {msg}", flush=True)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the latent-core subject on filtered Calc-ASDiv_A with the "
            "frozen Qwen/Qwen2.5-0.5B-Instruct backbone."
        )
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--slot-count", type=int, default=DEFAULT_SLOT_COUNT)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--save-dir", type=str, default="checkpoints/")
    parser.add_argument(
        "--problem-count",
        type=int,
        default=None,
        help="Limit Calc-ASDiv_A training records. Default uses all 570 train records.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print a progress line every N examples within an epoch.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume from. 'latest' finds the highest epoch in --save-dir.",
    )
    return parser.parse_args()


def _find_latest_checkpoint(save_dir: str) -> str | None:
    """Find the checkpoint with the highest epoch number in save_dir."""
    path = latest_epoch_checkpoint(save_dir)
    return str(path) if path is not None else None


def _load_checkpoint(
    path: str,
    *,
    identity: dict[str, Any],
    selections: dict[str, Any],
    learning_rate: float,
):
    """Validate a safe checkpoint before constructing a trainer."""
    _log(f"loading checkpoint: {path}")
    checkpoint = load_checkpoint(path, expected_mode="gradient")
    validate_compatibility(
        checkpoint,
        identity=identity,
        selections=selections,
        learning_rate=learning_rate,
    )
    epoch = checkpoint_epoch(path)
    _log(f"resumed from epoch {epoch}")
    return checkpoint, epoch


def _load_dataset(problem_count: int | None) -> list[tuple[str, str]]:
    _log("loading pinned Calc-ASDiv_A partitions...")
    t0 = time.monotonic()
    stage0_dataset = load_stage0_dataset()
    dataset = training_examples(
        stage0_dataset,
        mode="gradient",
        epoch=1,
        problem_count=problem_count,
    )
    _log(f"loaded {len(dataset)} problems ({_format_duration(time.monotonic() - t0)})")
    return dataset


def _load_dataset_context(
    problem_count: int | None,
) -> tuple[list[tuple[str, str]], dict[str, Any], dict[str, Any]]:
    _log("loading pinned Calc-ASDiv_A partitions...")
    t0 = time.monotonic()
    stage0_dataset = load_stage0_dataset()
    dataset = training_examples(stage0_dataset, mode="gradient", epoch=1, problem_count=problem_count)
    train = selection_metadata(stage0_dataset.train_selection, problem_count)
    held_out = selection_metadata(stage0_dataset.held_out_selection, None)
    selections = {"train": train, "held_out": held_out}
    identity = training_identity(runtime=runtime_identity(), held_out_item_ids=held_out["ordered_item_ids"])
    _log(f"loaded {len(dataset)} problems ({_format_duration(time.monotonic() - t0)})")
    return dataset, identity, selections


def _save_checkpoint(
    trainer: LatentCoreTrainer,
    save_dir: str,
    epoch: int,
    *,
    identity: dict[str, Any],
    selections: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"epoch-{epoch}.ckpt")
    checkpoint = build_checkpoint(
        mode="gradient",
        identity=identity,
        selections=selections,
        model_state=trainer.state.trainable_params,
        optimizer=trainer.optimizer,
        metrics=metrics,
    )
    save_checkpoint(checkpoint_path, checkpoint)
    return checkpoint_path


def main() -> None:
    configure_deterministic_runtime()
    args = _parse_args()

    _log(f"device={args.device}")
    _log(f"config: epochs={args.epochs} slots={args.slot_count} steps={args.num_steps} lr={args.lr}")

    dataset, checkpoint_identity, selections = _load_dataset_context(args.problem_count)

    resumed = None
    start_epoch = 0
    if args.resume is not None:
        path = _find_latest_checkpoint(args.save_dir) if args.resume == "latest" else args.resume
        if path is None:
            _log(f"no checkpoints found in {args.save_dir}, starting from scratch")
        else:
            resumed, start_epoch = _load_checkpoint(
                path,
                identity=checkpoint_identity,
                selections=selections,
                learning_rate=args.lr,
            )

    _log("building trainer (loading frozen Qwen/Qwen2.5-0.5B-Instruct + frozen MiniLM)...")
    t0 = time.monotonic()
    trainer = LatentCoreTrainer(
        slot_count=args.slot_count,
        num_steps=args.num_steps,
        lr=args.lr,
        device=args.device,
    )
    _log(
        f"trainer ready ({_format_duration(time.monotonic() - t0)}), "
        f"trainable_params={trainer.trainable_param_count():,}"
    )

    if resumed is not None:
        restore_checkpoint(
            resumed,
            model_parameters=trainer.state.trainable_params,
            optimizer=trainer.optimizer,
        )

    remaining_epochs = args.epochs - start_epoch
    if remaining_epochs <= 0:
        _log(f"already at epoch {start_epoch}/{args.epochs}, nothing to do")
        return

    _log(f"{'=' * 60}")
    _log(f"  epochs {start_epoch + 1} to {args.epochs} ({remaining_epochs} remaining) x {len(dataset)} examples")
    _log(f"{'=' * 60}")

    epoch_losses: list[float] = []
    training_start = time.monotonic()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.monotonic()
        recent_losses: list[float] = []
        recent_vars: list[float] = []

        def _on_step(
            step: int,
            total: int,
            result: StepResult,
            _recent_losses: list[float] = recent_losses,
            _recent_vars: list[float] = recent_vars,
            _epoch_start: float = epoch_start,
            _epoch: int = epoch,
        ) -> None:
            loss = getattr(result, "total_objective", getattr(result, "loss", None))
            variance = getattr(result, "shared_variance", getattr(result, "variance", None))
            if not isinstance(loss, (int, float)) or isinstance(loss, bool):
                raise TypeError("gradient trainer result must expose a numeric objective")
            if not isinstance(variance, (int, float)) or isinstance(variance, bool):
                raise TypeError("gradient trainer result must expose numeric variance")
            loss = float(loss)
            variance = float(variance)
            _recent_losses.append(loss)
            _recent_vars.append(variance)
            if (step + 1) % args.log_every == 0 or step + 1 == total:
                elapsed = time.monotonic() - _epoch_start
                per_example = elapsed / (step + 1)
                remaining = per_example * (total - step - 1)
                avg_loss = sum(_recent_losses) / len(_recent_losses)
                avg_var = sum(_recent_vars) / len(_recent_vars)
                _log(
                    f"  epoch {_epoch}/{args.epochs} "
                    f"[{step + 1}/{total}] "
                    f"loss={loss:.4f} avg={avg_loss:.4f} "
                    f"var={variance:.6f} avg_var={avg_var:.6f} "
                    f"ETA {_format_duration(remaining)}"
                )

        stats = trainer.train_epoch(dataset, on_step=_on_step)
        epoch_dt = time.monotonic() - epoch_start
        epoch_losses.append(stats.avg_loss)

        _log(
            f"epoch {epoch}/{args.epochs} done ({_format_duration(epoch_dt)}) "
            f"loss={stats.avg_loss:.4f} "
            f"variance={stats.avg_variance:.6f} [{stats.min_variance:.6f}, {stats.max_variance:.6f}] "
            f"covariance={stats.avg_covariance:.6f}"
        )

        if stats.avg_variance <= 1e-8:
            _log(f"WARNING: slots are collapsing. avg_variance={stats.avg_variance:.8f} across {stats.steps} steps")

        checkpoint_path = _save_checkpoint(
            trainer,
            args.save_dir,
            epoch,
            identity=checkpoint_identity,
            selections=selections,
            metrics={
                "avg_loss": stats.avg_loss,
                "avg_variance": stats.avg_variance,
                "min_variance": stats.min_variance,
                "max_variance": stats.max_variance,
                "avg_covariance": stats.avg_covariance,
            },
        )
        _log(f"saved: {checkpoint_path}")

        epochs_done = epoch - start_epoch
        if epochs_done > 1:
            total_elapsed = time.monotonic() - training_start
            per_epoch = total_elapsed / epochs_done
            epochs_left = args.epochs - epoch
            _log(f"  training progress: {epoch}/{args.epochs} epochs, ETA {_format_duration(per_epoch * epochs_left)}")

    _log(f"{'=' * 60}")
    if epoch_losses:
        total_time = time.monotonic() - training_start
        _log(f"  training complete in {_format_duration(total_time)}")
        _log(f"  first_epoch_loss={epoch_losses[0]:.4f}")
        _log(f"  last_epoch_loss={epoch_losses[-1]:.4f}")
        delta = epoch_losses[-1] - epoch_losses[0]
        sign = "+" if delta >= 0 else ""
        _log(f"  delta={sign}{delta:.4f} decreased={'yes' if delta < 0 else 'no'}")
    _log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
