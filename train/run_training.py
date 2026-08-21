"""CLI entry point for gradient training on the Stage 0 Calc-MAWPS order.

Loads the pinned filtered Calc-MAWPS train split, runs `LatentCoreTrainer` for
a configurable number of epochs, and saves a checkpoint after every epoch.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import time

import torch

from train.stage0_data import load_stage0_dataset, training_examples
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
        description="Train the latent-core subject on filtered Calc-MAWPS."
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
        help="Number of Calc-MAWPS train problems to use; default uses all 1,089.",
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
    pattern = os.path.join(save_dir, "epoch-*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    epoch_re = re.compile(r"epoch-(\d+)\.pt$")
    best_epoch = -1
    best_path = None
    for path in files:
        m = epoch_re.search(path)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = path
    return best_path


def _load_checkpoint(trainer: LatentCoreTrainer, path: str, device: str) -> int:
    """Load a checkpoint into the trainer. Returns the epoch it was saved at."""
    _log(f"loading checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    trainer.encoder.projection.load_state_dict(state["encoder_projection"])
    with torch.no_grad():
        trainer.encoder.slot_queries.copy_(state["encoder_slot_queries"])
        if "encoder_attn_log_temp" in state:
            trainer.encoder.attn_log_temp.copy_(state["encoder_attn_log_temp"])
    trainer.latent_loop.projection.load_state_dict(state["latent_loop_projection"])
    if "latent_loop_layer_norm" in state:
        trainer.latent_loop.layer_norm.load_state_dict(state["latent_loop_layer_norm"])
    trainer.optimizer.load_state_dict(state["optimizer"])
    epoch = state["epoch"]
    _log(f"resumed from epoch {epoch}")
    return epoch


def _load_dataset(problem_count: int | None) -> list[tuple[str, str]]:
    _log("loading pinned filtered Calc-MAWPS train and validation splits...")
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


def _save_checkpoint(trainer: LatentCoreTrainer, save_dir: str, epoch: int) -> str:
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"epoch-{epoch}.pt")
    state = {
        "encoder_projection": trainer.encoder.projection.state_dict(),
        "encoder_slot_queries": trainer.encoder.slot_queries,
        "encoder_attn_log_temp": trainer.encoder.attn_log_temp,
        "latent_loop_projection": trainer.latent_loop.projection.state_dict(),
        "latent_loop_layer_norm": trainer.latent_loop.layer_norm.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, checkpoint_path)
    return checkpoint_path


def main() -> None:
    args = _parse_args()

    _log(f"device={args.device}")
    _log(f"config: epochs={args.epochs} slots={args.slot_count} "
         f"steps={args.num_steps} lr={args.lr}")

    dataset = _load_dataset(args.problem_count)

    _log("building trainer (loading Pythia-160M + MiniLM)...")
    t0 = time.monotonic()
    trainer = LatentCoreTrainer(
        slot_count=args.slot_count,
        num_steps=args.num_steps,
        lr=args.lr,
        device=args.device,
    )
    _log(f"trainer ready ({_format_duration(time.monotonic() - t0)}), "
         f"trainable_params={trainer.trainable_param_count():,}")

    start_epoch = 0
    if args.resume is not None:
        if args.resume == "latest":
            path = _find_latest_checkpoint(args.save_dir)
            if path is None:
                _log(f"no checkpoints found in {args.save_dir}, starting from scratch")
            else:
                start_epoch = _load_checkpoint(trainer, path, args.device)
        else:
            start_epoch = _load_checkpoint(trainer, args.resume, args.device)

    remaining_epochs = args.epochs - start_epoch
    if remaining_epochs <= 0:
        _log(f"already at epoch {start_epoch}/{args.epochs}, nothing to do")
        return

    _log(f"{'=' * 60}")
    _log(f"  epochs {start_epoch + 1} to {args.epochs} ({remaining_epochs} remaining) "
         f"x {len(dataset)} examples")
    _log(f"{'=' * 60}")

    epoch_losses: list[float] = []
    training_start = time.monotonic()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.monotonic()
        recent_losses: list[float] = []
        recent_vars: list[float] = []

        def _on_step(step: int, total: int, result: StepResult) -> None:
            recent_losses.append(result.loss)
            recent_vars.append(result.variance)
            if (step + 1) % args.log_every == 0 or step + 1 == total:
                elapsed = time.monotonic() - epoch_start
                per_example = elapsed / (step + 1)
                remaining = per_example * (total - step - 1)
                avg_loss = sum(recent_losses) / len(recent_losses)
                avg_var = sum(recent_vars) / len(recent_vars)
                _log(f"  epoch {epoch}/{args.epochs} "
                     f"[{step + 1}/{total}] "
                     f"loss={result.loss:.4f} avg={avg_loss:.4f} "
                     f"var={result.variance:.6f} avg_var={avg_var:.6f} "
                     f"ETA {_format_duration(remaining)}")

        stats = trainer.train_epoch(dataset, on_step=_on_step)
        epoch_dt = time.monotonic() - epoch_start
        epoch_losses.append(stats.avg_loss)

        _log(f"epoch {epoch}/{args.epochs} done ({_format_duration(epoch_dt)}) "
             f"loss={stats.avg_loss:.4f} "
             f"variance={stats.avg_variance:.6f} [{stats.min_variance:.6f}, {stats.max_variance:.6f}] "
             f"covariance={stats.avg_covariance:.6f}")

        if stats.avg_variance <= 1e-8:
            _log(f"WARNING: slots are collapsing. "
                 f"avg_variance={stats.avg_variance:.8f} across {stats.steps} steps")

        checkpoint_path = _save_checkpoint(trainer, args.save_dir, epoch)
        _log(f"saved: {checkpoint_path}")

        epochs_done = epoch - start_epoch
        if epochs_done > 1:
            total_elapsed = time.monotonic() - training_start
            per_epoch = total_elapsed / epochs_done
            epochs_left = args.epochs - epoch
            _log(f"  training progress: {epoch}/{args.epochs} epochs, "
                 f"ETA {_format_duration(per_epoch * epochs_left)}")

    _log(f"{'=' * 60}")
    if epoch_losses:
        total_time = time.monotonic() - training_start
        _log(f"  training complete in {_format_duration(total_time)}")
        _log(f"  first_epoch_loss={epoch_losses[0]:.4f}")
        _log(f"  last_epoch_loss={epoch_losses[-1]:.4f}")
        delta = epoch_losses[-1] - epoch_losses[0]
        sign = "+" if delta >= 0 else ""
        _log(f"  delta={sign}{delta:.4f} "
             f"decreased={'yes' if delta < 0 else 'no'}")
    _log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
