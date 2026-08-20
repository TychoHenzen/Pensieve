"""CLI entry point: trains the latent core using EGGROLL evolution strategies.

Same dataset and architecture as run_training.py, but replaces gradient
descent with evolution strategies. Low-rank perturbations with antithetic
sampling estimate the parameter update direction without backpropagation,
bypassing gradient attenuation through the frozen Pythia backbone.
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from eval.stream.generators.gsm8k import _extract_answer, _load_split
from train.eggroll_trainer import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_NUM_STEPS,
    DEFAULT_POP_SIZE,
    DEFAULT_RANK,
    DEFAULT_SIGMA,
    DEFAULT_VARIANCE_WEIGHT,
    EggrollTrainer,
    StepResult,
)
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
        description="Train the latent core using EGGROLL evolution strategies."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--slot-count", type=int, default=DEFAULT_SLOT_COUNT)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
        help="Perturbed candidates evaluated together. Higher is faster but uses more VRAM.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        dest="use_amp",
        help="Use CUDA mixed precision. Benchmark this before long runs.",
    )
    parser.add_argument(
        "--variance-weight", type=float, default=DEFAULT_VARIANCE_WEIGHT
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--save-dir", type=str, default="checkpoints/eggroll/")
    parser.add_argument(
        "--problem-count",
        type=int,
        default=None,
        help="Limit GSM8K train problems. Default uses the full split.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def _load_dataset(problem_count: int | None) -> list[tuple[str, str]]:
    _log("loading GSM8K train split...")
    t0 = time.monotonic()
    problems = _load_split("train")
    if problem_count is not None:
        problems = problems[:problem_count]
    dataset = [
        (item["question"], _extract_answer(item["answer"])) for item in problems
    ]
    _log(f"loaded {len(dataset)} problems ({_format_duration(time.monotonic() - t0)})")
    return dataset


def _save_checkpoint(
    trainer: EggrollTrainer, save_dir: str, epoch: int
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"epoch-{epoch}.pt")
    state = {
        "encoder_projection": trainer.encoder.projection.state_dict(),
        "encoder_slot_queries": trainer.encoder.slot_queries,
        "latent_loop_projection": trainer.latent_loop.projection.state_dict(),
        "latent_loop_layer_norm": trainer.latent_loop.layer_norm.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "epoch": epoch,
        "pop_size": trainer.pop_size,
        "sigma": trainer.sigma,
        "rank": trainer.rank,
        "variance_weight": trainer.variance_weight,
        "eval_batch_size": trainer.eval_batch_size,
        "use_amp": trainer.use_amp,
    }
    torch.save(state, checkpoint_path)
    return checkpoint_path


def main() -> None:
    args = _parse_args()

    _log(f"device={args.device}")
    _log(
        f"config: epochs={args.epochs} slots={args.slot_count} "
        f"steps={args.num_steps} pop={args.pop_size} "
        f"sigma={args.sigma} lr={args.lr} rank={args.rank} "
        f"var_weight={args.variance_weight} eval_batch={args.eval_batch_size} "
        f"amp={args.use_amp}"
    )

    dataset = _load_dataset(args.problem_count)

    _log("building trainer (loading Pythia-160M + MiniLM)...")
    t0 = time.monotonic()
    trainer = EggrollTrainer(
        slot_count=args.slot_count,
        num_steps=args.num_steps,
        pop_size=args.pop_size,
        sigma=args.sigma,
        lr=args.lr,
        rank=args.rank,
        variance_weight=args.variance_weight,
        eval_batch_size=args.eval_batch_size,
        use_amp=args.use_amp,
        device=args.device,
    )
    _log(
        f"trainer ready ({_format_duration(time.monotonic() - t0)}), "
        f"trainable_params={trainer.trainable_param_count():,}"
    )

    _log(f"{'=' * 60}")
    _log(f"  EGGROLL ES training: {args.epochs} epochs x {len(dataset)} examples")
    _log(f"  population={args.pop_size} (antithetic pairs)")
    _log(f"{'=' * 60}")

    epoch_losses: list[float] = []
    training_start = time.monotonic()

    for epoch in range(1, args.epochs + 1):
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
                _log(
                    f"  epoch {epoch}/{args.epochs} "
                    f"[{step + 1}/{total}] "
                    f"loss={result.loss:.4f} avg={avg_loss:.4f} "
                    f"var={result.variance:.6f} avg_var={avg_var:.6f} "
                    f"best_fit={result.best_fitness:.4f} "
                    f"ETA {_format_duration(remaining)}"
                )

        stats = trainer.train_epoch(dataset, on_step=_on_step)
        epoch_dt = time.monotonic() - epoch_start
        epoch_losses.append(stats.avg_loss)

        _log(
            f"epoch {epoch}/{args.epochs} done ({_format_duration(epoch_dt)}) "
            f"loss={stats.avg_loss:.4f} "
            f"variance={stats.avg_variance:.6f} "
            f"[{stats.min_variance:.6f}, {stats.max_variance:.6f}]"
        )

        checkpoint_path = _save_checkpoint(trainer, args.save_dir, epoch)
        _log(f"saved: {checkpoint_path}")

        if epoch > 1:
            total_elapsed = time.monotonic() - training_start
            per_epoch = total_elapsed / epoch
            epochs_left = args.epochs - epoch
            _log(
                f"  ETA {_format_duration(per_epoch * epochs_left)} "
                f"for remaining {epochs_left} epochs"
            )

    _log(f"{'=' * 60}")
    if epoch_losses:
        total_time = time.monotonic() - training_start
        _log(f"  training complete in {_format_duration(total_time)}")
        _log(f"  first_epoch_loss={epoch_losses[0]:.4f}")
        _log(f"  last_epoch_loss={epoch_losses[-1]:.4f}")
        delta = epoch_losses[-1] - epoch_losses[0]
        sign = "+" if delta >= 0 else ""
        _log(f"  delta={sign}{delta:.4f}")
    _log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
