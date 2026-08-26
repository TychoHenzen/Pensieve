"""EGGROLL training on filtered Calc-ASDiv_A with the frozen Qwen backbone.

Same dataset and architecture as run_training.py, but replaces gradient
descent with evolution strategies. Low-rank perturbations with antithetic
sampling estimate the parameter update direction without backpropagation,
through the frozen Qwen/Qwen2.5-0.5B-Instruct backbone and frozen MiniLM.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence

import torch

from eval.stage0_identity import training_identity
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
    validate_eggroll_config,
)
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


def _training_progress_record(result: StepResult) -> dict[str, object]:
    """Return the structured per-update progress fields for standalone EGGROLL."""
    return {
        "update_method": result.position.update_method,
        "global_step": result.position.global_step,
        "epoch": result.position.epoch,
        "consumed_record_count": result.consumed_record_count,
        "next_example_position": result.next_example_position,
        "language_model_loss": result.language_model_loss,
        "total_objective": result.total_objective,
        "shared_variance": result.shared_variance,
        "optimizer_call_count": result.optimizer_call_count,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the latent core on filtered Calc-ASDiv_A using EGGROLL and "
            "the frozen Qwen/Qwen2.5-0.5B-Instruct backbone."
        )
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
        help="Limit Calc-ASDiv_A training records. Default uses all 570 train records.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume from. 'latest' finds the highest epoch in --save-dir.",
    )
    args = parser.parse_args(argv)
    validate_eggroll_config(
        args.pop_size, args.sigma, args.lr, args.rank, args.eval_batch_size
    )
    return args


def _load_dataset(problem_count: int | None) -> list[tuple[str, str]]:
    _log("loading pinned Calc-ASDiv_A partitions...")
    t0 = time.monotonic()
    stage0_dataset = load_stage0_dataset()
    dataset = training_examples(
        stage0_dataset,
        mode="eggroll",
        epoch=1,
        problem_count=problem_count,
    )
    _log(f"loaded {len(dataset)} problems ({_format_duration(time.monotonic() - t0)})")
    return dataset


def _load_dataset_context(problem_count: int | None):
    _log("loading pinned Calc-ASDiv_A partitions...")
    t0 = time.monotonic()
    stage0_dataset = load_stage0_dataset()
    dataset = training_examples(
        stage0_dataset, mode="eggroll", epoch=1, problem_count=problem_count
    )
    train = selection_metadata(stage0_dataset.train_selection, problem_count)
    held_out = selection_metadata(stage0_dataset.held_out_selection, None)
    selections = {"train": train, "held_out": held_out}
    identity = training_identity(
        runtime=runtime_identity(), held_out_item_ids=held_out["ordered_item_ids"]
    )
    _log(f"loaded {len(dataset)} problems ({_format_duration(time.monotonic() - t0)})")
    return dataset, identity, selections


def _find_latest_checkpoint(save_dir: str) -> str | None:
    path = latest_epoch_checkpoint(save_dir)
    return str(path) if path is not None else None


def _load_checkpoint(
    path: str,
    *,
    identity: dict[str, object],
    selections: dict[str, object],
    learning_rate: float,
):
    _log(f"loading checkpoint: {path}")
    checkpoint = load_checkpoint(path, expected_mode="eggroll")
    validate_compatibility(
        checkpoint,
        identity=identity,
        selections=selections,
        learning_rate=learning_rate,
    )
    epoch = checkpoint_epoch(path)
    _log(f"resumed from epoch {epoch}")
    return checkpoint, epoch


def _save_checkpoint(
    trainer: EggrollTrainer,
    save_dir: str,
    epoch: int,
    *,
    identity: dict[str, object],
    selections: dict[str, object],
    metrics: dict[str, object],
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"epoch-{epoch}.ckpt")
    checkpoint = build_checkpoint(
        mode="eggroll",
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
    _log(
        f"config: epochs={args.epochs} slots={args.slot_count} "
        f"steps={args.num_steps} pop={args.pop_size} "
        f"sigma={args.sigma} optimizer=SGD lr={args.lr} rank={args.rank} "
        f"var_weight={args.variance_weight} eval_batch={args.eval_batch_size} "
        f"amp={args.use_amp}"
    )

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

    _log(
        "building trainer (loading frozen Qwen/Qwen2.5-0.5B-Instruct "
        "+ frozen MiniLM)..."
    )

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

    if resumed is not None:
        restore_checkpoint(
            resumed,
            model_parameters=trainer.state.trainable_params,
            optimizer=trainer.optimizer,
        )

    _log(f"{'=' * 60}")
    _log(f"  EGGROLL ES training: {args.epochs} epochs x {len(dataset)} examples")
    _log(f"  population={args.pop_size} (antithetic pairs)")
    _log(f"{'=' * 60}")

    epoch_losses: list[float] = []
    training_start = time.monotonic()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.monotonic()
        recent_losses: list[float] = []
        recent_vars: list[float] = []

        def _on_step(step: int, total: int, result: StepResult) -> None:
            progress = _training_progress_record(result)
            recent_losses.append(result.total_objective)
            recent_vars.append(result.shared_variance)
            if (step + 1) % args.log_every == 0 or step + 1 == total:
                elapsed = time.monotonic() - epoch_start
                per_example = elapsed / (step + 1)
                remaining = per_example * (total - step - 1)
                avg_loss = sum(recent_losses) / len(recent_losses)
                avg_var = sum(recent_vars) / len(recent_vars)
                _log(
                    f"  epoch {epoch}/{args.epochs} "
                    f"[{step + 1}/{total}] "
                    f"loss={result.total_objective:.4f} avg={avg_loss:.4f} "
                    f"var={result.shared_variance:.6f} avg_var={avg_var:.6f} "
                    f"consumed={progress['consumed_record_count']} "
                    f"next={progress['next_example_position']} "
                    f"optimizer_calls={progress['optimizer_call_count']} "
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
            },
        )
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
