"""CLI entry point: trains the latent-core subject's learned parameters on GSM8K.

Loads the GSM8K train split (~7.5k problems), runs `LatentCoreTrainer` for a
configurable number of epochs, and after each epoch logs the average loss,
trainable parameter count, and the workspace's collapse-detection stats
(variance/covariance, see `Workspace.collapse_stats`). A checkpoint of the
trainable state dicts is saved after every epoch.
"""

from __future__ import annotations

import argparse
import os

import torch

from eval.stream.generators.gsm8k import _extract_answer, _load_split
from train.trainer import DEFAULT_LR, DEFAULT_NUM_STEPS, LatentCoreTrainer
from workspace.concept_slots import DEFAULT_SLOT_COUNT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the latent-core subject on GSM8K.")
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
        help="Number of GSM8K train problems to use; default uses the full split.",
    )
    return parser.parse_args()


def _load_dataset(problem_count: int | None) -> list[tuple[str, str]]:
    problems = _load_split("train")
    if problem_count is not None:
        problems = problems[:problem_count]
    return [(item["question"], _extract_answer(item["answer"])) for item in problems]


def _save_checkpoint(trainer: LatentCoreTrainer, save_dir: str, epoch: int) -> str:
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"epoch-{epoch}.pt")
    state = {
        "encoder_projection": trainer.encoder.projection.state_dict(),
        "encoder_cross_attention": trainer.encoder.cross_attention.state_dict(),
        "encoder_slot_queries": trainer.encoder.slot_queries,
        "latent_loop_projection": trainer.latent_loop.projection.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, checkpoint_path)
    return checkpoint_path


def main() -> None:
    args = _parse_args()

    dataset = _load_dataset(args.problem_count)

    trainer = LatentCoreTrainer(
        slot_count=args.slot_count,
        num_steps=args.num_steps,
        lr=args.lr,
        device=args.device,
    )

    epoch_losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        avg_loss = trainer.train_epoch(dataset)
        epoch_losses.append(avg_loss)

        param_count = trainer.trainable_param_count()
        collapse_stats = trainer.workspace.collapse_stats()

        print(
            f"epoch {epoch}/{args.epochs} "
            f"loss={avg_loss:.4f} "
            f"trainable_params={param_count} "
            f"variance={collapse_stats['variance']:.6f} "
            f"covariance={collapse_stats['covariance']:.6f}"
        )
        if collapse_stats["variance"] <= 0.0:
            print(f"warning: workspace variance collapsed to {collapse_stats['variance']} at epoch {epoch}")

        checkpoint_path = _save_checkpoint(trainer, args.save_dir, epoch)
        print(f"saved checkpoint: {checkpoint_path}")

    if epoch_losses:
        print(
            "training summary: "
            f"first_epoch_loss={epoch_losses[0]:.4f} "
            f"last_epoch_loss={epoch_losses[-1]:.4f} "
            f"decreased={epoch_losses[-1] < epoch_losses[0]}"
        )


if __name__ == "__main__":
    main()
