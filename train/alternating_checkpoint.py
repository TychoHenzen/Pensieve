"""Versioned checkpoints for resumable alternating-training experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Mapping

import torch


CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class CheckpointSchedule:
    """The cursor needed to continue the alternating update schedule."""

    active_phase: str
    completed_phase_steps: int
    global_step: int
    epoch: int
    dataset_position: int

    def to_dict(self) -> dict[str, int | str]:
        """Return a serializable schedule representation."""
        return {
            "active_phase": self.active_phase,
            "completed_phase_steps": self.completed_phase_steps,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "dataset_position": self.dataset_position,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointSchedule:
        """Build a schedule cursor from a serialized representation."""
        return cls(
            active_phase=str(value["active_phase"]),
            completed_phase_steps=int(value["completed_phase_steps"]),
            global_step=int(value["global_step"]),
            epoch=int(value["epoch"]),
            dataset_position=int(value["dataset_position"]),
        )


@dataclass(frozen=True)
class AlternatingCheckpoint:
    """All state required to resume one alternating-training experiment."""

    model_state: Mapping[str, Any]
    eggroll_optimizer_state: Mapping[str, Any]
    gradient_optimizer_state: Mapping[str, Any]
    schedule: CheckpointSchedule
    run_config: Mapping[str, Any]
    held_out_selection: Any
    metrics: Any
    python_random_state: object
    torch_random_state: torch.Tensor
    version: int = CHECKPOINT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned payload written to a checkpoint file."""
        return {
            "version": self.version,
            "model_state": dict(self.model_state),
            "eggroll_optimizer_state": dict(self.eggroll_optimizer_state),
            "gradient_optimizer_state": dict(self.gradient_optimizer_state),
            "schedule": self.schedule.to_dict(),
            "run_config": dict(self.run_config),
            "held_out_selection": self.held_out_selection,
            "metrics": self.metrics,
            "python_random_state": self.python_random_state,
            "torch_random_state": self.torch_random_state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AlternatingCheckpoint:
        """Load a supported versioned payload without restoring process state."""
        version = payload.get("version")
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported alternating checkpoint version {version!r}; "
                f"expected {CHECKPOINT_VERSION}"
            )
        return cls(
            version=version,
            model_state=payload["model_state"],
            eggroll_optimizer_state=payload["eggroll_optimizer_state"],
            gradient_optimizer_state=payload["gradient_optimizer_state"],
            schedule=CheckpointSchedule.from_dict(payload["schedule"]),
            run_config=payload["run_config"],
            held_out_selection=payload["held_out_selection"],
            metrics=payload["metrics"],
            python_random_state=payload["python_random_state"],
            torch_random_state=payload["torch_random_state"],
        )


def capture_checkpoint(
    *,
    model_state: Mapping[str, Any],
    eggroll_optimizer: torch.optim.Optimizer,
    gradient_optimizer: torch.optim.Optimizer,
    schedule: CheckpointSchedule,
    run_config: Mapping[str, Any],
    held_out_selection: Any,
    metrics: Any,
) -> AlternatingCheckpoint:
    """Capture named model, optimizer, schedule, metric, and RNG state."""
    return AlternatingCheckpoint(
        model_state=model_state,
        eggroll_optimizer_state=eggroll_optimizer.state_dict(),
        gradient_optimizer_state=gradient_optimizer.state_dict(),
        schedule=schedule,
        run_config=run_config,
        held_out_selection=held_out_selection,
        metrics=metrics,
        python_random_state=random.getstate(),
        torch_random_state=torch.get_rng_state(),
    )


def save_checkpoint(path: str | Path, checkpoint: AlternatingCheckpoint) -> None:
    """Write a versioned alternating-training checkpoint."""
    torch.save(checkpoint.to_dict(), Path(path))


def load_checkpoint(path: str | Path) -> AlternatingCheckpoint:
    """Read a versioned alternating-training checkpoint."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("alternating checkpoint payload must be a mapping")
    return AlternatingCheckpoint.from_dict(payload)
