"""Versioned checkpoints for resumable alternating-training experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import save as save_safetensors

from train.stage0_checkpoint import (
    EGGROLL_MODEL_PARAMETER_PATHS,
    LoadedCheckpointContainer,
    read_checkpoint_container,
    validate_checkpoint_metadata,
    write_checkpoint_container,
)


CHECKPOINT_VERSION = 2
SCHEDULE_DEFINING_SETTINGS = (
    "dataset_selection",
    "phase_steps",
    "model_shape",
    "gradient_optimizer",
    "eggroll_optimizer",
    "eggroll_population",
    "held_out_selection",
)


@dataclass(frozen=True)
class CheckpointSchedule:
    """The cursor needed to continue the alternating update schedule."""

    active_phase: str
    completed_phase_steps: int
    global_step: int
    epoch: int
    dataset_position: int
    phase_variance_sum: float = 0.0
    gradient_optimizer_calls: int = 0
    eggroll_optimizer_calls: int = 0

    def to_dict(self) -> dict[str, int | str]:
        """Return a serializable schedule representation."""
        return {
            "active_phase": self.active_phase,
            "completed_phase_steps": self.completed_phase_steps,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "dataset_position": self.dataset_position,
            "gradient_optimizer_calls": self.gradient_optimizer_calls,
            "eggroll_optimizer_calls": self.eggroll_optimizer_calls,
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
            gradient_optimizer_calls=int(value.get("gradient_optimizer_calls", 0)),
            eggroll_optimizer_calls=int(value.get("eggroll_optimizer_calls", 0)),
        )


@dataclass(frozen=True)
class AlternatingCheckpoint:
    """A fully specified safe checkpoint ready to write or apply."""

    metadata: Mapping[str, Any]
    tensors: Mapping[str, torch.Tensor]

    @property
    def schedule(self) -> CheckpointSchedule:
        """Expose the saved cursor through the scheduler's existing value type."""
        value = self.metadata["schedule"]
        next_position = int(value["next_dataset_position"])
        return CheckpointSchedule(
            active_phase=str(value["active_phase"]),
            completed_phase_steps=int(value["completed_phase_steps"]),
            global_step=int(value["global_step"]),
            epoch=int(value["epoch"]),
            dataset_position=next_position - 1,
            phase_variance_sum=float(
                self.metadata.get("metrics", {}).get("phase_variance_sum", 0.0)
            ),
            gradient_optimizer_calls=int(value.get("gradient_optimizer_calls", 0)),
            eggroll_optimizer_calls=int(value.get("eggroll_optimizer_calls", 0)),
        )


def capture_checkpoint(
    *,
    metadata: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
) -> AlternatingCheckpoint:
    """Validate caller-supplied identity and clone all tensors onto CPU."""
    validated = validate_checkpoint_metadata(metadata)
    if validated.mode != "alternating":
        raise ValueError("$.mode: alternating checkpoint capture requires 'alternating'")
    cpu_tensors = {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in tensors.items()
    }
    return AlternatingCheckpoint(validated.to_dict(), cpu_tensors)


def _tensor_dtype(tensor: torch.Tensor) -> str:
    names = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float64: "float64",
        torch.int64: "int64",
        torch.int32: "int32",
        torch.int16: "int16",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.uint32: "uint32",
        torch.bool: "bool",
    }
    try:
        return names[tensor.dtype]
    except KeyError as error:
        raise ValueError(f"unsupported checkpoint tensor dtype {tensor.dtype}") from error


def _manifest_item(name: str, tensor: torch.Tensor, role: str) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": _tensor_dtype(tensor),
        "role": role,
    }


def _optimizer_checkpoint_state(
    method: str,
    optimizer: torch.optim.Optimizer,
    tensors: dict[str, torch.Tensor],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    parameter_names = list(stage0_parameter_paths(method))
    state = optimizer.state_dict()
    groups = state["param_groups"]
    if len(groups) != 1 or len(groups[0]["params"]) != len(parameter_names):
        raise ValueError(
            f"$.optimizer_manifests.{method}.parameter_groups: "
            "must contain one canonical parameter group"
        )
    parameter_ids = list(groups[0]["params"])
    parameter_groups: list[dict[str, Any]] = []
    scalars: dict[str, Any] = {}
    for key, value in groups[0].items():
        if key == "params":
            continue
        if key == "betas":
            scalars["beta1"], scalars["beta2"] = value
        elif value is None:
            continue
        elif isinstance(value, (str, bool, int, float)):
            scalars[key] = value
        else:
            raise ValueError(
                f"$.optimizer_manifests.{method}.parameter_groups[0].scalars.{key}: "
                "must be a finite scalar"
            )
    parameter_groups.append(
        {"parameter_names": parameter_names, "scalars": scalars}
    )

    scalar_state: dict[str, dict[str, Any]] = {}
    references: dict[str, dict[str, str]] = {}
    for parameter_name, parameter_id in zip(parameter_names, parameter_ids):
        scalar_state[parameter_name] = {}
        references[parameter_name] = {}
        parameter_state = state["state"].get(parameter_id, {})
        for state_name, value in parameter_state.items():
            if isinstance(value, torch.Tensor) and (
                value.ndim > 0 or state_name != "step"
            ):
                tensor_name = f"optimizer.{method}.{parameter_name}.{state_name}"
                tensor = value.detach().to(device="cpu").contiguous().clone()
                tensors[tensor_name] = tensor
                manifest.append(_manifest_item(tensor_name, tensor, "optimizer_state"))
                references[parameter_name][state_name] = tensor_name
            elif isinstance(value, torch.Tensor):
                scalar_state[parameter_name][state_name] = value.item()
            elif isinstance(value, (str, bool, int, float)):
                scalar_state[parameter_name][state_name] = value
            else:
                raise ValueError(
                    f"$.optimizer_manifests.{method}.scalar_state.{parameter_name}.{state_name}: "
                    "must be a finite scalar or tensor"
                )
    return {
        "method": method,
        "optimizer_type": type(optimizer).__name__,
        "parameter_names": parameter_names,
        "parameter_groups": parameter_groups,
        "scalar_state": scalar_state,
        "tensor_references": references,
    }


def stage0_parameter_paths(method: str | None = None) -> tuple[str, ...]:
    """Return the canonical trainable-parameter order used by optimizers."""
    from train.stage0_checkpoint import ALLOWED_MODEL_PARAMETER_PATHS

    if method == "eggroll":
        return EGGROLL_MODEL_PARAMETER_PATHS
    return ALLOWED_MODEL_PARAMETER_PATHS


def build_alternating_checkpoint(
    *,
    identity: Mapping[str, Any],
    selections: Mapping[str, Any],
    model_state: Mapping[str, torch.Tensor],
    eggroll_optimizer: torch.optim.Optimizer,
    gradient_optimizer: torch.optim.Optimizer,
    schedule: CheckpointSchedule,
    phase_steps: int,
    next_dataset_position: int,
    metrics: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> AlternatingCheckpoint:
    """Capture complete alternating state from explicit identity and selections."""
    tensors: dict[str, torch.Tensor] = {}
    manifest: list[dict[str, Any]] = []
    expected_parameters = set(stage0_parameter_paths())
    if set(model_state) != expected_parameters:
        missing = sorted(expected_parameters - set(model_state))
        extra = sorted(set(model_state) - expected_parameters)
        raise ValueError(
            f"$.tensor_manifest: model parameter names differ; missing={missing!r}, extra={extra!r}"
        )
    for parameter_name in stage0_parameter_paths():
        tensor_name = f"model.{parameter_name}"
        tensor = model_state[parameter_name].detach().to(device="cpu").contiguous().clone()
        tensors[tensor_name] = tensor
        manifest.append(_manifest_item(tensor_name, tensor, "model_parameter"))

    optimizer_manifests = [
        _optimizer_checkpoint_state(
            "gradient", gradient_optimizer, tensors, manifest
        ),
        _optimizer_checkpoint_state("eggroll", eggroll_optimizer, tensors, manifest),
    ]

    numpy_state = np.random.get_state()
    numpy_tensor = torch.from_numpy(np.asarray(numpy_state[1], dtype=np.uint32).copy())
    tensors["rng.numpy.state"] = numpy_tensor
    manifest.append(_manifest_item("rng.numpy.state", numpy_tensor, "numpy_rng_state"))
    cpu_rng = torch.get_rng_state().detach().cpu().clone()
    tensors["rng.pytorch.cpu"] = cpu_rng
    manifest.append(
        _manifest_item("rng.pytorch.cpu", cpu_rng, "pytorch_cpu_rng_state")
    )
    cuda_metadata: list[dict[str, str]] = []
    for index, cuda_rng in enumerate(torch.cuda.get_rng_state_all()):
        tensor_name = f"rng.pytorch.cuda.{index}"
        tensor = cuda_rng.detach().cpu().clone()
        tensors[tensor_name] = tensor
        manifest.append(
            _manifest_item(tensor_name, tensor, "pytorch_cuda_rng_state")
        )
        cuda_metadata.append({"device": f"cuda:{index}", "state_tensor": tensor_name})

    python_state = random.getstate()
    metadata = {
        "schema_version": CHECKPOINT_VERSION,
        "identity": dict(identity),
        "mode": "alternating",
        "tensor_manifest": manifest,
        "optimizer_manifests": optimizer_manifests,
        "schedule": {
            "active_phase": schedule.active_phase,
            "completed_phase_steps": schedule.completed_phase_steps,
            "phase_steps": phase_steps,
            "global_step": schedule.global_step,
            "consumed_examples": schedule.global_step,
            "gradient_optimizer_calls": schedule.gradient_optimizer_calls,
            "eggroll_optimizer_calls": schedule.eggroll_optimizer_calls,
            "epoch": schedule.epoch,
            "next_dataset_position": next_dataset_position,
        },
        "selections": dict(selections),
        "metrics": {
            **dict(metrics),
            "consumed_examples": schedule.global_step,
            "gradient_optimizer_calls": schedule.gradient_optimizer_calls,
            "eggroll_optimizer_calls": schedule.eggroll_optimizer_calls,
        },
        "rng": {
            "python": {
                "version": python_state[0],
                "state": list(python_state[1]),
                "gaussian_cache": python_state[2],
            },
            "numpy": {
                "bit_generator": numpy_state[0],
                "position": numpy_state[2],
                "has_gaussian": bool(numpy_state[3]),
                "gaussian_cache": float(numpy_state[4]),
                "state_tensor": "rng.numpy.state",
            },
            "pytorch_cpu": {"state_tensor": "rng.pytorch.cpu"},
            "cuda": cuda_metadata,
        },
        "run_config": dict(run_config),
    }
    return capture_checkpoint(metadata=metadata, tensors=tensors)


def save_checkpoint(path: str | Path, checkpoint: AlternatingCheckpoint) -> None:
    """Write one atomic JSON-plus-safetensors checkpoint."""
    target = Path(path)
    if target.suffix != ".ckpt":
        raise ValueError("Stage 0 checkpoint paths must use the .ckpt suffix")
    write_checkpoint_container(
        target,
        metadata=checkpoint.metadata,
        tensor_payload=save_safetensors(dict(checkpoint.tensors)),
    )


def load_checkpoint(path: str | Path) -> AlternatingCheckpoint:
    """Read a validated version-2 checkpoint without accepting pickle."""
    loaded: LoadedCheckpointContainer = read_checkpoint_container(Path(path))
    return AlternatingCheckpoint(loaded.metadata, loaded.tensors)


def latest_checkpoint(directory: str | Path) -> Path | None:
    """Return the alternating checkpoint with the highest stored global step."""
    checkpoint_directory = Path(directory)
    if not checkpoint_directory.is_dir():
        return None

    candidates = {
        *checkpoint_directory.glob("phase-*.ckpt"),
        *checkpoint_directory.glob("epoch-*.ckpt"),
    }
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: int(load_checkpoint(path).metadata["schedule"]["global_step"]),
    )


def save_boundary_checkpoints(
    directory: str | Path,
    checkpoint: AlternatingCheckpoint,
    *,
    phase_boundary: bool,
    epoch_boundary: bool,
) -> tuple[Path, ...]:
    """Save one boundary payload and expose each applicable checkpoint name."""
    paths: list[Path] = []
    checkpoint_directory = Path(directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    if phase_boundary:
        paths.append(
            checkpoint_directory / f"phase-{checkpoint.schedule.global_step}.ckpt"
        )
    if epoch_boundary:
        paths.append(checkpoint_directory / f"epoch-{checkpoint.schedule.epoch}.ckpt")
    if not paths:
        return ()

    paths[0].unlink(missing_ok=True)
    save_checkpoint(paths[0], checkpoint)
    for alias in paths[1:]:
        alias.unlink(missing_ok=True)
        os.link(paths[0], alias)
    return tuple(paths)


def resume_config_conflicts(
    *,
    checkpoint_config: Mapping[str, Any],
    resume_config: Mapping[str, Any],
    completed_epochs: int,
) -> tuple[str, ...]:
    """Return canonical paths for settings that cannot change during resume."""
    conflicts = [
        f"$.run_config.{setting}"
        for setting in SCHEDULE_DEFINING_SETTINGS
        if checkpoint_config.get(setting) != resume_config.get(setting)
    ]

    epoch_target = resume_config.get("epochs")
    if (
        not isinstance(epoch_target, int)
        or isinstance(epoch_target, bool)
        or epoch_target < completed_epochs
    ):
        conflicts.append("$.run_config.epochs")

    return tuple(dict.fromkeys(conflicts))


def validate_resume_config(
    *,
    checkpoint_config: Mapping[str, Any],
    resume_config: Mapping[str, Any],
    completed_epochs: int,
) -> None:
    """Reject resume settings that would change the saved training schedule."""
    conflicts = resume_config_conflicts(
        checkpoint_config=checkpoint_config,
        resume_config=resume_config,
        completed_epochs=completed_epochs,
    )

    if conflicts:
        raise ValueError(
            "incompatible resume configuration: " + ", ".join(conflicts)
        )
