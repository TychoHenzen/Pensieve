"""Safe version-2 checkpoints shared by standalone Stage 0 trainers."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save as save_safetensors

from eval.stage0_identity import ASDIV_REVISION
from eval.stream.generators.asdiv_a import (
    AsdivSelection,
    asdiv_a_selection_identity,
)
from train.stage0_checkpoint import (
    ALLOWED_MODEL_PARAMETER_PATHS,
    EGGROLL_MODEL_PARAMETER_PATHS,
    LoadedCheckpointContainer,
    read_checkpoint_container,
    validate_checkpoint_metadata,
    write_checkpoint_container,
)


def configure_deterministic_runtime() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def runtime_identity() -> dict[str, object]:
    return {
        "initialization_seed": 0,
        "python_version": platform.python_version(),
        "numpy_version": _version("numpy"),
        "pytorch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda or "none"),
        "transformers_version": _version("transformers"),
        "datasets_version": _version("datasets"),
        "sentence_transformers_version": _version("sentence-transformers"),
        "device_topology": [f"cuda:{index}" for index in range(torch.cuda.device_count())],
        "dtype": "float32",
        "attention_implementation": "eager",
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def selection_metadata(
    selection: AsdivSelection, count: int | None
) -> dict[str, object]:
    normalized_count = selection.problem_count if count is None else count
    records = selection.records[:normalized_count]
    identity = (
        selection.identity
        if normalized_count == selection.problem_count
        else asdiv_a_selection_identity(
            records=records,
            split=selection.split,
            seed=selection.seed,
            problem_count=normalized_count,
            revision=ASDIV_REVISION,
        )
    )
    return {
        "identity": identity,
        "split": selection.split,
        "seed": selection.seed,
        "problem_count": normalized_count,
        "ordered_item_ids": [record.id for record in records],
    }


def _dtype(tensor: torch.Tensor) -> str:
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
    if tensor.dtype not in names:
        raise ValueError(f"unsupported checkpoint tensor dtype {tensor.dtype}")
    return names[tensor.dtype]


def _manifest(name: str, tensor: torch.Tensor, role: str) -> dict[str, object]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": _dtype(tensor),
        "role": role,
    }


def _optimizer_manifest(
    method: str,
    optimizer: torch.optim.Optimizer,
    tensors: dict[str, torch.Tensor],
    tensor_manifest: list[dict[str, object]],
) -> dict[str, object]:
    parameter_names = list(
        EGGROLL_MODEL_PARAMETER_PATHS
        if method == "eggroll"
        else ALLOWED_MODEL_PARAMETER_PATHS
    )
    state = optimizer.state_dict()
    groups = state["param_groups"]
    if len(groups) != 1 or len(groups[0]["params"]) != len(parameter_names):
        raise ValueError("optimizer must contain one canonical Stage 0 parameter group")
    scalars: dict[str, object] = {}
    for key, value in groups[0].items():
        if key == "params" or value is None:
            continue
        if key == "betas":
            scalars["beta1"], scalars["beta2"] = value
        elif isinstance(value, (str, bool, int, float)):
            scalars[key] = value
        else:
            raise ValueError(f"optimizer group scalar {key!r} is not JSON-compatible")
    scalar_state: dict[str, dict[str, object]] = {}
    references: dict[str, dict[str, str]] = {}
    for name, parameter_id in zip(parameter_names, groups[0]["params"]):
        scalar_state[name] = {}
        references[name] = {}
        for state_name, value in state["state"].get(parameter_id, {}).items():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                tensor_name = f"optimizer.{method}.{name}.{state_name}"
                tensor = value.detach().cpu().contiguous().clone()
                tensors[tensor_name] = tensor
                tensor_manifest.append(_manifest(tensor_name, tensor, "optimizer_state"))
                references[name][state_name] = tensor_name
            elif isinstance(value, torch.Tensor):
                scalar_state[name][state_name] = value.item()
            elif isinstance(value, (str, bool, int, float)):
                scalar_state[name][state_name] = value
            else:
                raise ValueError(f"optimizer state {state_name!r} is not checkpoint-safe")
    return {
        "method": method,
        "optimizer_type": type(optimizer).__name__,
        "parameter_names": parameter_names,
        "parameter_groups": [
            {"parameter_names": parameter_names, "scalars": scalars}
        ],
        "scalar_state": scalar_state,
        "tensor_references": references,
    }


def build_checkpoint(
    *,
    mode: str,
    identity: Mapping[str, Any],
    selections: Mapping[str, Any],
    model_state: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    metrics: Mapping[str, Any],
    schedule: Mapping[str, Any] | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> LoadedCheckpointContainer:
    if mode not in {"gradient", "eggroll"}:
        raise ValueError("standalone checkpoint mode must be gradient or eggroll")
    if set(model_state) != set(ALLOWED_MODEL_PARAMETER_PATHS):
        raise ValueError("model_state must use every canonical Stage 0 parameter name")
    tensors: dict[str, torch.Tensor] = {}
    tensor_manifest: list[dict[str, object]] = []
    for name in ALLOWED_MODEL_PARAMETER_PATHS:
        tensor_name = f"model.{name}"
        tensor = model_state[name].detach().cpu().contiguous().clone()
        tensors[tensor_name] = tensor
        tensor_manifest.append(_manifest(tensor_name, tensor, "model_parameter"))
    optimizer_manifest = _optimizer_manifest(
        mode, optimizer, tensors, tensor_manifest
    )
    numpy_state = np.random.get_state()
    numpy_tensor = torch.from_numpy(np.asarray(numpy_state[1], dtype=np.uint32).copy())
    tensors["rng.numpy.state"] = numpy_tensor
    tensor_manifest.append(_manifest("rng.numpy.state", numpy_tensor, "numpy_rng_state"))
    cpu_rng = torch.get_rng_state().detach().cpu().clone()
    tensors["rng.pytorch.cpu"] = cpu_rng
    tensor_manifest.append(_manifest("rng.pytorch.cpu", cpu_rng, "pytorch_cpu_rng_state"))
    cuda: list[dict[str, str]] = []
    for index, state in enumerate(torch.cuda.get_rng_state_all()):
        name = f"rng.pytorch.cuda.{index}"
        tensor = state.detach().cpu().clone()
        tensors[name] = tensor
        tensor_manifest.append(_manifest(name, tensor, "pytorch_cuda_rng_state"))
        cuda.append({"device": f"cuda:{index}", "state_tensor": name})
    python_state = random.getstate()
    saved_schedule = dict(schedule or {})
    saved_metrics = dict(metrics)
    if mode == "eggroll" and saved_schedule:
        saved_metrics.update(
            {
                "consumed_examples": saved_schedule["consumed_examples"],
                "eggroll_optimizer_calls": saved_schedule[
                    "eggroll_optimizer_calls"
                ],
            }
        )
    metadata = {
        "schema_version": 2,
        "identity": dict(identity),
        "mode": mode,
        "tensor_manifest": tensor_manifest,
        "optimizer_manifests": [optimizer_manifest],
        "schedule": saved_schedule,
        "selections": dict(selections),
        "metrics": saved_metrics,
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
            "cuda": cuda,
        },
    }
    if run_config is not None:
        metadata["run_config"] = dict(run_config)
    validated = validate_checkpoint_metadata(metadata)
    return LoadedCheckpointContainer(validated.to_dict(), tensors)


def save_checkpoint(path: str | Path, checkpoint: LoadedCheckpointContainer) -> None:
    target = Path(path)
    if target.suffix != ".ckpt":
        raise ValueError("Stage 0 checkpoint paths must use the .ckpt suffix")
    write_checkpoint_container(
        target,
        metadata=checkpoint.metadata,
        tensor_payload=save_safetensors(dict(checkpoint.tensors)),
    )


def load_checkpoint(path: str | Path, *, expected_mode: str) -> LoadedCheckpointContainer:
    loaded = read_checkpoint_container(path)
    if loaded.metadata["mode"] != expected_mode:
        raise ValueError(
            f"$.mode: expected {expected_mode!r}, got {loaded.metadata['mode']!r}"
        )
    return loaded


def validate_compatibility(
    checkpoint: LoadedCheckpointContainer,
    *,
    identity: Mapping[str, Any],
    selections: Mapping[str, Any],
    learning_rate: float | None = None,
    run_config: Mapping[str, Any] | None = None,
    completed_epochs: int | None = None,
) -> None:
    def plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [plain(item) for item in value]
        return value

    def differences(actual: Any, expected: Any, path: str) -> list[str]:
        if isinstance(actual, Mapping) and isinstance(expected, Mapping):
            paths: list[str] = []
            for key in sorted(set(actual) | set(expected)):
                if key not in actual or key not in expected:
                    paths.append(f"{path}.{key}")
                else:
                    paths.extend(differences(actual[key], expected[key], f"{path}.{key}"))
            return paths
        if isinstance(actual, list) and isinstance(expected, list):
            paths = []
            for index in range(max(len(actual), len(expected))):
                if index >= len(actual) or index >= len(expected):
                    paths.append(f"{path}[{index}]")
                else:
                    paths.extend(differences(actual[index], expected[index], f"{path}[{index}]"))
            return paths
        return [] if actual == expected else [path]

    conflicts = differences(
        plain(checkpoint.metadata["identity"]), plain(identity), "$.identity"
    )
    conflicts.extend(
        differences(
            plain(checkpoint.metadata["selections"]),
            plain(selections),
            "$.selections",
        )
    )
    if run_config is not None:
        checkpoint_config = checkpoint.metadata.get("run_config")
        if not isinstance(checkpoint_config, Mapping):
            conflicts.append("$.run_config")
        else:
            guarded_checkpoint = {
                key: value
                for key, value in checkpoint_config.items()
                if key not in {"epochs", "logging_frequency"}
            }
            guarded_resume = {
                key: value
                for key, value in run_config.items()
                if key not in {"epochs", "logging_frequency"}
            }
            conflicts.extend(
                differences(
                    plain(guarded_checkpoint),
                    plain(guarded_resume),
                    "$.run_config",
                )
            )
            epoch_target = run_config.get("epochs")
            progress = (
                int(checkpoint.metadata["schedule"]["epoch"])
                if completed_epochs is None
                else completed_epochs
            )
            if (
                not isinstance(epoch_target, int)
                or isinstance(epoch_target, bool)
                or epoch_target < progress
            ):
                conflicts.append("$.run_config.epochs")
    if learning_rate is not None:
        saved_lr = checkpoint.metadata["optimizer_manifests"][0]["parameter_groups"][0]["scalars"].get("lr")
        if saved_lr != learning_rate:
            conflicts.append(
                "$.optimizer_manifests[0].parameter_groups[0].scalars.lr"
            )
    if conflicts:
        raise ValueError("incompatible resume configuration: " + ", ".join(conflicts))


def restore_checkpoint(
    checkpoint: LoadedCheckpointContainer,
    *,
    model_parameters: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> None:
    incompatibilities: list[str] = []
    if set(model_parameters) != set(ALLOWED_MODEL_PARAMETER_PATHS):
        incompatibilities.append("$.tensor_manifest")
    for name, parameter in model_parameters.items():
        saved = checkpoint.tensors.get(f"model.{name}")
        if saved is None or saved.shape != parameter.shape:
            incompatibilities.append(f"$.tensor_manifest.model.{name}.shape")
        elif saved.dtype != parameter.dtype:
            incompatibilities.append(f"$.tensor_manifest.model.{name}.dtype")
    manifest = checkpoint.metadata["optimizer_manifests"][0]
    current = optimizer.state_dict()
    parameter_ids = current["param_groups"][0]["params"]
    parameters = optimizer.param_groups[0]["params"]
    names = list(manifest["parameter_names"])
    if len(parameter_ids) != len(names) or len(parameters) != len(names):
        incompatibilities.append("$.optimizer_manifests[0].parameter_names")
    restored_state: dict[int, dict[str, object]] = {}
    for parameter_id, parameter, name in zip(parameter_ids, parameters, names):
        values = dict(manifest["scalar_state"][name])
        for state_name, tensor_name in manifest["tensor_references"][name].items():
            tensor = checkpoint.tensors[tensor_name]
            if tensor.shape != parameter.shape:
                incompatibilities.append(
                    f"$.optimizer_manifests[0].tensor_references.{name}.{state_name}"
                )
            values[state_name] = tensor
        if values:
            restored_state[parameter_id] = values
    if incompatibilities:
        raise ValueError("incompatible resume configuration: " + ", ".join(incompatibilities))
    group = dict(current["param_groups"][0])
    scalars = dict(manifest["parameter_groups"][0]["scalars"])
    if "beta1" in scalars:
        group["betas"] = (scalars.pop("beta1"), scalars.pop("beta2"))
    group.update(scalars)
    group["params"] = parameter_ids
    optimizer_state = {"state": restored_state, "param_groups": [group]}
    rng = checkpoint.metadata["rng"]
    python_rng = rng["python"]
    numpy_rng = rng["numpy"]
    with torch.no_grad():
        for name, parameter in model_parameters.items():
            parameter.copy_(checkpoint.tensors[f"model.{name}"])
    optimizer.load_state_dict(optimizer_state)
    random.setstate(
        (python_rng["version"], tuple(python_rng["state"]), python_rng["gaussian_cache"])
    )
    np.random.set_state(
        (
            numpy_rng["bit_generator"],
            checkpoint.tensors[numpy_rng["state_tensor"]].numpy(),
            numpy_rng["position"],
            int(numpy_rng["has_gaussian"]),
            numpy_rng["gaussian_cache"],
        )
    )
    torch.set_rng_state(checkpoint.tensors[rng["pytorch_cpu"]["state_tensor"]])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(
            [checkpoint.tensors[item["state_tensor"]] for item in rng["cuda"]]
        )


def latest_epoch_checkpoint(directory: str | Path) -> Path | None:
    root = Path(directory)
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("epoch-*.ckpt") if root.is_dir() else ():
        match = re.fullmatch(r"epoch-(\d+)\.ckpt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def checkpoint_epoch(path: str | Path) -> int:
    match = re.fullmatch(r"epoch-(\d+)\.ckpt", Path(path).name)
    if match is None:
        raise ValueError("standalone checkpoint filename must be epoch-<number>.ckpt")
    return int(match.group(1))
