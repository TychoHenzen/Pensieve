"""Bounded version-2 Stage 0 checkpoint metadata and container writer."""

from __future__ import annotations

import json
import math
import os
import struct
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from zipfile import BadZipFile, ZIP_STORED, ZipFile, ZipInfo

from eval.stage0_identity import STAGE0_IDENTITY


CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_MODES = frozenset({"gradient", "eggroll", "alternating"})
OPTIMIZER_METHODS = frozenset({"gradient", "eggroll"})
TENSOR_ROLES = frozenset(
    {
        "model_parameter",
        "optimizer_state",
        "numpy_rng_state",
        "pytorch_cpu_rng_state",
        "pytorch_cuda_rng_state",
    }
)
ALLOWED_MODEL_PARAMETER_PATHS = (
    "encoder.projection.weight",
    "encoder.projection.bias",
    "encoder.slot_queries",
    "encoder.attn_log_temp",
    "latent_loop.projection.weight",
    "latent_loop.projection.bias",
    "latent_loop.proj_norm.weight",
    "latent_loop.proj_norm.bias",
    "latent_loop.layer_norm.weight",
    "latent_loop.layer_norm.bias",
)

MAX_METADATA_BYTES = 1 * 1024 * 1024
MAX_TENSOR_BYTES = 64 * 1024 * 1024
MAX_METADATA_STRING_LENGTH = 16 * 1024
METADATA_MEMBER = "metadata.json"
TENSORS_MEMBER = "tensors.safetensors"

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "identity",
        "mode",
        "tensor_manifest",
        "optimizer_manifests",
        "schedule",
        "selections",
        "metrics",
        "rng",
    }
)
_TENSOR_FIELDS = frozenset({"name", "shape", "dtype", "role"})
_OPTIMIZER_FIELDS = frozenset(
    {
        "method",
        "optimizer_type",
        "parameter_names",
        "parameter_groups",
        "scalar_state",
        "tensor_references",
    }
)
_SCHEDULE_FIELDS = frozenset(
    {
        "active_phase",
        "completed_phase_steps",
        "phase_steps",
        "global_step",
        "epoch",
        "next_dataset_position",
    }
)
_SELECTION_FIELDS = frozenset(
    {"identity", "split", "seed", "problem_count", "ordered_item_ids"}
)
_RUNTIME_FIELDS = frozenset(
    {
        "initialization_seed",
        "python_version",
        "numpy_version",
        "pytorch_version",
        "cuda_version",
        "transformers_version",
        "datasets_version",
        "sentence_transformers_version",
        "device_topology",
        "dtype",
        "attention_implementation",
        "cublas_workspace_config",
        "deterministic_algorithms",
        "tf32_enabled",
        "cudnn_benchmark",
    }
)
_DTYPE_TO_SAFETENSORS = {
    "float32": "F32",
    "float16": "F16",
    "bfloat16": "BF16",
    "float64": "F64",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "uint8": "U8",
    "uint32": "U32",
    "bool": "BOOL",
}
_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float64": 8,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "uint32": 4,
    "bool": 1,
}


class CheckpointContainerError(ValueError):
    """A checkpoint cannot be represented by the bounded ZIP container."""


class CheckpointMetadataError(CheckpointContainerError):
    """Checkpoint metadata violates the version-2 schema."""


@dataclass(frozen=True)
class LoadedCheckpointContainer:
    """Validated metadata and CPU tensors read from a checkpoint container."""

    metadata: Mapping[str, Any]
    tensors: Any


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ValidatedCheckpointMetadata:
    schema_version: int
    identity: Mapping[str, Any]
    mode: str
    tensor_manifest: tuple[Mapping[str, Any], ...]
    optimizer_manifests: tuple[Mapping[str, Any], ...]
    schedule: Mapping[str, Any]
    selections: Mapping[str, Any]
    metrics: Mapping[str, Any]
    rng: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": _thaw(self.identity),
            "mode": self.mode,
            "tensor_manifest": _thaw(self.tensor_manifest),
            "optimizer_manifests": _thaw(self.optimizer_manifests),
            "schedule": _thaw(self.schedule),
            "selections": _thaw(self.selections),
            "metrics": _thaw(self.metrics),
            "rng": _thaw(self.rng),
        }


class _Validator:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def add(self, path: str, message: str) -> None:
        self.errors.append(f"{path}: {message}")

    def object(self, value: Any, path: str) -> Mapping[str, Any] | None:
        if not isinstance(value, Mapping):
            self.add(path, "must be an object")
            return None
        return value

    def exact_fields(
        self, value: Mapping[str, Any], fields: frozenset[str], path: str
    ) -> None:
        for field in sorted(fields - value.keys()):
            self.add(f"{path}.{field}", "is required")
        for field in sorted(value.keys() - fields):
            self.add(f"{path}.{field}", "is not allowed")

    def integer(
        self, value: Any, path: str, *, minimum: int | None = None
    ) -> bool:
        if not isinstance(value, int) or isinstance(value, bool):
            self.add(path, "must be an integer")
            return False
        if minimum is not None and value < minimum:
            self.add(path, f"must be at least {minimum}")
            return False
        return True

    def string(self, value: Any, path: str, *, nonempty: bool = True) -> bool:
        if not isinstance(value, str):
            self.add(path, "must be a string")
            return False
        if nonempty and not value:
            self.add(path, "must not be empty")
            return False
        if len(value) > MAX_METADATA_STRING_LENGTH:
            self.add(path, f"exceeds {MAX_METADATA_STRING_LENGTH} characters")
            return False
        return True


def _finite_scalar(value: Any) -> bool:
    return isinstance(value, (str, bool, int, float)) and not (
        isinstance(value, float) and not math.isfinite(value)
    )


def _validate_identity(
    validator: _Validator, identity: Any, selections: Any, rng: Any
) -> None:
    obj = validator.object(identity, "$.identity")
    if obj is None:
        return
    expected_fields = frozenset(STAGE0_IDENTITY) | {"runtime", "held_out_item_ids"}
    validator.exact_fields(obj, expected_fields, "$.identity")
    for key, expected in STAGE0_IDENTITY.items():
        if key in obj and obj[key] != expected:
            validator.add(f"$.identity.{key}", f"must equal {expected!r}")

    runtime = validator.object(obj.get("runtime"), "$.identity.runtime")
    if runtime is not None:
        validator.exact_fields(runtime, _RUNTIME_FIELDS, "$.identity.runtime")
        for key in _RUNTIME_FIELDS & runtime.keys():
            value = runtime[key]
            path = f"$.identity.runtime.{key}"
            if key == "initialization_seed":
                if validator.integer(value, path, minimum=0) and value != 0:
                    validator.add(path, "must equal 0")
            elif key == "device_topology":
                if not isinstance(value, list):
                    validator.add(path, "must be a list")
                else:
                    for index, device in enumerate(value):
                        validator.string(device, f"{path}[{index}]")
            elif key in {"deterministic_algorithms", "tf32_enabled", "cudnn_benchmark"}:
                if not isinstance(value, bool):
                    validator.add(path, "must be a boolean")
            else:
                validator.string(value, path)
        fixed = {
            "dtype": "float32",
            "attention_implementation": "eager",
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "tf32_enabled": False,
            "cudnn_benchmark": False,
        }
        for key, expected in fixed.items():
            if key in runtime and runtime[key] != expected:
                validator.add(f"$.identity.runtime.{key}", f"must equal {expected!r}")

    held = obj.get("held_out_item_ids")
    if not isinstance(held, list):
        validator.add("$.identity.held_out_item_ids", "must be a list")
    else:
        for index, item_id in enumerate(held):
            validator.string(item_id, f"$.identity.held_out_item_ids[{index}]")
        if isinstance(selections, Mapping):
            held_selection = selections.get("held_out")
            if isinstance(held_selection, Mapping):
                ordered = held_selection.get("ordered_item_ids")
                if isinstance(ordered, list) and held != ordered:
                    validator.add(
                        "$.identity.held_out_item_ids",
                        "must equal $.selections.held_out.ordered_item_ids",
                    )

    if runtime is not None and isinstance(rng, Mapping):
        topology = runtime.get("device_topology")
        cuda = rng.get("cuda")
        if isinstance(topology, list) and isinstance(cuda, list):
            devices = [item.get("device") for item in cuda if isinstance(item, Mapping)]
            if topology != devices:
                validator.add(
                    "$.identity.runtime.device_topology",
                    "must equal the ordered CUDA RNG topology",
                )


def _validate_tensor_manifest(
    validator: _Validator, manifest: Any
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    by_name: dict[str, Mapping[str, Any]] = {}
    names: list[str] = []
    if not isinstance(manifest, list):
        validator.add("$.tensor_manifest", "must be a list")
        return by_name, names
    for index, item in enumerate(manifest):
        path = f"$.tensor_manifest[{index}]"
        obj = validator.object(item, path)
        if obj is None:
            continue
        validator.exact_fields(obj, _TENSOR_FIELDS, path)
        name = obj.get("name")
        shape = obj.get("shape")
        dtype = obj.get("dtype")
        role = obj.get("role")
        valid_name = validator.string(name, f"{path}.name")
        if valid_name:
            if name in by_name:
                validator.add(f"{path}.name", f"duplicates tensor {name!r}")
            else:
                by_name[name] = obj
            names.append(name)
        if not isinstance(shape, list):
            validator.add(f"{path}.shape", "must be a list")
        else:
            for dimension_index, dimension in enumerate(shape):
                validator.integer(
                    dimension, f"{path}.shape[{dimension_index}]", minimum=0
                )
        if validator.string(dtype, f"{path}.dtype") and dtype not in _DTYPE_TO_SAFETENSORS:
            validator.add(f"{path}.dtype", "is not a supported tensor dtype")
        if role not in TENSOR_ROLES:
            validator.add(f"{path}.role", f"must be one of {sorted(TENSOR_ROLES)}")

        if valid_name and isinstance(role, str):
            if role == "model_parameter":
                allowed = {f"model.{parameter}" for parameter in ALLOWED_MODEL_PARAMETER_PATHS}
                if name not in allowed:
                    validator.add(f"{path}.name", "is not an allowed model parameter")
            elif role == "optimizer_state":
                parts = name.split(".")
                valid = (
                    len(parts) >= 5
                    and parts[0] == "optimizer"
                    and parts[1] in OPTIMIZER_METHODS
                    and ".".join(parts[2:-1]) in ALLOWED_MODEL_PARAMETER_PATHS
                    and bool(parts[-1])
                )
                if not valid:
                    validator.add(
                        f"{path}.name",
                        "must use optimizer.<method>.<parameter_path>.<state_name>",
                    )

    for parameter in ALLOWED_MODEL_PARAMETER_PATHS:
        name = f"model.{parameter}"
        item = by_name.get(name)
        if item is None:
            validator.add(f"$.tensor_manifest.{name}", "is required")
        elif item.get("role") != "model_parameter":
            validator.add(f"$.tensor_manifest.{name}", "must have model_parameter role")
    return by_name, names


def _validate_optimizer_manifests(
    validator: _Validator,
    value: Any,
    mode: Any,
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(value, list):
        validator.add("$.optimizer_manifests", "must be a list")
        return
    expected_methods = {
        "gradient": ["gradient"],
        "eggroll": ["eggroll"],
        "alternating": ["gradient", "eggroll"],
    }.get(mode, [])
    seen: set[str] = set()
    orders: list[list[str]] = []
    for index, item in enumerate(value):
        path = f"$.optimizer_manifests[{index}]"
        obj = validator.object(item, path)
        if obj is None:
            continue
        validator.exact_fields(obj, _OPTIMIZER_FIELDS, path)
        method = obj.get("method")
        if method not in OPTIMIZER_METHODS:
            validator.add(f"{path}.method", "must be gradient or eggroll")
        elif method in seen:
            validator.add(f"{path}.method", f"duplicates method {method!r}")
        else:
            seen.add(method)
        validator.string(obj.get("optimizer_type"), f"{path}.optimizer_type")

        parameter_names = obj.get("parameter_names")
        if not isinstance(parameter_names, list):
            validator.add(f"{path}.parameter_names", "must be a list")
            parameter_names = []
        else:
            expected = list(ALLOWED_MODEL_PARAMETER_PATHS)
            for position, parameter in enumerate(parameter_names):
                item_path = f"{path}.parameter_names[{position}]"
                if not validator.string(parameter, item_path):
                    continue
                if parameter not in ALLOWED_MODEL_PARAMETER_PATHS:
                    validator.add(item_path, "is not an allowed parameter")
                if parameter in parameter_names[:position]:
                    validator.add(item_path, "duplicates an earlier parameter")
            if parameter_names != expected:
                validator.add(f"{path}.parameter_names", "must use the canonical order")
            orders.append(parameter_names)

        groups = obj.get("parameter_groups")
        if not isinstance(groups, list) or not groups:
            validator.add(f"{path}.parameter_groups", "must be a non-empty list")
        else:
            grouped: list[str] = []
            for group_index, group in enumerate(groups):
                group_path = f"{path}.parameter_groups[{group_index}]"
                group_obj = validator.object(group, group_path)
                if group_obj is None:
                    continue
                validator.exact_fields(
                    group_obj, frozenset({"parameter_names", "scalars"}), group_path
                )
                names = group_obj.get("parameter_names")
                if not isinstance(names, list) or any(
                    not isinstance(name, str) for name in names
                ):
                    validator.add(f"{group_path}.parameter_names", "must be a string list")
                else:
                    grouped.extend(names)
                    if any(name not in parameter_names for name in names) or names != parameter_names:
                        validator.add(
                            f"{group_path}.parameter_names",
                            "must use the declared parameter order",
                        )
                scalars = group_obj.get("scalars")
                if not isinstance(scalars, Mapping) or any(
                    not isinstance(key, str) or not _finite_scalar(scalar)
                    for key, scalar in (scalars.items() if isinstance(scalars, Mapping) else [])
                ):
                    validator.add(
                        f"{group_path}.scalars", "must contain only finite scalar values"
                    )
            if grouped != parameter_names:
                validator.add(
                    f"{path}.parameter_groups", "must cover the canonical parameter order"
                )

        scalar_state = obj.get("scalar_state")
        if not isinstance(scalar_state, Mapping):
            validator.add(f"{path}.scalar_state", "must be an object")
        elif set(scalar_state) != set(parameter_names) or any(
            not isinstance(state, Mapping)
            or any(not isinstance(key, str) or not _finite_scalar(scalar) for key, scalar in state.items())
            for state in scalar_state.values()
        ):
            validator.add(
                f"{path}.scalar_state",
                "must contain finite scalar state for every declared parameter",
            )

        references = obj.get("tensor_references")
        valid_references = isinstance(references, Mapping) and set(references) == set(parameter_names)
        referenced_tensors: set[str] = set()
        if valid_references:
            for parameter, states in references.items():
                if not isinstance(states, Mapping):
                    valid_references = False
                    break
                for state_name, tensor_name in states.items():
                    expected_name = f"optimizer.{method}.{parameter}.{state_name}"
                    tensor = tensors.get(tensor_name) if isinstance(tensor_name, str) else None
                    if tensor_name != expected_name or tensor is None or tensor.get("role") != "optimizer_state":
                        valid_references = False
                        break
                    referenced_tensors.add(tensor_name)
        declared_optimizer_tensors = {
            name
            for name, tensor in tensors.items()
            if tensor.get("role") == "optimizer_state"
            and name.startswith(f"optimizer.{method}.")
        }
        if referenced_tensors != declared_optimizer_tensors:
            valid_references = False
        if not valid_references:
            validator.add(
                f"{path}.tensor_references",
                "must reference each declared optimizer-state tensor",
            )

    if [item.get("method") for item in value if isinstance(item, Mapping)] != expected_methods:
        validator.add("$.optimizer_manifests", f"must declare methods {expected_methods!r}")
    if len(orders) > 1 and any(order != orders[0] for order in orders[1:]):
        validator.add(
            "$.optimizer_manifests[1].parameter_names",
            "must match the first optimizer parameter order",
        )


def _validate_schedule(validator: _Validator, schedule: Any, mode: Any) -> None:
    obj = validator.object(schedule, "$.schedule")
    if obj is None:
        return
    if mode in {"gradient", "eggroll"}:
        if obj:
            validator.add("$.schedule", "must be empty for a standalone mode")
        return
    if mode != "alternating":
        return
    validator.exact_fields(obj, _SCHEDULE_FIELDS, "$.schedule")
    if "active_phase" in obj and obj["active_phase"] not in OPTIMIZER_METHODS:
        validator.add("$.schedule.active_phase", "must be gradient or eggroll")
    integer_bounds = {
        "completed_phase_steps": 0,
        "phase_steps": 1,
        "global_step": 0,
        "epoch": 1,
        "next_dataset_position": 0,
    }
    for field, minimum in integer_bounds.items():
        if field in obj:
            validator.integer(obj[field], f"$.schedule.{field}", minimum=minimum)
    completed = obj.get("completed_phase_steps")
    phase_steps = obj.get("phase_steps")
    if (
        isinstance(completed, int)
        and not isinstance(completed, bool)
        and isinstance(phase_steps, int)
        and not isinstance(phase_steps, bool)
        and completed >= phase_steps
    ):
        validator.add(
            "$.schedule.completed_phase_steps", "must be less than phase_steps"
        )


def _validate_selections(validator: _Validator, selections: Any) -> None:
    obj = validator.object(selections, "$.selections")
    if obj is None:
        return
    validator.exact_fields(obj, frozenset({"train", "held_out"}), "$.selections")
    expected_splits = {"train": "train", "held_out": "validation"}
    for name, expected_split in expected_splits.items():
        if name not in obj:
            continue
        path = f"$.selections.{name}"
        selection = validator.object(obj[name], path)
        if selection is None:
            continue
        validator.exact_fields(selection, _SELECTION_FIELDS, path)
        identity = selection.get("identity")
        if not validator.string(identity, f"{path}.identity") or (
            isinstance(identity, str)
            and (len(identity) != 64 or any(character not in "0123456789abcdef" for character in identity))
        ):
            validator.add(f"{path}.identity", "must be a lowercase SHA-256 digest")
        if selection.get("split") != expected_split:
            validator.add(f"{path}.split", f"must equal {expected_split!r}")
        seed = selection.get("seed")
        if validator.integer(seed, f"{path}.seed", minimum=0) and seed != 0:
            validator.add(f"{path}.seed", "must equal 0")
        count = selection.get("problem_count")
        validator.integer(count, f"{path}.problem_count", minimum=0)
        ordered = selection.get("ordered_item_ids")
        if not isinstance(ordered, list):
            validator.add(f"{path}.ordered_item_ids", "must be a list")
        else:
            seen: set[str] = set()
            for index, item_id in enumerate(ordered):
                item_path = f"{path}.ordered_item_ids[{index}]"
                if validator.string(item_id, item_path):
                    if item_id in seen:
                        validator.add(item_path, "duplicates an earlier item id")
                    seen.add(item_id)
            if isinstance(count, int) and not isinstance(count, bool) and count != len(ordered):
                validator.add(f"{path}.problem_count", "must equal ordered_item_ids length")


def _validate_metrics(validator: _Validator, metrics: Any) -> None:
    obj = validator.object(metrics, "$.metrics")
    if obj is None:
        return
    for key, value in obj.items():
        path = f"$.metrics.{key}"
        if not isinstance(key, str) or not key:
            validator.add(path, "metric names must be non-empty strings")
            continue
        if isinstance(value, list):
            if any(not _finite_scalar(item) for item in value):
                validator.add(path, "lists must contain only finite JSON scalars")
            for index, item in enumerate(value):
                if isinstance(item, str) and len(item) > MAX_METADATA_STRING_LENGTH:
                    validator.add(f"{path}[{index}]", "string is too long")
        elif not _finite_scalar(value):
            validator.add(path, "must be a finite JSON scalar or flat scalar list")
        elif isinstance(value, str) and len(value) > MAX_METADATA_STRING_LENGTH:
            validator.add(path, "string is too long")


def _validate_rng(
    validator: _Validator, rng: Any, tensors: Mapping[str, Mapping[str, Any]]
) -> None:
    obj = validator.object(rng, "$.rng")
    if obj is None:
        return
    validator.exact_fields(
        obj, frozenset({"python", "numpy", "pytorch_cpu", "cuda"}), "$.rng"
    )
    python = validator.object(obj.get("python"), "$.rng.python")
    if python is not None:
        validator.exact_fields(
            python, frozenset({"version", "state", "gaussian_cache"}), "$.rng.python"
        )
        if "version" in python:
            validator.integer(python["version"], "$.rng.python.version", minimum=0)
        state = python.get("state")
        if not isinstance(state, list):
            validator.add("$.rng.python.state", "must be an integer list")
        else:
            for index, item in enumerate(state):
                validator.integer(item, f"$.rng.python.state[{index}]", minimum=0)
        cache = python.get("gaussian_cache")
        if cache is not None and (
            not isinstance(cache, (int, float))
            or isinstance(cache, bool)
            or not math.isfinite(cache)
        ):
            validator.add("$.rng.python.gaussian_cache", "must be null or finite")

    numpy = validator.object(obj.get("numpy"), "$.rng.numpy")
    if numpy is not None:
        validator.exact_fields(
            numpy,
            frozenset(
                {"bit_generator", "position", "has_gaussian", "gaussian_cache", "state_tensor"}
            ),
            "$.rng.numpy",
        )
        if "bit_generator" in numpy:
            validator.string(numpy["bit_generator"], "$.rng.numpy.bit_generator")
        if "position" in numpy:
            validator.integer(numpy["position"], "$.rng.numpy.position", minimum=0)
        if "has_gaussian" in numpy and not isinstance(numpy["has_gaussian"], bool):
            validator.add("$.rng.numpy.has_gaussian", "must be a boolean")
        cache = numpy.get("gaussian_cache")
        if not isinstance(cache, (int, float)) or isinstance(cache, bool) or not math.isfinite(cache):
            validator.add("$.rng.numpy.gaussian_cache", "must be finite")
        _validate_rng_reference(
            validator,
            numpy.get("state_tensor"),
            "$.rng.numpy.state_tensor",
            tensors,
            "uint32",
            "numpy_rng_state",
        )

    cpu = validator.object(obj.get("pytorch_cpu"), "$.rng.pytorch_cpu")
    if cpu is not None:
        validator.exact_fields(cpu, frozenset({"state_tensor"}), "$.rng.pytorch_cpu")
        _validate_rng_reference(
            validator,
            cpu.get("state_tensor"),
            "$.rng.pytorch_cpu.state_tensor",
            tensors,
            "uint8",
            "pytorch_cpu_rng_state",
        )

    cuda = obj.get("cuda")
    if not isinstance(cuda, list):
        validator.add("$.rng.cuda", "must be a list")
    else:
        for index, item in enumerate(cuda):
            path = f"$.rng.cuda[{index}]"
            cuda_item = validator.object(item, path)
            if cuda_item is None:
                continue
            validator.exact_fields(cuda_item, frozenset({"device", "state_tensor"}), path)
            expected_device = f"cuda:{index}"
            if cuda_item.get("device") != expected_device:
                validator.add(f"{path}.device", f"must equal {expected_device!r}")
            _validate_rng_reference(
                validator,
                cuda_item.get("state_tensor"),
                f"{path}.state_tensor",
                tensors,
                "uint8",
                "pytorch_cuda_rng_state",
            )


def _validate_rng_reference(
    validator: _Validator,
    name: Any,
    path: str,
    tensors: Mapping[str, Mapping[str, Any]],
    dtype: str,
    role: str,
) -> None:
    tensor = tensors.get(name) if isinstance(name, str) else None
    if tensor is None or tensor.get("dtype") != dtype or tensor.get("role") != role:
        validator.add(path, f"must reference a declared {dtype} {role} tensor")


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
) -> ValidatedCheckpointMetadata:
    """Validate and deeply freeze complete version-2 checkpoint metadata."""
    validator = _Validator()
    root = validator.object(metadata, "$")
    if root is None:
        raise CheckpointMetadataError("; ".join(validator.errors))
    validator.exact_fields(root, _ROOT_FIELDS, "$")
    version = root.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != CHECKPOINT_SCHEMA_VERSION
    ):
        validator.add("$.schema_version", "must be the integer 2")
    mode = root.get("mode")
    if mode not in CHECKPOINT_MODES:
        validator.add("$.mode", f"must be one of {sorted(CHECKPOINT_MODES)}")

    tensors, _ = _validate_tensor_manifest(validator, root.get("tensor_manifest"))
    _validate_optimizer_manifests(
        validator, root.get("optimizer_manifests"), mode, tensors
    )
    _validate_schedule(validator, root.get("schedule"), mode)
    _validate_selections(validator, root.get("selections"))
    _validate_metrics(validator, root.get("metrics"))
    _validate_rng(validator, root.get("rng"), tensors)
    _validate_identity(
        validator, root.get("identity"), root.get("selections"), root.get("rng")
    )

    try:
        encoded = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > MAX_METADATA_BYTES:
            validator.add("$", f"encoded metadata exceeds {MAX_METADATA_BYTES} bytes")
    except (TypeError, ValueError) as error:
        validator.add("$", f"must be finite JSON data: {error}")

    if validator.errors:
        raise CheckpointMetadataError("; ".join(validator.errors))
    frozen = _freeze(dict(root))
    return ValidatedCheckpointMetadata(
        schema_version=frozen["schema_version"],
        identity=frozen["identity"],
        mode=frozen["mode"],
        tensor_manifest=frozen["tensor_manifest"],
        optimizer_manifests=frozen["optimizer_manifests"],
        schedule=frozen["schedule"],
        selections=frozen["selections"],
        metrics=frozen["metrics"],
        rng=frozen["rng"],
    )


def _canonical_metadata_bytes(
    metadata: Mapping[str, Any],
) -> tuple[bytes, dict[str, Mapping[str, Any]] | None]:
    validated = validate_checkpoint_metadata(metadata)
    canonical = validated.to_dict()
    declared_tensors = {
        item["name"]: item for item in canonical["tensor_manifest"]
    }
    try:
        payload = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CheckpointMetadataError(f"$: must be finite JSON data: {error}") from error
    if len(payload) > MAX_METADATA_BYTES:
        raise CheckpointContainerError(
            f"{METADATA_MEMBER} size {len(payload)} exceeds {MAX_METADATA_BYTES} bytes"
        )
    return payload, declared_tensors


def _duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    counts = Counter(key for key, _ in pairs)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise CheckpointContainerError(
            f"safetensors header has duplicate key {duplicates[0]!r}"
        )
    return dict(pairs)


def _validate_safetensors_payload(
    payload: bytes, declared_tensors: Mapping[str, Mapping[str, Any]] | None
) -> None:
    if not isinstance(payload, bytes):
        raise CheckpointContainerError("tensor_payload must be bytes")
    if len(payload) > MAX_TENSOR_BYTES:
        raise CheckpointContainerError(
            f"{TENSORS_MEMBER} size {len(payload)} exceeds {MAX_TENSOR_BYTES} bytes"
        )
    if len(payload) < 8:
        raise CheckpointContainerError("tensor_payload is not a safetensors container")
    header_size = struct.unpack("<Q", payload[:8])[0]
    if header_size == 0 or header_size > len(payload) - 8:
        raise CheckpointContainerError("tensor_payload has an invalid safetensors header size")
    try:
        header = json.loads(
            payload[8 : 8 + header_size].decode("utf-8"),
            object_pairs_hook=_duplicate_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CheckpointContainerError(f"invalid safetensors header: {error}") from error
    if not isinstance(header, dict):
        raise CheckpointContainerError("safetensors header must be an object")
    metadata = header.get("__metadata__")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        raise CheckpointContainerError("safetensors __metadata__ must map strings to strings")
    entries = {key: value for key, value in header.items() if key != "__metadata__"}
    if declared_tensors is not None and set(entries) != set(declared_tensors):
        missing = sorted(set(declared_tensors) - set(entries))
        extra = sorted(set(entries) - set(declared_tensors))
        raise CheckpointContainerError(
            f"safetensors names do not match tensor_manifest; missing={missing!r}, extra={extra!r}"
        )
    data_size = len(payload) - 8 - header_size
    validated_offsets: list[tuple[int, int, str]] = []
    for name, item in entries.items():
        if not isinstance(item, dict):
            raise CheckpointContainerError(f"safetensors tensor {name!r} has invalid metadata")
        offsets = item.get("data_offsets")
        shape = item.get("shape")
        dtype = item.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > data_size
            or not isinstance(shape, list)
            or any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape)
            or dtype not in _DTYPE_TO_SAFETENSORS.values()
        ):
            raise CheckpointContainerError(f"safetensors tensor {name!r} has invalid metadata")
        if declared_tensors is not None:
            declared = declared_tensors[name]
            expected_dtype = _DTYPE_TO_SAFETENSORS[declared["dtype"]]
            if shape != list(declared["shape"]):
                raise CheckpointContainerError(
                    f"safetensors tensor {name!r} shape does not match tensor_manifest"
                )
            if dtype != expected_dtype:
                raise CheckpointContainerError(
                    f"safetensors tensor {name!r} dtype does not match tensor_manifest"
                )
            element_count = math.prod(shape)
            expected_bytes = element_count * _DTYPE_BYTES[declared["dtype"]]
            if offsets[1] - offsets[0] != expected_bytes:
                raise CheckpointContainerError(
                    f"safetensors tensor {name!r} byte size does not match its shape and dtype"
                )
        validated_offsets.append((offsets[0], offsets[1], name))

    cursor = 0
    for start, end, name in sorted(validated_offsets):
        if start != cursor:
            raise CheckpointContainerError(
                f"safetensors tensor {name!r} has overlapping or non-contiguous data offsets"
            )
        cursor = end
    if cursor != data_size:
        raise CheckpointContainerError(
            "safetensors data size does not match declared tensor offsets"
        )


def write_checkpoint_container(
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any],
    tensor_payload: bytes,
) -> None:
    """Atomically write canonical metadata and safetensors as a stored ZIP."""
    metadata_payload, declared_tensors = _canonical_metadata_bytes(metadata)
    _validate_safetensors_payload(tensor_payload, declared_tensors)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with ZipFile(temporary, "w", compression=ZIP_STORED) as archive:
                archive.writestr(METADATA_MEMBER, metadata_payload, compress_type=ZIP_STORED)
                archive.writestr(TENSORS_MEMBER, tensor_payload, compress_type=ZIP_STORED)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _duplicate_metadata_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    counts = Counter(key for key, _ in pairs)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise CheckpointContainerError(
            f"metadata.json has duplicate key {duplicates[0]!r}"
        )
    return dict(pairs)


def _validate_archive_structure(archive: ZipFile) -> dict[str, ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    duplicate_names = sorted(
        name for name, count in Counter(names).items() if count > 1
    )
    if duplicate_names:
        raise CheckpointContainerError(
            f"checkpoint has duplicate member {duplicate_names[0]!r}"
        )

    for name in names:
        path = Path(name)
        if (
            name not in {METADATA_MEMBER, TENSORS_MEMBER}
            and (
                path.is_absolute()
                or "/" in name
                or "\\" in name
                or ":" in name
                or name in {".", ".."}
            )
        ):
            raise CheckpointContainerError(f"checkpoint has unsafe member name {name!r}")

    expected = {METADATA_MEMBER, TENSORS_MEMBER}
    present = set(names)
    missing = sorted(expected - present)
    if missing:
        raise CheckpointContainerError(
            f"checkpoint is missing required member {missing[0]!r}"
        )
    extra = sorted(present - expected)
    if extra:
        raise CheckpointContainerError(
            f"checkpoint has unexpected extra member {extra[0]!r}"
        )

    by_name = {info.filename: info for info in infos}
    for name, maximum in (
        (METADATA_MEMBER, MAX_METADATA_BYTES),
        (TENSORS_MEMBER, MAX_TENSOR_BYTES),
    ):
        info = by_name[name]
        if info.is_dir():
            raise CheckpointContainerError(f"checkpoint member {name!r} must be a file")
        if info.compress_type != ZIP_STORED:
            raise CheckpointContainerError(
                f"checkpoint member {name!r} must use ZIP_STORED compression"
            )
        if info.file_size > maximum or info.compress_size > maximum:
            raise CheckpointContainerError(
                f"{name} size exceeds {maximum} bytes"
            )
    return by_name


def _read_bounded_member(
    archive: ZipFile,
    info: ZipInfo,
    maximum: int,
) -> bytes:
    try:
        with archive.open(info, "r") as member:
            payload = member.read(maximum + 1)
            trailing = member.read(1)
    except (BadZipFile, OSError, RuntimeError) as error:
        raise CheckpointContainerError(
            f"checkpoint member {info.filename!r} cannot be read: {error}"
        ) from error
    if len(payload) > maximum or trailing:
        raise CheckpointContainerError(
            f"{info.filename} actual size exceeds {maximum} bytes"
        )
    if len(payload) != info.file_size:
        raise CheckpointContainerError(
            f"checkpoint member {info.filename!r} actual size does not match its ZIP declaration"
        )
    return payload


def _parse_checkpoint_metadata(payload: bytes) -> ValidatedCheckpointMetadata:
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CheckpointContainerError(
            f"metadata.json is not valid UTF-8: {error}"
        ) from error
    try:
        metadata = json.loads(
            decoded,
            object_pairs_hook=_duplicate_metadata_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, CheckpointContainerError):
            raise
        raise CheckpointContainerError(f"invalid metadata.json: {error}") from error
    if not isinstance(metadata, dict):
        raise CheckpointContainerError("metadata.json root must be an object")
    version = metadata.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != CHECKPOINT_SCHEMA_VERSION
    ):
        raise CheckpointMetadataError(
            f"$.schema_version: unsupported value {version!r}; expected integer 2"
        )
    return validate_checkpoint_metadata(metadata)


def _load_safetensors_on_cpu(payload: bytes) -> Mapping[str, Any]:
    try:
        from safetensors.torch import load

        tensors = load(payload)
    except Exception as error:
        raise CheckpointContainerError(f"cannot load tensors.safetensors: {error}") from error
    for name, tensor in tensors.items():
        device = getattr(tensor, "device", None)
        if device is None or device.type != "cpu":
            raise CheckpointContainerError(
                f"safetensors tensor {name!r} was not loaded on CPU"
            )
    return tensors


def read_checkpoint_container(
    path: str | os.PathLike[str],
    *,
    tensor_loader: Callable[[bytes], Any] | None = None,
) -> LoadedCheckpointContainer:
    """Read and fully validate a bounded checkpoint before exposing CPU tensors."""
    try:
        with ZipFile(path, "r") as archive:
            members = _validate_archive_structure(archive)
            metadata_payload = _read_bounded_member(
                archive, members[METADATA_MEMBER], MAX_METADATA_BYTES
            )
            metadata = _parse_checkpoint_metadata(metadata_payload)
            declared_tensors = {
                item["name"]: item for item in metadata.tensor_manifest
            }
            tensor_payload = _read_bounded_member(
                archive, members[TENSORS_MEMBER], MAX_TENSOR_BYTES
            )
    except CheckpointContainerError:
        raise
    except (BadZipFile, OSError, ValueError) as error:
        raise CheckpointContainerError(
            f"checkpoint is not a valid ZIP container: {error}"
        ) from error

    _validate_safetensors_payload(tensor_payload, declared_tensors)
    loader = tensor_loader or _load_safetensors_on_cpu
    tensors = loader(tensor_payload)
    return LoadedCheckpointContainer(
        metadata=_freeze(metadata.to_dict()),
        tensors=tensors,
    )
