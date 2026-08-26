"""Contracts for version-2 Stage 0 checkpoint metadata.

The fixtures are plain JSON-compatible values.  They keep this suite independent
from model, dataset, CUDA, and safetensors availability.
"""

from __future__ import annotations

import copy
import importlib
import math
import re
from collections.abc import Callable
from typing import Any

import pytest

from eval.stage0_identity import training_identity


ROOT_FIELDS = {
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
ALTERNATING_ROOT_FIELDS = ROOT_FIELDS | {"run_config"}

RUN_CONFIG = {
    "dataset_selection": {
        "dataset": "MU-NLPC/Calc-mawps",
        "revision": "38c10053efeafd20ab6ff4e08c3ec17de26c19b7",
        "split": "train",
        "seed": 0,
        "count": None,
    },
    "epochs": 5,
    "phase_steps": 500,
    "model_shape": {"slot_count": 16, "num_steps": 2},
    "gradient_optimizer": {
        "type": "Adam",
        "momentum": 0.0,
        "learning_rate": 0.0001,
        "parameter_paths": [],
    },
    "eggroll_optimizer": {
        "type": "SGD",
        "momentum": 0.0,
        "learning_rate": 0.001,
        "parameter_paths": [
            "encoder.projection.weight",
            "encoder.slot_queries",
            "latent_loop.projection.weight",
        ],
    },
    "eggroll_population": {
        "size": 128,
        "sigma": 0.02,
        "rank": 4,
        "variance_weight": 1.0,
        "eval_batch_size": 8,
        "fitness_batch_size": 8,
        "use_amp": False,
    },
    "held_out_selection": {
        "dataset": "MU-NLPC/Calc-mawps",
        "revision": "38c10053efeafd20ab6ff4e08c3ec17de26c19b7",
        "split": "validation",
        "seed": 0,
        "count": 128,
    },
    "stability_report_identity": "c" * 64,
    "logging_frequency": 50,
}

ALLOWED_PARAMETER_PATHS = (
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
RUN_CONFIG["gradient_optimizer"]["parameter_paths"] = list(ALLOWED_PARAMETER_PATHS)
EGGROLL_PARAMETER_PATHS = (
    "encoder.projection.weight",
    "encoder.slot_queries",
    "latent_loop.projection.weight",
)

MODEL_SHAPES = {
    "encoder.projection.weight": [896, 384],
    "encoder.projection.bias": [896],
    "encoder.slot_queries": [16, 896],
    "encoder.attn_log_temp": [],
    "latent_loop.projection.weight": [896, 896],
    "latent_loop.projection.bias": [896],
    "latent_loop.proj_norm.weight": [896],
    "latent_loop.proj_norm.bias": [896],
    "latent_loop.layer_norm.weight": [896],
    "latent_loop.layer_norm.bias": [896],
}

RUNTIME_IDENTITY = {
    "initialization_seed": 0,
    "python_version": "3.13.5",
    "numpy_version": "2.3.2",
    "pytorch_version": "2.8.0",
    "cuda_version": "12.8",
    "transformers_version": "4.55.2",
    "datasets_version": "4.0.0",
    "sentence_transformers_version": "5.1.0",
    "device_topology": ["cuda:0", "cuda:1"],
    "dtype": "float32",
    "attention_implementation": "eager",
    "cublas_workspace_config": ":4096:8",
    "deterministic_algorithms": True,
    "tf32_enabled": False,
    "cudnn_benchmark": False,
}


def _checkpoint_module():
    return importlib.import_module("train.stage0_checkpoint")


def _tensor(name: str, shape: list[int], dtype: str, role: str) -> dict[str, Any]:
    return {"name": name, "shape": shape, "dtype": dtype, "role": role}


def _optimizer_manifest(method: str) -> dict[str, Any]:
    parameter_paths = (
        EGGROLL_PARAMETER_PATHS if method == "eggroll" else ALLOWED_PARAMETER_PATHS
    )
    tensor_references = {
        path: (
            {}
            if method == "eggroll"
            else {
                state_name: f"optimizer.{method}.{path}.{state_name}"
                for state_name in ("exp_avg", "exp_avg_sq")
            }
        )
        for path in parameter_paths
    }
    return {
        "method": method,
        "optimizer_type": "SGD" if method == "eggroll" else "Adam",
        "parameter_names": list(parameter_paths),
        "parameter_groups": [
            {
                "parameter_names": list(parameter_paths),
                "scalars": {
                    "lr": 1e-4 if method == "gradient" else 1e-3,
                    **(
                        {"momentum": 0.0}
                        if method == "eggroll"
                        else {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}
                    ),
                    "weight_decay": 0.0,
                    **({} if method == "eggroll" else {"amsgrad": False}),
                },
            }
        ],
        "scalar_state": {
            path: ({} if method == "eggroll" else {"step": 7})
            for path in parameter_paths
        },
        "tensor_references": tensor_references,
    }


def _selection(
    split: str,
    item_ids: list[str],
    identity_character: str,
) -> dict[str, Any]:
    return {
        "identity": identity_character * 64,
        "split": split,
        "seed": 0,
        "problem_count": len(item_ids),
        "ordered_item_ids": item_ids,
    }


def _metadata(mode: str = "alternating") -> dict[str, Any]:
    methods = {
        "gradient": ("gradient",),
        "eggroll": ("eggroll",),
        "alternating": ("gradient", "eggroll"),
    }[mode]
    held_out_ids = ["validation-2", "validation-1"]
    manifest = [
        _tensor(f"model.{path}", MODEL_SHAPES[path], "float32", "model_parameter")
        for path in ALLOWED_PARAMETER_PATHS
    ]
    for method in methods:
        parameter_paths = (
            EGGROLL_PARAMETER_PATHS
            if method == "eggroll"
            else ALLOWED_PARAMETER_PATHS
        )
        for path in parameter_paths:
            if method == "eggroll":
                continue
            for state_name in ("exp_avg", "exp_avg_sq"):
                manifest.append(
                    _tensor(
                        f"optimizer.{method}.{path}.{state_name}",
                        MODEL_SHAPES[path],
                        "float32",
                        "optimizer_state",
                    )
                )
    manifest.extend(
        [
            _tensor("rng.numpy.state", [624], "uint32", "numpy_rng_state"),
            _tensor("rng.pytorch.cpu", [5056], "uint8", "pytorch_cpu_rng_state"),
            _tensor("rng.pytorch.cuda.0", [5056], "uint8", "pytorch_cuda_rng_state"),
            _tensor("rng.pytorch.cuda.1", [5056], "uint8", "pytorch_cuda_rng_state"),
        ]
    )
    schedule: dict[str, Any] = {}
    if mode == "alternating":
        schedule = {
            "active_phase": "eggroll",
            "completed_phase_steps": 1,
            "phase_steps": 500,
            "global_step": 1,
            "consumed_examples": 1,
            "gradient_optimizer_calls": 0,
            "eggroll_optimizer_calls": 1,
            "epoch": 1,
            "next_dataset_position": 1,
        }
    elif mode == "eggroll":
        schedule = {
            "epoch": 1,
            "consumed_examples": 2,
            "eggroll_optimizer_calls": 1,
            "next_dataset_position": 0,
        }
    run_config = copy.deepcopy(RUN_CONFIG)
    if mode == "eggroll":
        run_config.pop("phase_steps")
        run_config.pop("gradient_optimizer")
        run_config["eggroll_population"].pop("variance_lower_threshold")
        run_config["eggroll_population"].pop("variance_upper_threshold")
    return {
        "schema_version": 2,
        "identity": training_identity(
            runtime=RUNTIME_IDENTITY,
            held_out_item_ids=held_out_ids,
        ),
        "mode": mode,
        "tensor_manifest": manifest,
        "optimizer_manifests": [_optimizer_manifest(method) for method in methods],
        "schedule": schedule,
        "selections": {
            "train": _selection("train", ["train-2", "train-1"], "a"),
            "held_out": _selection("validation", held_out_ids, "b"),
        },
        "metrics": {
            "loss": 1.25,
            "phase_complete": False,
            "boundary": "phase",
            "recent_losses": [1.5, 1.25],
            **(
                {
                    "consumed_examples": schedule["consumed_examples"],
                    "eggroll_optimizer_calls": schedule["eggroll_optimizer_calls"],
                    **(
                        {
                            "gradient_optimizer_calls": schedule[
                                "gradient_optimizer_calls"
                            ]
                        }
                        if mode == "alternating"
                        else {}
                    ),
                }
                if mode in {"eggroll", "alternating"}
                else {}
            ),
        },
        "rng": {
            "python": {
                "version": 3,
                "state": [2147483648, 766982754, 497961170],
                "gaussian_cache": None,
            },
            "numpy": {
                "bit_generator": "MT19937",
                "position": 17,
                "has_gaussian": False,
                "gaussian_cache": 0.0,
                "state_tensor": "rng.numpy.state",
            },
            "pytorch_cpu": {"state_tensor": "rng.pytorch.cpu"},
            "cuda": [
                {"device": "cuda:0", "state_tensor": "rng.pytorch.cuda.0"},
                {"device": "cuda:1", "state_tensor": "rng.pytorch.cuda.1"},
            ],
        },
        **({"run_config": run_config} if mode in {"eggroll", "alternating"} else {}),
    }


def _reject(metadata: dict[str, Any], path: str) -> None:
    module = _checkpoint_module()
    with pytest.raises(module.CheckpointMetadataError, match=re.escape(path)):
        module.validate_checkpoint_metadata(metadata)


@pytest.mark.parametrize("mode", ["gradient", "eggroll", "alternating"])
def test_valid_v2_metadata_for_every_training_mode_is_typed_and_immutable(
    mode: str,
) -> None:
    module = _checkpoint_module()
    metadata = _metadata(mode)

    validated = module.validate_checkpoint_metadata(metadata)

    assert set(metadata) == (
        ALTERNATING_ROOT_FIELDS if mode in {"eggroll", "alternating"} else ROOT_FIELDS
    )
    assert validated.schema_version == 2
    assert validated.mode == mode
    assert validated.to_dict() == metadata
    with pytest.raises((AttributeError, TypeError)):
        validated.mode = "gradient"
    with pytest.raises(TypeError):
        validated.selections["train"] = {}


def test_public_schema_constants_pin_modes_roles_and_model_names() -> None:
    module = _checkpoint_module()

    assert module.CHECKPOINT_SCHEMA_VERSION == 2
    assert module.CHECKPOINT_MODES == frozenset(
        {"gradient", "eggroll", "alternating"}
    )
    assert module.OPTIMIZER_METHODS == frozenset({"gradient", "eggroll"})
    assert module.TENSOR_ROLES == frozenset(
        {
            "model_parameter",
            "optimizer_state",
            "numpy_rng_state",
            "pytorch_cpu_rng_state",
            "pytorch_cuda_rng_state",
        }
    )
    assert module.ALLOWED_MODEL_PARAMETER_PATHS == ALLOWED_PARAMETER_PATHS


@pytest.mark.parametrize("field", sorted(ALTERNATING_ROOT_FIELDS))
def test_root_rejects_each_missing_field(field: str) -> None:
    metadata = _metadata()
    del metadata[field]

    _reject(metadata, f"$.{field}")


def test_standalone_root_rejects_alternating_run_config() -> None:
    metadata = _metadata("gradient")
    metadata["run_config"] = RUN_CONFIG

    _reject(metadata, "$.run_config")


@pytest.mark.parametrize(
    ("mutation", "path"),
    [
        (lambda config: config.pop("model_shape"), "$.run_config.model_shape"),
        (
            lambda config: config["model_shape"].update({"legacy_hidden_size": 896}),
            "$.run_config.model_shape.legacy_hidden_size",
        ),
        (
            lambda config: config["eggroll_population"].update({"sigma": float("nan")}),
            "$.run_config.eggroll_population.sigma",
        ),
        (
            lambda config: config["eggroll_population"].update({"use_amp": 1}),
            "$.run_config.eggroll_population.use_amp",
        ),
        (
            lambda config: config["dataset_selection"].update({"seed": True}),
            "$.run_config.dataset_selection.seed",
        ),
    ],
)
def test_alternating_run_config_rejects_missing_extra_nonfinite_and_wrong_types(
    mutation: Callable[[dict[str, Any]], object], path: str
) -> None:
    metadata = _metadata()
    mutation(metadata["run_config"])

    _reject(metadata, path)


@pytest.mark.parametrize("legacy_kind", ["adam", "non_matrix_state"])
# covers: train/stage0-training :: Stabilized EGGROLL identity is resumable and incompatible changes are rejected :: Reject legacy Adam Eggroll checkpoint
def test_legacy_eggroll_optimizer_state_is_rejected_before_tensor_loading(
    legacy_kind: str,
) -> None:
    metadata = _metadata()
    eggroll_manifest = metadata["optimizer_manifests"][1]
    if legacy_kind == "adam":
        eggroll_manifest["optimizer_type"] = "Adam"
    else:
        metadata["tensor_manifest"].append(
            _tensor(
                "optimizer.eggroll.encoder.projection.bias.momentum_buffer",
                MODEL_SHAPES["encoder.projection.bias"],
                "float32",
                "optimizer_state",
            )
        )

    _reject(metadata, "$.optimizer_manifests[1]")


def test_root_rejects_extra_fields() -> None:
    metadata = _metadata()
    metadata["legacy_state"] = {}

    _reject(metadata, "$.legacy_state")


@pytest.mark.parametrize("schema_version", [1, 3, True, "2", 2.0])
def test_schema_version_must_be_the_integer_two(schema_version: Any) -> None:
    metadata = _metadata()
    metadata["schema_version"] = schema_version

    _reject(metadata, "$.schema_version")


@pytest.mark.parametrize("mode", ["hybrid", "Gradient", "", 7, None])
def test_mode_must_be_one_of_the_three_declared_modes(mode: Any) -> None:
    metadata = _metadata()
    metadata["mode"] = mode

    _reject(metadata, "$.mode")


@pytest.mark.parametrize("field", ["name", "shape", "dtype", "role"])
def test_tensor_manifest_item_rejects_a_missing_field(field: str) -> None:
    metadata = _metadata()
    del metadata["tensor_manifest"][0][field]

    _reject(metadata, f"$.tensor_manifest[0].{field}")


def test_tensor_manifest_item_rejects_extra_and_wrong_typed_fields() -> None:
    metadata = _metadata()
    metadata["tensor_manifest"][0]["device"] = "cuda:0"
    metadata["tensor_manifest"][1]["shape"] = [896, True]
    metadata["tensor_manifest"][2]["dtype"] = 32
    metadata["tensor_manifest"][3]["role"] = "parameter"

    module = _checkpoint_module()
    with pytest.raises(module.CheckpointMetadataError) as error:
        module.validate_checkpoint_metadata(metadata)

    message = str(error.value)
    assert "$.tensor_manifest[0].device" in message
    assert "$.tensor_manifest[1].shape[1]" in message
    assert "$.tensor_manifest[2].dtype" in message
    assert "$.tensor_manifest[3].role" in message


def test_tensor_manifest_rejects_duplicate_names() -> None:
    metadata = _metadata()
    metadata["tensor_manifest"][1]["name"] = metadata["tensor_manifest"][0]["name"]

    _reject(metadata, "$.tensor_manifest[1].name")


def test_model_manifest_requires_every_allowed_trainable_name_exactly_once() -> None:
    metadata = _metadata()
    metadata["tensor_manifest"][0]["name"] = "model.backbone.layers.0.weight"

    _reject(metadata, "$.tensor_manifest[0].name")


def test_model_manifest_rejects_a_missing_allowed_trainable_parameter() -> None:
    metadata = _metadata()
    missing_name = "model.encoder.projection.bias"
    metadata["tensor_manifest"] = [
        item for item in metadata["tensor_manifest"] if item["name"] != missing_name
    ]

    _reject(metadata, f"$.tensor_manifest.{missing_name}")


def test_optimizer_tensor_names_follow_method_parameter_and_state_paths() -> None:
    metadata = _metadata("gradient")
    item = next(
        item
        for item in metadata["tensor_manifest"]
        if item["role"] == "optimizer_state"
    )
    item["name"] = "optimizer.gradient.exp_avg.encoder.projection.weight"

    _reject(metadata, "$.tensor_manifest")


@pytest.mark.parametrize(
    "field",
    [
        "method",
        "optimizer_type",
        "parameter_names",
        "parameter_groups",
        "scalar_state",
        "tensor_references",
    ],
)
def test_optimizer_manifest_rejects_a_missing_field(field: str) -> None:
    metadata = _metadata("gradient")
    del metadata["optimizer_manifests"][0][field]

    _reject(metadata, f"$.optimizer_manifests[0].{field}")


def test_optimizer_manifests_reject_duplicate_methods() -> None:
    metadata = _metadata("alternating")
    metadata["optimizer_manifests"][1]["method"] = "gradient"

    _reject(metadata, "$.optimizer_manifests[1].method")


def test_optimizer_manifest_requires_stable_unique_allowed_parameter_order() -> None:
    metadata = _metadata("gradient")
    names = metadata["optimizer_manifests"][0]["parameter_names"]
    names[1] = names[0]

    _reject(metadata, "$.optimizer_manifests[0].parameter_names[1]")


def test_alternating_optimizers_must_share_the_same_parameter_order() -> None:
    metadata = _metadata("alternating")
    second = metadata["optimizer_manifests"][1]["parameter_names"]
    second[0], second[1] = second[1], second[0]

    _reject(metadata, "$.optimizer_manifests[1].parameter_names")


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("scalars", {"lr": [1e-4]}),
        ("scalars", {"lr": math.inf}),
        ("parameter_names", ["encoder.projection.bias"]),
    ],
)
def test_optimizer_groups_accept_only_finite_scalars_for_their_named_parameters(
    field: str,
    bad_value: Any,
) -> None:
    metadata = _metadata("gradient")
    metadata["optimizer_manifests"][0]["parameter_groups"][0][field] = bad_value

    _reject(metadata, f"$.optimizer_manifests[0].parameter_groups[0].{field}")


def test_optimizer_scalar_state_rejects_containers_and_nonfinite_values() -> None:
    metadata = _metadata("gradient")
    state = metadata["optimizer_manifests"][0]["scalar_state"]
    state[ALLOWED_PARAMETER_PATHS[0]]["step"] = [7]
    state[ALLOWED_PARAMETER_PATHS[1]]["step"] = math.nan

    module = _checkpoint_module()
    with pytest.raises(module.CheckpointMetadataError) as error:
        module.validate_checkpoint_metadata(metadata)

    assert "$.optimizer_manifests[0].scalar_state" in str(error.value)


def test_optimizer_tensor_references_must_exist_and_match_their_declared_state() -> None:
    metadata = _metadata("gradient")
    references = metadata["optimizer_manifests"][0]["tensor_references"]
    references[ALLOWED_PARAMETER_PATHS[0]]["exp_avg"] = "model.encoder.projection.bias"

    _reject(metadata, "$.optimizer_manifests[0].tensor_references")


def test_rng_metadata_covers_python_numpy_cpu_and_ordered_cuda_devices() -> None:
    module = _checkpoint_module()
    validated = module.validate_checkpoint_metadata(_metadata())

    assert validated.rng["python"]["version"] == 3
    assert validated.rng["numpy"]["bit_generator"] == "MT19937"
    assert validated.rng["pytorch_cpu"]["state_tensor"] == "rng.pytorch.cpu"
    assert [item["device"] for item in validated.rng["cuda"]] == [
        "cuda:0",
        "cuda:1",
    ]


@pytest.mark.parametrize("source", ["python", "numpy", "pytorch_cpu", "cuda"])
def test_rng_metadata_rejects_each_missing_source(source: str) -> None:
    metadata = _metadata()
    del metadata["rng"][source]

    _reject(metadata, f"$.rng.{source}")


@pytest.mark.parametrize(
    ("path", "mutate"),
    [
        (
            "$.rng.python.state[1]",
            lambda rng: rng["python"]["state"].__setitem__(1, True),
        ),
        (
            "$.rng.python.gaussian_cache",
            lambda rng: rng["python"].__setitem__("gaussian_cache", math.inf),
        ),
        (
            "$.rng.numpy.position",
            lambda rng: rng["numpy"].__setitem__("position", -1),
        ),
        (
            "$.rng.numpy.has_gaussian",
            lambda rng: rng["numpy"].__setitem__("has_gaussian", 0),
        ),
        (
            "$.rng.numpy.gaussian_cache",
            lambda rng: rng["numpy"].__setitem__("gaussian_cache", math.nan),
        ),
        (
            "$.rng.pytorch_cpu.state_tensor",
            lambda rng: rng["pytorch_cpu"].__setitem__(
                "state_tensor", "rng.numpy.state"
            ),
        ),
    ],
)
def test_rng_metadata_rejects_wrong_types_bounds_and_references(
    path: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    metadata = _metadata()
    mutate(metadata["rng"])

    _reject(metadata, path)


def test_rng_metadata_rejects_duplicate_or_unordered_cuda_topology() -> None:
    metadata = _metadata()
    metadata["rng"]["cuda"][1]["device"] = "cuda:0"

    _reject(metadata, "$.rng.cuda[1].device")


def test_rng_references_require_uint8_tensors_with_matching_roles() -> None:
    metadata = _metadata()
    cpu_tensor = next(
        item for item in metadata["tensor_manifest"] if item["name"] == "rng.pytorch.cpu"
    )
    cpu_tensor["dtype"] = "float32"
    cpu_tensor["role"] = "optimizer_state"

    _reject(metadata, "$.rng.pytorch_cpu.state_tensor")


@pytest.mark.parametrize("mode", ["gradient", "eggroll"])
def test_standalone_modes_require_an_empty_schedule(mode: str) -> None:
    metadata = _metadata(mode)
    metadata["schedule"] = {"global_step": 1}

    _reject(metadata, "$.schedule")


@pytest.mark.parametrize(
    "field",
    [
        "active_phase",
        "completed_phase_steps",
        "phase_steps",
        "global_step",
        "epoch",
        "next_dataset_position",
    ],
)
def test_alternating_schedule_rejects_each_missing_field(field: str) -> None:
    metadata = _metadata()
    del metadata["schedule"][field]

    _reject(metadata, f"$.schedule.{field}")


def test_alternating_schedule_rejects_extra_and_wrong_typed_fields() -> None:
    metadata = _metadata()
    metadata["schedule"]["dataset_position"] = 1006
    metadata["schedule"]["epoch"] = True

    module = _checkpoint_module()
    with pytest.raises(module.CheckpointMetadataError) as error:
        module.validate_checkpoint_metadata(metadata)

    message = str(error.value)
    assert "$.schedule.dataset_position" in message
    assert "$.schedule.epoch" in message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("active_phase", "token"),
        ("completed_phase_steps", 500),
        ("phase_steps", 0),
        ("global_step", -1),
        ("epoch", 0),
        ("next_dataset_position", -1),
    ],
)
def test_alternating_schedule_rejects_invalid_or_inconsistent_values(
    field: str,
    value: Any,
) -> None:
    metadata = _metadata()
    metadata["schedule"][field] = value

    _reject(metadata, f"$.schedule.{field}")


@pytest.mark.parametrize("selection", ["train", "held_out"])
def test_selections_require_both_named_entries(selection: str) -> None:
    metadata = _metadata()
    del metadata["selections"][selection]

    _reject(metadata, f"$.selections.{selection}")


def test_selection_requires_identity_split_seed_count_and_ordered_ids() -> None:
    metadata = _metadata()
    del metadata["selections"]["train"]["identity"]
    metadata["selections"]["train"]["ordered_item_ids"].append("train-2")
    metadata["selections"]["held_out"]["split"] = "test"

    module = _checkpoint_module()
    with pytest.raises(module.CheckpointMetadataError) as error:
        module.validate_checkpoint_metadata(metadata)

    message = str(error.value)
    assert "$.selections.train.identity" in message
    assert "$.selections.train.ordered_item_ids[2]" in message
    assert "$.selections.held_out.split" in message


def test_selection_count_and_checkpoint_identity_match_the_ordered_ids() -> None:
    metadata = _metadata()
    metadata["selections"]["held_out"]["problem_count"] = 3
    metadata["identity"]["held_out_item_ids"] = ["validation-1", "validation-2"]

    module = _checkpoint_module()
    with pytest.raises(module.CheckpointMetadataError) as error:
        module.validate_checkpoint_metadata(metadata)

    message = str(error.value)
    assert "$.selections.held_out.problem_count" in message
    assert "$.identity.held_out_item_ids" in message


@pytest.mark.parametrize(
    "bad_value",
    [None, {"nested": "mapping"}, math.nan, math.inf, [1, [2]], [object()]],
)
def test_metrics_reject_values_outside_bounded_json_scalars_and_flat_lists(
    bad_value: Any,
) -> None:
    metadata = _metadata()
    metadata["metrics"]["invalid"] = bad_value

    _reject(metadata, "$.metrics.invalid")


def test_metrics_reject_overlong_strings() -> None:
    module = _checkpoint_module()
    metadata = _metadata()
    metadata["metrics"]["note"] = "x" * (module.MAX_METADATA_STRING_LENGTH + 1)

    with pytest.raises(module.CheckpointMetadataError, match=r"\$\.metrics\.note"):
        module.validate_checkpoint_metadata(metadata)
