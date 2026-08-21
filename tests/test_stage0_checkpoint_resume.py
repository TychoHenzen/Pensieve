"""Resume contracts for version-2 Stage 0 checkpoints.

These tests use tiny tensors and injected loaders. They do not construct or
download either frozen backbone.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest
import torch
from safetensors.torch import save as save_safetensors

from eval.stage0_identity import training_identity
from train import alternating_checkpoint, stage0_checkpoint


PARAMETER_PATHS = stage0_checkpoint.ALLOWED_MODEL_PARAMETER_PATHS
RUNTIME_IDENTITY = {
    "initialization_seed": 0,
    "python_version": "3.13.5",
    "numpy_version": "2.3.2",
    "pytorch_version": "2.8.0",
    "cuda_version": "none",
    "transformers_version": "4.55.2",
    "datasets_version": "4.0.0",
    "sentence_transformers_version": "5.1.0",
    "device_topology": [],
    "dtype": "float32",
    "attention_implementation": "eager",
    "cublas_workspace_config": ":4096:8",
    "deterministic_algorithms": True,
    "tf32_enabled": False,
    "cudnn_benchmark": False,
}


def _optimizer_manifest(method: str) -> dict[str, object]:
    references = {
        path: {
            name: f"optimizer.{method}.{path}.{name}"
            for name in ("exp_avg", "exp_avg_sq")
        }
        for path in PARAMETER_PATHS
    }
    return {
        "method": method,
        "optimizer_type": "Adam",
        "parameter_names": list(PARAMETER_PATHS),
        "parameter_groups": [
            {
                "parameter_names": list(PARAMETER_PATHS),
                "scalars": {
                    "lr": 1e-4 if method == "gradient" else 1e-3,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "amsgrad": False,
                },
            }
        ],
        "scalar_state": {path: {"step": 7} for path in PARAMETER_PATHS},
        "tensor_references": references,
    }


def _metadata(*, epoch_boundary: bool = False) -> dict[str, object]:
    held_out_ids = ["validation-2", "validation-1"]
    manifest: list[dict[str, object]] = [
        {
            "name": f"model.{path}",
            "shape": [1],
            "dtype": "float32",
            "role": "model_parameter",
        }
        for path in PARAMETER_PATHS
    ]
    for method in ("gradient", "eggroll"):
        for path in PARAMETER_PATHS:
            for state_name in ("exp_avg", "exp_avg_sq"):
                manifest.append(
                    {
                        "name": f"optimizer.{method}.{path}.{state_name}",
                        "shape": [1],
                        "dtype": "float32",
                        "role": "optimizer_state",
                    }
                )
    manifest.extend(
        [
            {
                "name": "rng.numpy.state",
                "shape": [1],
                "dtype": "uint32",
                "role": "numpy_rng_state",
            },
            {
                "name": "rng.pytorch.cpu",
                "shape": [1],
                "dtype": "uint8",
                "role": "pytorch_cpu_rng_state",
            },
        ]
    )
    schedule = {
        "active_phase": "eggroll" if epoch_boundary else "gradient",
        "completed_phase_steps": 3 if epoch_boundary else 0,
        "phase_steps": 5 if epoch_boundary else 2,
        "global_step": 3 if epoch_boundary else 2,
        "epoch": 1,
        "next_dataset_position": 0 if epoch_boundary else 2,
    }
    return {
        "schema_version": 2,
        "identity": training_identity(
            runtime=RUNTIME_IDENTITY,
            held_out_item_ids=held_out_ids,
        ),
        "mode": "alternating",
        "tensor_manifest": manifest,
        "optimizer_manifests": [
            _optimizer_manifest("gradient"),
            _optimizer_manifest("eggroll"),
        ],
        "schedule": schedule,
        "selections": {
            "train": {
                "identity": "a" * 64,
                "split": "train",
                "seed": 0,
                "problem_count": 3,
                "ordered_item_ids": ["train-0", "train-1", "train-2"],
            },
            "held_out": {
                "identity": "b" * 64,
                "split": "validation",
                "seed": 0,
                "problem_count": 2,
                "ordered_item_ids": held_out_ids,
            },
        },
        "metrics": {"loss": 1.25, "boundary": "epoch" if epoch_boundary else "phase"},
        "rng": {
            "python": {"version": 3, "state": [1, 2, 3], "gaussian_cache": None},
            "numpy": {
                "bit_generator": "MT19937",
                "position": 7,
                "has_gaussian": False,
                "gaussian_cache": 0.0,
                "state_tensor": "rng.numpy.state",
            },
            "pytorch_cpu": {"state_tensor": "rng.pytorch.cpu"},
            "cuda": [],
        },
    }


def _tensor_values(metadata: dict[str, object]) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for index, item in enumerate(metadata["tensor_manifest"]):  # type: ignore[index]
        name = item["name"]  # type: ignore[index]
        dtype = item["dtype"]  # type: ignore[index]
        if dtype == "uint8":
            values[name] = torch.tensor([index % 255], dtype=torch.uint8)
        elif dtype == "uint32":
            values[name] = torch.tensor([index], dtype=torch.uint32)
        else:
            values[name] = torch.tensor([float(index)], dtype=torch.float32)
    return values


def _write_unchecked(
    path: Path,
    metadata: dict[str, object],
    tensors: dict[str, torch.Tensor] | None = None,
) -> None:
    payload = save_safetensors(tensors or _tensor_values(metadata))
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        archive.writestr(
            stage0_checkpoint.METADATA_MEMBER,
            json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        )
        archive.writestr(stage0_checkpoint.TENSORS_MEMBER, payload)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _assert_rejected_without_exposure(
    path: Path, *, expected_paths: tuple[str, ...]
) -> None:
    exposed: list[object] = []
    mutable_state = {
        "model": torch.tensor([17.0]),
        "optimizer": {"step": 11, "value": torch.tensor([23.0])},
        "schedule": {"global_step": 29},
        "metrics": [31.0],
    }
    before = copy.deepcopy(mutable_state)

    with pytest.raises(stage0_checkpoint.CheckpointContainerError) as error:
        stage0_checkpoint.read_checkpoint_container(
            path,
            tensor_loader=lambda payload: exposed.append(payload),
        )

    message = str(error.value)
    assert all(path in message for path in expected_paths), message
    assert exposed == []
    assert torch.equal(mutable_state["model"], before["model"])
    assert torch.equal(
        mutable_state["optimizer"]["value"],  # type: ignore[index]
        before["optimizer"]["value"],  # type: ignore[index]
    )
    assert mutable_state["optimizer"]["step"] == before["optimizer"]["step"]  # type: ignore[index]
    assert mutable_state["schedule"] == before["schedule"]
    assert mutable_state["metrics"] == before["metrics"]


@pytest.mark.parametrize("epoch_boundary", [False, True])
def test_compatible_v2_resume_preserves_every_state_component_and_boundary_cursor(
    tmp_path: Path, epoch_boundary: bool
) -> None:
    metadata = _metadata(epoch_boundary=epoch_boundary)
    tensors = _tensor_values(metadata)
    path = tmp_path / ("epoch-1.ckpt" if epoch_boundary else "phase-2.ckpt")
    _write_unchecked(path, metadata, tensors)

    loaded = alternating_checkpoint.load_checkpoint(path)

    assert _plain(loaded.metadata) == metadata
    assert set(loaded.tensors) == set(tensors)
    assert all(torch.equal(loaded.tensors[name], value) for name, value in tensors.items())
    assert _plain(loaded.metadata["optimizer_manifests"]) == metadata["optimizer_manifests"]
    assert _plain(loaded.metadata["rng"]) == metadata["rng"]
    assert _plain(loaded.metadata["selections"]) == metadata["selections"]
    assert _plain(loaded.metadata["metrics"]) == metadata["metrics"]
    assert _plain(loaded.metadata["schedule"]) == metadata["schedule"]


def test_legacy_pickle_checkpoint_is_rejected_without_calling_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "gsm8k-pythia.pt"
    path.write_bytes(b"legacy pickle payload")
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("torch.load must not inspect legacy checkpoints"),
    )

    with pytest.raises(
        stage0_checkpoint.CheckpointContainerError,
        match=r"checkpoint|ZIP|schema_version|legacy",
    ):
        alternating_checkpoint.load_checkpoint(path)


def test_mismatched_identities_report_every_canonical_path_before_tensor_loading(
    tmp_path: Path,
) -> None:
    metadata = _metadata()
    metadata["identity"]["dataset_revision"] = "legacy-dataset"  # type: ignore[index]
    metadata["identity"]["model"] = "EleutherAI/pythia-160m"  # type: ignore[index]
    path = tmp_path / "identity.ckpt"
    _write_unchecked(path, metadata)

    _assert_rejected_without_exposure(
        path,
        expected_paths=("$.identity.dataset_revision", "$.identity.model"),
    )


def test_unexpected_tensor_is_rejected_before_state_exposure(tmp_path: Path) -> None:
    metadata = _metadata()
    tensors = _tensor_values(metadata)
    tensors["model.legacy_adapter.weight"] = torch.tensor([99.0])
    path = tmp_path / "unexpected-tensor.ckpt"
    _write_unchecked(path, metadata, tensors)

    _assert_rejected_without_exposure(path, expected_paths=("extra",))


def test_incomplete_optimizer_state_reports_canonical_reference_path() -> None:
    metadata = _metadata()
    references = metadata["optimizer_manifests"][0]["tensor_references"]  # type: ignore[index]
    del references[PARAMETER_PATHS[0]]["exp_avg_sq"]
    mutable_state = {"optimizer_step": 11, "parameter": torch.tensor([23.0])}
    before = copy.deepcopy(mutable_state)

    with pytest.raises(stage0_checkpoint.CheckpointMetadataError) as error:
        stage0_checkpoint.validate_checkpoint_metadata(metadata)

    assert "$.optimizer_manifests[0].tensor_references" in str(error.value)
    assert mutable_state["optimizer_step"] == before["optimizer_step"]
    assert torch.equal(mutable_state["parameter"], before["parameter"])


def test_incompatible_schedule_reports_all_canonical_paths_without_mutation(
    tmp_path: Path,
) -> None:
    metadata = _metadata()
    metadata["schedule"]["completed_phase_steps"] = 500  # type: ignore[index]
    metadata["schedule"]["phase_steps"] = 0  # type: ignore[index]
    path = tmp_path / "schedule.ckpt"
    _write_unchecked(path, metadata)

    _assert_rejected_without_exposure(
        path,
        expected_paths=(
            "$.schedule.completed_phase_steps",
            "$.schedule.phase_steps",
        ),
    )
