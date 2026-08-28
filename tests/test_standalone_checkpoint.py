"""No-download resume tests for standalone Stage 0 checkpoints."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from test_stage0_checkpoint_resume import RUN_CONFIG, _metadata, _plain
from torch import nn

from train import standalone_checkpoint
from train.stage0_checkpoint import EGGROLL_MODEL_PARAMETER_PATHS, CheckpointContainerError


def _state(mode: str):
    parameters = {
        name: nn.Parameter(torch.tensor([float(index)]))
        for index, name in enumerate(standalone_checkpoint.ALLOWED_MODEL_PARAMETER_PATHS)
    }
    owned_parameters = (
        [parameters[name] for name in EGGROLL_MODEL_PARAMETER_PATHS]
        if mode == "eggroll"
        else list(parameters.values())
    )
    optimizer = (
        torch.optim.SGD(owned_parameters, lr=0.001, momentum=0.0)
        if mode == "eggroll"
        else torch.optim.Adam(owned_parameters, lr=1e-4)
    )
    for parameter in owned_parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad()
    fixture = _metadata()
    fixture["identity"]["runtime"]["device_topology"] = []
    return parameters, optimizer, fixture


def _eggroll_checkpoint_fields() -> dict[str, object]:
    run_config = copy.deepcopy(RUN_CONFIG)
    run_config.pop("phase_steps")
    run_config.pop("gradient_optimizer")
    population = run_config["eggroll_population"]
    population.pop("variance_lower_threshold")
    population.pop("variance_upper_threshold")
    return {
        "schedule": {
            "epoch": 1,
            "consumed_examples": 3,
            "eggroll_optimizer_calls": 1,
            "next_dataset_position": 0,
        },
        "run_config": run_config,
    }


@pytest.mark.parametrize("mode", ["gradient", "eggroll"])
def test_standalone_checkpoint_round_trips_complete_v2_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    parameters, optimizer, fixture = _state(mode)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    checkpoint = standalone_checkpoint.build_checkpoint(
        mode=mode,
        identity=fixture["identity"],
        selections=fixture["selections"],
        model_state=parameters,
        optimizer=optimizer,
        metrics={"avg_loss": 1.25},
        **(_eggroll_checkpoint_fields() if mode == "eggroll" else {}),
    )
    path = tmp_path / "epoch-3.ckpt"

    standalone_checkpoint.save_checkpoint(path, checkpoint)
    loaded = standalone_checkpoint.load_checkpoint(path, expected_mode=mode)

    assert loaded.metadata["schema_version"] == 2
    assert loaded.metadata["mode"] == mode
    assert _plain(loaded.metadata["schedule"]) == (
        _eggroll_checkpoint_fields()["schedule"] if mode == "eggroll" else {}
    )
    assert len(loaded.metadata["optimizer_manifests"]) == 1
    assert loaded.metadata["optimizer_manifests"][0]["method"] == mode
    assert all(
        torch.equal(loaded.tensors[f"model.{name}"], parameter.detach())
        for name, parameter in parameters.items()
    )


@pytest.mark.parametrize("mode", ["gradient", "eggroll"])
def test_compatible_standalone_resume_restores_model_and_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    parameters, optimizer, fixture = _state(mode)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    checkpoint = standalone_checkpoint.build_checkpoint(
        mode=mode,
        identity=fixture["identity"],
        selections=fixture["selections"],
        model_state=parameters,
        optimizer=optimizer,
        metrics={},
        **(_eggroll_checkpoint_fields() if mode == "eggroll" else {}),
    )
    path = tmp_path / "epoch-2.ckpt"
    standalone_checkpoint.save_checkpoint(path, checkpoint)
    loaded = standalone_checkpoint.load_checkpoint(path, expected_mode=mode)
    for parameter in parameters.values():
        parameter.data.add_(100.0)
    restored_optimizer = (
        torch.optim.SGD(
            [parameters[name] for name in EGGROLL_MODEL_PARAMETER_PATHS],
            lr=0.9,
            momentum=0.0,
        )
        if mode == "eggroll"
        else torch.optim.Adam(parameters.values(), lr=0.9)
    )

    standalone_checkpoint.validate_compatibility(
        loaded, identity=fixture["identity"], selections=fixture["selections"]
    )
    standalone_checkpoint.restore_checkpoint(
        loaded, model_parameters=parameters, optimizer=restored_optimizer
    )

    assert all(
        torch.equal(parameter, loaded.tensors[f"model.{name}"])
        for name, parameter in parameters.items()
    )
    assert restored_optimizer.param_groups[0]["lr"] == (
        0.001 if mode == "eggroll" else 1e-4
    )
    if mode == "gradient":
        assert all(
            restored_optimizer.state[parameter] for parameter in parameters.values()
        )
    else:
        assert restored_optimizer.state == {}


def test_standalone_legacy_rejection_never_calls_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "epoch-1.pt"
    path.write_bytes(b"legacy pickle")
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("torch.load must not inspect legacy state"),
    )

    with pytest.raises(CheckpointContainerError):
        standalone_checkpoint.load_checkpoint(path, expected_mode="gradient")


def test_incompatible_standalone_identity_rejects_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters, optimizer, fixture = _state("gradient")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    checkpoint = standalone_checkpoint.build_checkpoint(
        mode="gradient",
        identity=fixture["identity"],
        selections=fixture["selections"],
        model_state=parameters,
        optimizer=optimizer,
        metrics={},
    )
    before_parameters = {name: value.detach().clone() for name, value in parameters.items()}
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    incompatible = copy.deepcopy(fixture["identity"])
    incompatible["model_revision"] = "changed"

    with pytest.raises(ValueError, match=r"\$\.identity\.model_revision"):
        standalone_checkpoint.validate_compatibility(
            checkpoint, identity=incompatible, selections=fixture["selections"]
        )

    assert all(
        torch.equal(parameters[name], value) for name, value in before_parameters.items()
    )
    assert _same_state(optimizer.state_dict(), before_optimizer)


def test_standalone_mode_mismatch_rejects_before_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parameters, optimizer, fixture = _state("gradient")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    checkpoint = standalone_checkpoint.build_checkpoint(
        mode="gradient",
        identity=fixture["identity"],
        selections=fixture["selections"],
        model_state=parameters,
        optimizer=optimizer,
        metrics={},
    )
    path = tmp_path / "epoch-1.ckpt"
    standalone_checkpoint.save_checkpoint(path, checkpoint)

    with pytest.raises(ValueError, match=r"\$\.mode"):
        standalone_checkpoint.load_checkpoint(path, expected_mode="eggroll")


def test_changed_standalone_learning_rate_names_canonical_manifest_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters, optimizer, fixture = _state("gradient")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    checkpoint = standalone_checkpoint.build_checkpoint(
        mode="gradient",
        identity=fixture["identity"],
        selections=fixture["selections"],
        model_state=parameters,
        optimizer=optimizer,
        metrics={},
    )

    with pytest.raises(
        ValueError,
        match=r"\$\.optimizer_manifests\[0\]\.parameter_groups\[0\]\.scalars\.lr",
    ):
        standalone_checkpoint.validate_compatibility(
            checkpoint,
            identity=fixture["identity"],
            selections=fixture["selections"],
            learning_rate=0.2,
        )


def test_latest_standalone_checkpoint_uses_numeric_epoch_and_ignores_legacy(
    tmp_path: Path,
) -> None:
    for name in ("epoch-9.ckpt", "epoch-10.ckpt", "epoch-999.pt", "other.ckpt"):
        (tmp_path / name).write_bytes(b"not read for filename ordering")

    assert standalone_checkpoint.latest_epoch_checkpoint(tmp_path) == tmp_path / "epoch-10.ckpt"
    assert standalone_checkpoint.checkpoint_epoch(tmp_path / "epoch-10.ckpt") == 10


def _same_state(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_state(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_state(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right
