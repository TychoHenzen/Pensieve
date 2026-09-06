"""No-download contract tests for the Calc-ASDiv_A slot-count ablation command."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from eval.gate import slot_ablation


def _records() -> tuple[SimpleNamespace, ...]:
    return (
        SimpleNamespace(id="asdiv_a__test_alpha", target="1"),
        SimpleNamespace(id="asdiv_a__test_beta", target="2"),
    )


def _prepared(records: tuple[SimpleNamespace, ...]) -> SimpleNamespace:
    identity = {
        "selection": {
            "ordered_item_ids": [record.id for record in records],
            "problem_count": len(records),
        },
        "development_only": True,
    }
    return SimpleNamespace(
        identity=identity,
        selection=SimpleNamespace(records=records),
        backbone=object(),
    )


def _latent_result(
    *,
    slot_count: int,
    seeds: list[int],
    num_steps: int,
    digest: str,
    token_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runs = []
    for index, seed in enumerate(seeds):
        correct = index + 1
        runs.append(
            {
                "seed": seed,
                "items": [{"correct": item_index < correct} for item_index in range(2)],
                "correct": correct,
                "total": 2,
            }
        )
    identity = dict(token_identity or {})
    identity.update(
        {
            "slot_count": slot_count,
            "latent_step_count": num_steps,
            "checkpoint_sha256": digest,
            "seeds": list(seeds),
        }
    )
    return {
        "schema_version": 2,
        "identity": identity,
        "checkpoint_sha256": digest,
        "seeds": list(seeds),
        "runs": runs,
    }


def test_help_parses_after_deterministic_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        slot_ablation,
        "configure_deterministic_runtime",
        lambda: events.append("runtime"),
    )

    def stop_after_parse_boundary() -> object:
        assert events == ["runtime"]
        raise RuntimeError("stop after parser ordering check")

    monkeypatch.setattr(slot_ablation, "_parse_args", stop_after_parse_boundary)
    with pytest.raises(RuntimeError, match="parser ordering"):
        slot_ablation.main()


@pytest.mark.parametrize(
    ("raw", "kind"),
    [("", "slot count"), ("1,1", "duplicate"), ("0", "positive"), ("a", "integer")],
)
def test_slot_count_parser_rejects_malformed_values(raw: str, kind: str) -> None:
    with pytest.raises(ValueError, match=kind):
        slot_ablation.parse_integer_list(raw, name="slot counts", minimum=1)


@pytest.mark.parametrize(
    ("raw", "kind"),
    [("", "seed"), ("0,0", "duplicate"), ("-1", "non-negative"), ("x", "integer")],
)
def test_seed_parser_rejects_malformed_values(raw: str, kind: str) -> None:
    with pytest.raises(ValueError, match=kind):
        slot_ablation.parse_integer_list(raw, name="seeds", minimum=0)


def test_ablation_prevalidates_all_safe_checkpoints_before_evaluation(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "slot-1.ckpt").write_bytes(b"one")
    (checkpoint_dir / "slot-4.ckpt").write_bytes(b"four")
    events: list[str] = []

    def load_checkpoint(path: Path) -> SimpleNamespace:
        events.append(f"load:{path.name}")
        count = 1 if path.name == "slot-1.ckpt" else 3
        return SimpleNamespace(tensors={"model.encoder.slot_queries": torch.zeros(count, 2)})

    with pytest.raises(ValueError, match=r"slot-4\.ckpt.*requested slot count 4"):
        slot_ablation.run_ablation(
            slot_counts=[1, 4],
            seeds=[0],
            development_limit=1,
            num_steps=2,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
            token_result_path=tmp_path / "token.json",
            records=_records(),
            baseline_preparer=lambda **_kwargs: _prepared(_records()),
            token_reader=lambda *_args, **_kwargs: {"identity": {}},
            checkpoint_loader=load_checkpoint,
            latent_runner=lambda **_kwargs: pytest.fail(
                "evaluation must not start before every checkpoint is validated"
            ),
        )

    assert events == ["load:slot-1.ckpt", "load:slot-4.ckpt"]


def test_ablation_reuses_one_current_identity_order_and_backbone(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for count in (1, 4):
        (checkpoint_dir / f"slot-{count}.ckpt").write_bytes(str(count).encode())
    records = _records()
    prepared = _prepared(records)
    events: list[Any] = []

    def prepare(**kwargs: Any) -> SimpleNamespace:
        events.append(("prepare", kwargs))
        return prepared

    def read_token(path: Path, **kwargs: Any) -> dict[str, Any]:
        events.append(("read", path, kwargs))
        return {"schema_version": 2, "identity": prepared.identity, "items": []}

    def load_checkpoint(path: Path) -> SimpleNamespace:
        count = int(path.stem.removeprefix("slot-"))
        return SimpleNamespace(tensors={"model.encoder.slot_queries": torch.zeros(count, 2)})

    def run_latent(**kwargs: Any) -> dict[str, Any]:
        events.append(("run", kwargs))
        digest = hashlib.sha256(kwargs["checkpoint_path"].read_bytes()).hexdigest()
        return _latent_result(
            slot_count=kwargs["slot_count"],
            seeds=kwargs["seeds"],
            num_steps=kwargs["num_steps"],
            digest=digest,
            token_identity=prepared.identity,
        )

    report = slot_ablation.run_ablation(
        slot_counts=[1, 4],
        seeds=[0, 1],
        development_limit=1,
        num_steps=2,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        token_result_path=tmp_path / "token.json",
        records=records,
        baseline_preparer=prepare,
        token_reader=read_token,
        checkpoint_loader=load_checkpoint,
        latent_runner=run_latent,
    )

    assert [event[0] for event in events] == ["prepare", "read", "run", "run"]
    read = events[1]
    assert read[2] == {
        "expected_identity": prepared.identity,
        "records": prepared.selection.records,
    }
    for event, count in zip(events[2:], (1, 4), strict=True):
        kwargs = event[1]
        assert kwargs["seeds"] == [0, 1]
        assert kwargs["development_limit"] == 1
        assert kwargs["num_steps"] == 2
        assert kwargs["records"] is records
        assert kwargs["backbone_loader"]() is prepared.backbone
        assert kwargs["slot_count"] == count
    assert report["schema_version"] == 1
    assert report["identity"]["token"] == prepared.identity
    assert report["identity"]["selection"] == prepared.identity["selection"]
    assert report["slot_counts"] == [1, 4]
    assert report["seeds"] == [0, 1]
    assert report["development_only"] is True
    assert report["results"]["1"]["runs"] == [
        {"seed": 0, "correct": 1, "total": 2, "accuracy": 0.5},
        {"seed": 1, "correct": 2, "total": 2, "accuracy": 1.0},
    ]
    assert report["results"]["1"]["mean"] == 0.75
    assert report["results"]["4"]["mean"] == 0.75
    assert report["single_vector_mean"] == 0.75
    assert report["best_multi_slot_mean"] == 0.75
    assert report["multi_slot_advantage"] is False


def test_missing_or_legacy_checkpoint_is_rejected_before_identity_loading(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "slot-1.pt").write_bytes(b"legacy pickle")

    with pytest.raises(FileNotFoundError, match=r"slot-1\.ckpt"):
        slot_ablation.run_ablation(
            slot_counts=[1],
            seeds=[0],
            development_limit=1,
            num_steps=2,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
            token_result_path=tmp_path / "token.json",
            records=_records(),
            baseline_preparer=lambda **_kwargs: pytest.fail("token identity must not load before checkpoints exist"),
        )


def test_inconsistent_latent_result_is_rejected_before_report_write(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "slot-1.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    records = _records()

    with pytest.raises(ValueError, match="seed order"):
        slot_ablation.run_ablation(
            slot_counts=[1],
            seeds=[0, 1],
            development_limit=1,
            num_steps=2,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
            token_result_path=tmp_path / "token.json",
            records=records,
            baseline_preparer=lambda **_kwargs: _prepared(records),
            token_reader=lambda *_args, **_kwargs: {"identity": {}},
            checkpoint_loader=lambda _path: SimpleNamespace(tensors={"model.encoder.slot_queries": torch.zeros(1, 2)}),
            latent_runner=lambda **kwargs: _latent_result(
                slot_count=1,
                seeds=[1, 0],
                num_steps=2,
                digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                token_identity=_prepared(records).identity,
            ),
        )
