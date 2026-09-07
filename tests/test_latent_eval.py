"""Focused tests for latent-gate scoring and safe checkpoint loading."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eval.gate import latent_eval
from eval.gate.answer_scoring import score_numerical_answer
from eval.stream.generators.asdiv_a import AsdivRecord, select_asdiv_a_records


def test_numerical_answer_scoring_extracts_last_generated_number() -> None:
    assert score_numerical_answer("work: 12 + 23 = 35", "35")
    assert score_numerical_answer("The answer is 1,250", "1250")
    assert not score_numerical_answer("work: 12 + 23 = 35", "23")
    assert not score_numerical_answer("no numerical answer", "35")


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2)
        self.slot_queries = nn.Parameter(torch.zeros(4, 2))
        self.attn_log_temp = nn.Parameter(torch.zeros(()))


class _LatentLoop(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2)
        self.proj_norm = nn.LayerNorm(2)
        self.layer_norm = nn.LayerNorm(2)


class _Subject:
    def __init__(self, **kwargs: object) -> None:
        self.arguments = kwargs
        self.encoder = _Encoder()
        self.latent_loop = _LatentLoop()


def test_load_subject_uses_safe_v2_checkpoint_loader(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_encoder = _Encoder()
    source_latent = _LatentLoop()
    tensors = {
        **{f"model.encoder.{name}": tensor.detach().clone() for name, tensor in source_encoder.state_dict().items()},
        **{f"model.latent_loop.{name}": tensor.detach().clone() for name, tensor in source_latent.state_dict().items()},
    }
    checkpoint_path = tmp_path / "alternating.ckpt"
    checkpoint_path.write_bytes(b"safe-container-placeholder")
    calls: list[object] = []

    def safe_loader(path: object) -> object:
        calls.append(path)
        return SimpleNamespace(tensors=tensors)

    monkeypatch.setattr(latent_eval, "load_alternating_checkpoint", safe_loader)
    monkeypatch.setattr(latent_eval, "LatentCoreSubject", _Subject)
    loaded = latent_eval.load_subject(
        checkpoint_path,
        slot_count=16,
        num_steps=2,
        device="cpu",
        backbone=object(),
    )

    assert calls == [checkpoint_path]
    assert loaded.arguments["slot_count"] == 4
    for name, tensor in source_encoder.state_dict().items():
        assert torch.equal(loaded.encoder.state_dict()[name], tensor), name
    for name, tensor in source_latent.state_dict().items():
        assert torch.equal(loaded.latent_loop.state_dict()[name], tensor), name


def test_load_subject_rejects_legacy_pickle_suffix_before_loading(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"model_state": {}}, path)
    monkeypatch.setattr(
        latent_eval,
        "load_alternating_checkpoint",
        lambda *_args: pytest.fail("legacy pickle must not reach the safe loader"),
    )

    with pytest.raises(ValueError, match=r"version-2.*\.ckpt"):
        latent_eval.load_subject(path, 16, 2, "cpu", backbone=object())


def test_run_eval_configures_runtime_before_subject_construction() -> None:
    records = tuple(
        AsdivRecord(
            id=f"asdiv_a__test_{index:03d}",
            split="test",
            question=f"What is {index} plus zero?",
            target=str(index),
        )
        for index in range(520)
    )
    ordered = select_asdiv_a_records({"test": records}, split="test", seed=0, problem_count=None).ordered_item_ids
    token_result = {
        "schema_version": 2,
        "identity": {
            "selection": {
                "split": "test",
                "seed": 0,
                "problem_count": len(ordered),
                "ordered_item_ids": list(ordered),
            },
            "development_only": False,
        },
        "items": [{"item_id": item_id} for item_id in ordered],
    }
    events: list[str] = []

    def configure_runtime() -> None:
        events.append("runtime")

    def load_subject(*_args: object, **_kwargs: object) -> object:
        assert events == ["runtime"]
        events.append("subject")
        return object()

    latent_eval.run_eval(
        checkpoint_path=None,
        seeds=[0],
        development_limit=1,
        slot_count=16,
        num_steps=2,
        device="cuda",
        token_result=token_result,
        records=records,
        subject_loader=load_subject,
        item_evaluator=lambda _subject, record: {
            "prediction": record.target,
            "input_ids": [1, 2, 3],
        },
        runtime_configurer=configure_runtime,
    )

    assert events == ["runtime", "subject"]


def test_main_configures_runtime_before_parsing_device_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(latent_eval, "configure_deterministic_runtime", lambda: events.append("runtime"))

    def stop_after_parse_boundary() -> object:
        assert events == ["runtime"]
        events.append("parse")
        raise RuntimeError("stop after parser ordering check")

    monkeypatch.setattr(latent_eval, "_parse_args", stop_after_parse_boundary)
    with pytest.raises(RuntimeError, match="parser ordering"):
        latent_eval.main()

    assert events == ["runtime", "parse"]
