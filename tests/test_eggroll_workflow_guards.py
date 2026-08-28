from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.eggroll_stability_fixtures import write_stability_report
from train import run_alternating, run_eggroll
from train.eggroll_stability import StabilityReportValidationError
from train.standalone_checkpoint import configure_deterministic_runtime


def _eggroll_args(report: Path | None) -> SimpleNamespace:
    return SimpleNamespace(
        epochs=1,
        slot_count=16,
        num_steps=2,
        pop_size=128,
        sigma=0.001,
        lr=0.1,
        rank=4,
        variance_weight=1.0,
        prompt_alignment_weight=0.1,
        eval_batch_size=8,
        fitness_batch_size=8,
        use_amp=False,
        device="cpu",
        save_dir="unused",
        problem_count=None,
        log_every=10,
        resume=None,
        stability_report=None if report is None else str(report),
    )


def _alternating_args(report: Path | None) -> SimpleNamespace:
    return SimpleNamespace(
        epochs=1,
        phase_steps=50,
        variance_lower_threshold=0.01,
        variance_upper_threshold=0.02,
        slot_count=16,
        num_steps=2,
        gradient_lr=0.001,
        eggroll_lr=0.1,
        pop_size=128,
        sigma=0.001,
        rank=4,
        variance_weight=1.0,
        prompt_alignment_weight=0.1,
        eval_batch_size=8,
        fitness_batch_size=8,
        use_amp=False,
        eval_problem_count=128,
        problem_count=4,
        device="cpu",
        log_every=50,
        save_dir="unused",
        resume=None,
        stability_report=None if report is None else str(report),
    )


def _stale_report(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["configuration"]["implementation"]["sha256"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("outcome", ["missing", "failed", "stale", "mismatched"])
def test_long_standalone_rejects_invalid_report_before_dataset_or_model_access(
    outcome: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_deterministic_runtime()
    report_path = tmp_path / f"{outcome}.json"
    report: Path | None = report_path
    if outcome == "missing":
        report = None
    elif outcome == "failed":
        write_stability_report(
            report_path, status="failed", prompt_alignment_weight=0.1
        )
    elif outcome == "stale":
        write_stability_report(report_path, prompt_alignment_weight=0.1)
        _stale_report(report_path)
    else:
        write_stability_report(
            report_path, population=64, prompt_alignment_weight=0.1
        )
    accesses: list[str] = []
    monkeypatch.setattr(run_eggroll, "_parse_args", lambda: _eggroll_args(report))
    monkeypatch.setattr(
        run_eggroll,
        "_load_dataset_context",
        lambda _count: accesses.append("dataset"),
    )
    monkeypatch.setattr(
        run_eggroll,
        "EggrollTrainer",
        lambda **_kwargs: accesses.append("model"),
    )

    with pytest.raises((ValueError, StabilityReportValidationError)) as error:
        run_eggroll.main()

    assert accesses == []
    assert "stability" in str(error.value).lower()
    if outcome == "mismatched":
        assert "$.configuration.eggroll.population" in str(error.value)
    if outcome == "stale":
        assert "$.configuration.implementation.sha256" in str(error.value)


def test_standalone_development_run_at_256_examples_needs_no_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _eggroll_args(None)
    args.problem_count = 256
    accesses: list[str] = []
    monkeypatch.setattr(run_eggroll, "_parse_args", lambda: args)
    monkeypatch.setattr(
        run_eggroll,
        "_load_dataset_context",
        lambda _count: accesses.append("dataset") or (_ for _ in ()).throw(
            RuntimeError("development dataset reached")
        ),
    )

    with pytest.raises(RuntimeError, match="development dataset reached"):
        run_eggroll.main()

    assert accesses == ["dataset"]


@pytest.mark.parametrize("outcome", ["missing", "failed", "stale", "mismatched"])
def test_alternating_rejects_invalid_report_before_checkpoint_dataset_or_model_access(
    outcome: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_deterministic_runtime()
    report_path = tmp_path / f"{outcome}.json"
    report: Path | None = report_path
    if outcome == "missing":
        report = None
    elif outcome == "failed":
        write_stability_report(
            report_path, status="failed", prompt_alignment_weight=0.1
        )
    elif outcome == "stale":
        write_stability_report(report_path, prompt_alignment_weight=0.1)
        _stale_report(report_path)
    else:
        write_stability_report(
            report_path, sigma=0.002, prompt_alignment_weight=0.1
        )
    accesses: list[str] = []
    monkeypatch.setattr(
        run_alternating, "_parse_args", lambda _argv=None: _alternating_args(report)
    )
    monkeypatch.setattr(
        run_alternating,
        "load_checkpoint",
        lambda _path: accesses.append("checkpoint"),
    )
    monkeypatch.setattr(
        run_alternating,
        "load_stage0_dataset",
        lambda: accesses.append("dataset"),
    )
    monkeypatch.setattr(
        run_alternating,
        "TrainingState",
        lambda *_args: accesses.append("model"),
    )

    with pytest.raises((ValueError, StabilityReportValidationError)) as error:
        run_alternating.main([])

    assert accesses == []
    assert "stability" in str(error.value).lower()
    if outcome == "mismatched":
        assert "$.configuration.eggroll.sigma" in str(error.value)
    if outcome == "stale":
        assert "$.configuration.implementation.sha256" in str(error.value)
