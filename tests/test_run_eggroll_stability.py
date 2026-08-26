from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eval.stream.generators.asdiv_a import AsdivRecord
from train import run_eggroll_stability
from train.eggroll_stability import (
    BASELINE_PROBLEM_COUNT,
    ImplementationIdentity,
    ParameterRms,
    StabilityMetrics,
    StabilityReportValidationError,
    build_stability_asset_identity,
    build_stability_configuration,
    load_compatible_stability_report,
)
from train.training_state import EGGROLL_PARAMETER_PATHS


class FixtureWorkspace:
    def __init__(self) -> None:
        self.slots = torch.tensor([[1.0, 2.0]])

    def snapshot(self) -> torch.Tensor:
        return self.slots.clone()

    def restore(self, state: torch.Tensor) -> None:
        self.slots = state.clone()


class FixtureState:
    def __init__(self) -> None:
        self.trainable_params = {
            path: nn.Parameter(torch.full((2, 2), float(index + 1)))
            for index, path in enumerate(EGGROLL_PARAMETER_PATHS)
        }
        self.workspace = FixtureWorkspace()

    def eggroll_parameters(self):
        return iter(self.trainable_params.values())


class FixtureTrainer:
    def __init__(self) -> None:
        self.state = FixtureState()
        self.optimizer = torch.optim.SGD(
            self.state.eggroll_parameters(), lr=0.1, momentum=0.0
        )
        self.fitness_batch_size = 8
        self.pop_size = 128
        self.sigma = 0.001
        self.rank = 4
        self.eval_batch_size = 8
        self.variance_weight = 1.0
        self.prompt_alignment_weight = 0.0
        self.seen_ids: list[str] = []

    def train_fitness_batch(
        self,
        fitness_batch,
        position=None,
        *,
        optimizer_call_count: int = 1,
    ):
        del position, optimizer_call_count
        self.seen_ids.extend(record.id for record in fitness_batch.records)
        with torch.no_grad():
            for parameter in self.state.eggroll_parameters():
                parameter.add_(0.001)
        return SimpleNamespace(
            consumed_record_count=fitness_batch.consumed_record_count
        )


def _records(split: str, count: int) -> tuple[AsdivRecord, ...]:
    return tuple(
        AsdivRecord(
            id=f"{split}-{index:03d}",
            split=split,
            question=f"question {index}",
            target=str(index),
        )
        for index in range(count)
    )


def _metrics(
    trainer: FixtureTrainer,
    *,
    loss: float,
    exact: float,
    first: float,
    separation: float = 0.8,
) -> StabilityMetrics:
    return StabilityMetrics(
        problem_count=BASELINE_PROBLEM_COUNT,
        parameter_rms=tuple(
            ParameterRms(
                path,
                float(torch.sqrt(torch.mean(parameter.detach().square())).item()),
            )
            for path, parameter in trainer.state.trainable_params.items()
        ),
        language_model_loss=loss,
        exact_accuracy=exact,
        first_token_accuracy=first,
        valid_answer_rate=0.75,
        output_diversity=0.5,
        output_dominance=0.25,
        shared_slot_variance=0.02,
        student_teacher_mse=0.4,
        student_cross_problem_cosine=0.6,
        teacher_cross_problem_cosine=0.5,
        separation_retention=separation,
    )


def _install_fixture(
    monkeypatch: pytest.MonkeyPatch,
    evaluator,
) -> tuple[FixtureTrainer, tuple[AsdivRecord, ...]]:
    trainer = FixtureTrainer()
    held_out = _records("validation", BASELINE_PROBLEM_COUNT)
    monkeypatch.setattr(
        run_eggroll_stability,
        "_load_fresh_records",
        lambda: (_records("train", 300), held_out),
    )
    monkeypatch.setattr(run_eggroll_stability, "_build_trainer", lambda _args: trainer)
    monkeypatch.setattr(run_eggroll_stability, "evaluate_stability_metrics", evaluator)
    monkeypatch.setattr(run_eggroll_stability, "configure_deterministic_runtime", lambda: None)
    monkeypatch.setattr(
        run_eggroll_stability,
        "runtime_identity",
        lambda: {"initialization_seed": 0, "device": "fixture"},
    )
    monkeypatch.setattr(
        run_eggroll_stability,
        "canonical_implementation_identity",
        lambda _root: ImplementationIdentity(
            sha256="a" * 64,
            sources=(("train/eggroll_stability.py", "b" * 64),),
        ),
    )
    return trainer, held_out


@pytest.mark.parametrize(
    ("outcome", "expected_status", "expected_code", "expected_checkpoints"),
    (
        ("passing", "passed", 0, [0, 8, 32, 256]),
        ("early_regression", "failed", 1, [0, 8]),
        ("missing_improvement", "failed", 1, [0, 8, 32, 256]),
    ),
)
def test_public_command_retains_structured_fixture_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: str,
    expected_status: str,
    expected_code: int,
    expected_checkpoints: list[int],
) -> None:
    def evaluator(trainer, _problems, *, device, max_decode_tokens):
        assert device == "cpu"
        assert max_decode_tokens > 0
        if not trainer.seen_ids:
            return _metrics(trainer, loss=1.0, exact=0.0, first=0.0)
        if outcome == "early_regression":
            return _metrics(trainer, loss=1.051, exact=0.0, first=0.0)
        if outcome == "missing_improvement":
            return _metrics(trainer, loss=0.9, exact=0.0, first=0.0)
        return _metrics(trainer, loss=0.9, exact=0.125, first=0.25)

    _install_fixture(monkeypatch, evaluator)
    report_path = tmp_path / f"{outcome}.json"
    progress_path = tmp_path / f"{outcome}.jsonl"
    code = run_eggroll_stability.main(
        [
            "--output",
            str(report_path),
            "--progress-output",
            str(progress_path),
            "--device",
            "cpu",
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    progress = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads(capsys.readouterr().out.strip())

    assert code == expected_code
    assert report["status"] == expected_status
    assert report["outcome_code"] == expected_code
    assert [item["checkpoint"]["consumed_examples"] for item in progress] == expected_checkpoints
    assert summary["kind"] == "eggroll_stability_summary"
    assert summary["status"] == expected_status
    assert summary["outcome_code"] == expected_code
    assert summary["report"] == str(report_path)
    assert summary["progress"] == str(progress_path)


@pytest.mark.parametrize(
    ("mutations", "expected_paths"),
    (
        (
            (("implementation", "sha256", "c" * 64),),
            ("$.configuration.implementation.sha256",),
        ),
        (
            (
                ("eggroll", "population", 64),
                ("eggroll", "sigma", 0.002),
            ),
            (
                "$.configuration.eggroll.population",
                "$.configuration.eggroll.sigma",
            ),
        ),
    ),
)
def test_stale_or_mismatched_fixture_is_rejected_before_runtime_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutations,
    expected_paths,
) -> None:
    def evaluator(trainer, _problems, *, device, max_decode_tokens):
        del device, max_decode_tokens
        if not trainer.seen_ids:
            return _metrics(trainer, loss=1.0, exact=0.0, first=0.0)
        return _metrics(trainer, loss=0.9, exact=0.125, first=0.25)

    trainer, held_out = _install_fixture(monkeypatch, evaluator)
    report_path = tmp_path / "passing.json"
    assert run_eggroll_stability.main(
        ["--output", str(report_path), "--device", "cpu"]
    ) == 0
    expected = build_stability_configuration(
        trainer,
        asset_identity=build_stability_asset_identity(
            runtime={"initialization_seed": 0, "device": "fixture"},
            held_out_records=held_out,
        ),
        implementation=ImplementationIdentity(
            sha256="a" * 64,
            sources=(("train/eggroll_stability.py", "b" * 64),),
        ),
        initialization_seed=0,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    for section, field, value in mutations:
        payload["configuration"][section][field] = value
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        run_eggroll_stability,
        "_build_trainer",
        lambda _args: (_ for _ in ()).throw(AssertionError("constructed model")),
    )
    monkeypatch.setattr(
        run_eggroll_stability,
        "_load_fresh_records",
        lambda: (_ for _ in ()).throw(AssertionError("accessed dataset")),
    )

    with pytest.raises(StabilityReportValidationError) as caught:
        load_compatible_stability_report(report_path, expected)

    assert tuple(
        path for path in expected_paths if any(path in issue for issue in caught.value.issues)
    ) == expected_paths
    assert all(path in str(caught.value) for path in expected_paths)


def test_public_command_rejects_one_shared_artifact_path_before_runtime_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "shared.json"

    def unexpected_runtime_access(*_args, **_kwargs):
        raise AssertionError("accessed runtime, model, or dataset")

    monkeypatch.setattr(
        run_eggroll_stability,
        "configure_deterministic_runtime",
        unexpected_runtime_access,
    )
    monkeypatch.setattr(
        run_eggroll_stability,
        "canonical_implementation_identity",
        unexpected_runtime_access,
    )
    monkeypatch.setattr(
        run_eggroll_stability,
        "_load_fresh_records",
        unexpected_runtime_access,
    )
    monkeypatch.setattr(
        run_eggroll_stability,
        "_build_trainer",
        unexpected_runtime_access,
    )

    with pytest.raises(ValueError, match="must use distinct paths"):
        run_eggroll_stability.main(
            [
                "--output",
                str(artifact),
                "--progress-output",
                str(artifact),
                "--device",
                "cpu",
            ]
        )

    assert not artifact.exists()
    assert list(tmp_path.iterdir()) == []
