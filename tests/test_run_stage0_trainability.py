from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from eval.stream.generators.asdiv_a import AsdivRecord
from train import run_stage0_trainability
from train.stage0_trainability import (
    ImplementationIdentity,
    MethodArm,
    OverfitProbeResult,
    StabilityMetrics,
    TrainabilityAssetIdentity,
    TrainabilityConfiguration,
    TrainabilityProgressRecord,
    TrainabilityReport,
)


def test_command_rejects_missing_stability_report(tmp_path: Path) -> None:
    """Argument validation rejects missing stability report."""
    missing_report = tmp_path / "missing.json"
    output = tmp_path / "output.json"
    with pytest.raises(FileNotFoundError):
        run_stage0_trainability.main(
            [
                "--stability-report",
                str(missing_report),
                "--final-output",
                str(output),
            ]
        )


def test_command_rejects_non_json_stability_report(tmp_path: Path) -> None:
    """Argument validation rejects malformed stability report."""
    malformed_report = tmp_path / "malformed.json"
    malformed_report.write_text("not valid json", encoding="utf-8")
    output = tmp_path / "output.json"
    with pytest.raises(ValueError, match="stability report malformed"):
        run_stage0_trainability.main(
            [
                "--stability-report",
                str(malformed_report),
                "--final-output",
                str(output),
            ]
        )


def test_command_rejects_non_object_stability_report(tmp_path: Path) -> None:
    """Argument validation rejects stability report that is not an object."""
    report = tmp_path / "report.json"
    report.write_text('["array", "not", "object"]', encoding="utf-8")
    output = tmp_path / "output.json"
    with pytest.raises(TypeError, match="root must be an object"):
        run_stage0_trainability.main(
            [
                "--stability-report",
                str(report),
                "--final-output",
                str(output),
            ]
        )


def test_command_rejects_passing_stability_report(tmp_path: Path) -> None:
    """Argument validation rejects stability report with status=passed."""
    report = tmp_path / "report.json"
    report_data = {"status": "passed"}
    report.write_text(json.dumps(report_data), encoding="utf-8")
    output = tmp_path / "output.json"
    with pytest.raises(ValueError, match="must have status=failed"):
        run_stage0_trainability.main(
            [
                "--stability-report",
                str(report),
                "--final-output",
                str(output),
            ]
        )


def test_command_rejects_missing_status_in_stability_report(tmp_path: Path) -> None:
    """Argument validation rejects stability report without status field."""
    report = tmp_path / "report.json"
    report_data = {"configuration": {}}
    report.write_text(json.dumps(report_data), encoding="utf-8")
    output = tmp_path / "output.json"
    with pytest.raises(ValueError, match="must have status=failed"):
        run_stage0_trainability.main(
            [
                "--stability-report",
                str(report),
                "--final-output",
                str(output),
            ]
        )


def test_command_rejects_existing_output_path(tmp_path: Path) -> None:
    """Argument validation rejects when output file already exists."""
    report = tmp_path / "report.json"
    report_data = {"status": "failed"}
    report.write_text(json.dumps(report_data), encoding="utf-8")
    output = tmp_path / "output.json"
    output.write_text("pre-existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_stage0_trainability.main(
            [
                "--stability-report",
                str(report),
                "--final-output",
                str(output),
            ]
        )


def test_command_rejects_existing_progress_path(tmp_path: Path) -> None:
    """Argument validation rejects when progress file already exists."""
    report = tmp_path / "report.json"
    report_data = {"status": "failed"}
    report.write_text(json.dumps(report_data), encoding="utf-8")
    output = tmp_path / "output.json"
    progress = tmp_path / "progress.jsonl"
    progress.write_text("pre-existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_stage0_trainability.main(
            [
                "--stability-report",
                str(report),
                "--final-output",
                str(output),
                "--progress-output",
                str(progress),
            ]
        )


def test_command_rejects_same_output_and_progress_path(tmp_path: Path) -> None:
    """Argument validation rejects when output and progress paths are identical."""
    report = tmp_path / "report.json"
    report_data = {"status": "failed"}
    report.write_text(json.dumps(report_data), encoding="utf-8")
    same_path = tmp_path / "output.json"
    with pytest.raises(ValueError, match="must use distinct paths"):
        run_stage0_trainability.main(
            [
                "--stability-report",
                str(report),
                "--final-output",
                str(same_path),
                "--progress-output",
                str(same_path),
            ]
        )


def test_command_rejects_implementation_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Command rejects when implementation identity cannot be computed."""
    report = tmp_path / "report.json"
    report_data = {"status": "failed"}
    report.write_text(json.dumps(report_data), encoding="utf-8")
    output = tmp_path / "output.json"
    monkeypatch.setattr(
        run_stage0_trainability,
        "canonical_trainability_implementation_identity",
        lambda _root: (_ for _ in ()).throw(ValueError("test error")),
    )
    code = run_stage0_trainability.main(
        [
            "--stability-report",
            str(report),
            "--final-output",
            str(output),
        ]
    )
    assert code == 1


def test_progress_writer_exclusive_create(tmp_path: Path) -> None:
    """Progress writer creates JSONL with exclusive-create semantics."""
    progress_path = tmp_path / "progress.jsonl"
    with run_stage0_trainability.TrainabilityProgressWriter(progress_path) as writer:
        record = TrainabilityProgressRecord(
            schema_version=1,
            sequence_number=0,
            kind="arm_checkpoint",
            probe_or_arm="gradient_only",
            consumed_examples=8,
            optimizer_call_count=8,
            elapsed_seconds=1.5,
            eta_seconds=2.0,
        )
        writer.write_record(record)
    assert progress_path.exists()
    lines = progress_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["kind"] == "arm_checkpoint"
    assert data["consumed_examples"] == 8


def test_progress_writer_append_and_flush(tmp_path: Path) -> None:
    """Progress writer appends records and flushes each write."""
    progress_path = tmp_path / "progress.jsonl"
    with run_stage0_trainability.TrainabilityProgressWriter(progress_path) as writer:
        for i in range(3):
            record = TrainabilityProgressRecord(
                schema_version=1,
                sequence_number=i,
                kind="arm_checkpoint",
                probe_or_arm=f"arm_{i}",
                consumed_examples=i * 8,
                optimizer_call_count=i * 8,
                elapsed_seconds=float(i),
                eta_seconds=None,
            )
            writer.write_record(record)
    lines = progress_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    for i, line in enumerate(lines):
        data = json.loads(line)
        assert data["sequence_number"] == i


def test_exit_code_viable_outcome() -> None:
    """Exit code is 0 for bounded_trainability_observed."""
    assert run_stage0_trainability._compute_exit_code("bounded_trainability_observed") == 0


@pytest.mark.parametrize(
    "status",
    [
        "shared_loss_behavior_conflict",
        "objective_untrainable",
        "method_specific_failure",
        "inconclusive",
    ],
)
def test_exit_code_non_viable_outcomes(status: str) -> None:
    """Exit code is nonzero for all non-viable outcomes."""
    assert run_stage0_trainability._compute_exit_code(status) == 1


def test_support_objective_forwards_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The causal support objective evaluates every record on the selected device."""
    records = tuple((f"Q{i}", f"A{i}") for i in range(8))
    observed_devices: list[str] = []

    def fake_evaluate(
        _trainer: object,
        _question: str,
        _answer: str,
        *,
        device: str,
    ) -> tuple[float, float]:
        observed_devices.append(device)
        return 1.0, 2.0

    monkeypatch.setattr(run_stage0_trainability, "evaluate_single_record_loss", fake_evaluate)

    objective = run_stage0_trainability._support_objective(object(), records, device="cuda")

    assert objective == 2.0
    assert observed_devices == ["cuda"] * 8


def test_causal_probe_normalizes_non_finite_classifier_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported non-finite classifier output becomes typed inconclusive evidence."""
    records = tuple((f"Q{i}", f"A{i}") for i in range(8))
    held_out = tuple((f"V{i}", f"B{i}") for i in range(64))
    metrics = StabilityMetrics(
        problem_count=64,
        parameter_rms=(),
        language_model_loss=1.0,
        exact_accuracy=0.0,
        first_token_accuracy=0.0,
        valid_answer_rate=1.0,
        output_diversity=0.5,
        output_dominance=0.5,
        shared_slot_variance=0.5,
        student_teacher_mse=1.0,
        student_cross_problem_cosine=0.5,
        teacher_cross_problem_cosine=0.5,
        separation_retention=1.0,
    )

    class FakeTrainer:
        last_predicted_objective_delta = -0.1

        def train_batch(self, _records: object) -> None:
            return None

    monkeypatch.setattr(run_stage0_trainability, "LatentCoreTrainer", lambda **_kwargs: FakeTrainer())
    monkeypatch.setattr(run_stage0_trainability, "_support_objective", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(run_stage0_trainability, "evaluate_held_out_metrics", lambda *_args, **_kwargs: metrics)
    monkeypatch.setattr(run_stage0_trainability, "_capture_matrix_parameters", lambda _trainer: ())
    monkeypatch.setattr(
        run_stage0_trainability,
        "classify_causal_result",
        lambda *_args: ("non_finite", ["predicted_delta_non_finite"]),
    )
    monkeypatch.setattr(
        run_stage0_trainability,
        "validate_causal_safety_checks",
        lambda *_args: ("passed", []),
    )

    result = run_stage0_trainability._run_causal_probe(
        "gradient",
        records,
        held_out,
        device="cpu",
        max_decode_tokens=4,
    )

    assert result.status == "inconclusive"
    assert "causal_non_finite" in result.failed_conditions


def test_write_final_report_uses_exclusive_create(tmp_path: Path) -> None:
    """Final report is written with exclusive-create semantics and temp file cleanup."""
    output_path = tmp_path / "report.json"
    report = TrainabilityReport(
        schema_version=1,
        configuration=TrainabilityConfiguration(
            asset_identity=TrainabilityAssetIdentity(
                stability_report_digest="a" * 64,
                held_out_record_identifiers=(("id1", "hash1"),),
                training_record_identifiers_overfit=(("id2", "hash2"),),
                training_record_identifiers_32=(("id3", "hash3"),),
            ),
            implementation=ImplementationIdentity(
                sha256="b" * 64,
                sources=(("test_source.py", "c" * 64),),
            ),
            stability_configuration={},
        ),
        asset_identity_digest="d" * 64,
        initial_state_digest="e" * 64,
        overall_status="inconclusive",
        overfit_probe=OverfitProbeResult(
            status="inconclusive",
            attempts=(),
            failed_conditions=("test",),
        ),
        causal_probes=(),
        arms=(),
        elapsed_seconds=1.0,
    )
    run_stage0_trainability._write_report_to_file(report, output_path)
    assert output_path.exists()
    assert not output_path.with_name(f".{output_path.name}.tmp").exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["overall_status"] == "inconclusive"
    with pytest.raises(FileExistsError):
        run_stage0_trainability._write_report_to_file(report, output_path)
    assert json.loads(output_path.read_text(encoding="utf-8"))["overall_status"] == "inconclusive"


def test_pre_existing_output_rejected_before_model_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-existing output path is rejected BEFORE model loading."""
    report = tmp_path / "report.json"
    report_data = {"status": "failed"}
    report.write_text(json.dumps(report_data), encoding="utf-8")
    output = tmp_path / "output.json"
    output.write_text("pre-existing", encoding="utf-8")

    model_load_called = False

    def fake_configure():
        nonlocal model_load_called
        model_load_called = True

    monkeypatch.setattr(
        run_stage0_trainability,
        "configure_deterministic_runtime",
        fake_configure,
    )

    with pytest.raises(FileExistsError):
        run_stage0_trainability.main(
            [
                "--stability-report",
                str(report),
                "--final-output",
                str(output),
            ]
        )

    assert not model_load_called, "Model loading should not occur when path exists"


def test_progress_record_schema_required_fields() -> None:
    """Progress records have all required schema fields."""
    record = TrainabilityProgressRecord(
        schema_version=1,
        sequence_number=0,
        kind="arm_checkpoint",
        probe_or_arm="gradient_only",
        consumed_examples=8,
        optimizer_call_count=8,
        elapsed_seconds=1.0,
        eta_seconds=2.0,
    )
    data = record.to_dict()
    assert "schema_version" in data
    assert "sequence_number" in data
    assert "kind" in data
    assert "probe_or_arm" in data
    assert "consumed_examples" in data
    assert "optimizer_call_count" in data
    assert "elapsed_seconds" in data
    assert "eta_seconds" in data


def test_progress_record_eta_field_validation() -> None:
    """Progress records validate ETA fields (must be non-negative or None)."""
    with pytest.raises(ValueError, match="eta_seconds must be finite"):
        TrainabilityProgressRecord(
            schema_version=1,
            sequence_number=0,
            kind="arm_checkpoint",
            probe_or_arm="gradient_only",
            consumed_examples=8,
            optimizer_call_count=8,
            elapsed_seconds=1.0,
            eta_seconds=float("inf"),
        )


def test_final_report_schema_required_fields() -> None:
    """Final reports have all required schema fields."""
    report = TrainabilityReport(
        schema_version=1,
        configuration=TrainabilityConfiguration(
            asset_identity=TrainabilityAssetIdentity(
                stability_report_digest="a" * 64,
                held_out_record_identifiers=(("id1", "hash1"),),
                training_record_identifiers_overfit=(("id2", "hash2"),),
                training_record_identifiers_32=(("id3", "hash3"),),
            ),
            implementation=ImplementationIdentity(
                sha256="b" * 64,
                sources=(("test_source.py", "c" * 64),),
            ),
            stability_configuration={},
        ),
        asset_identity_digest="d" * 64,
        initial_state_digest="e" * 64,
        overall_status="inconclusive",
        overfit_probe=OverfitProbeResult(
            status="inconclusive",
            attempts=(),
            failed_conditions=("test",),
        ),
        causal_probes=(),
        arms=(),
        elapsed_seconds=1.0,
    )
    data = report.to_dict()
    assert "schema_version" in data
    assert "overall_status" in data
    assert "elapsed_seconds" in data
    assert "overfit_probe" in data
    assert "causal_probes" in data
    assert "arms" in data


def test_final_report_exit_code_mapping() -> None:
    """Exit code mapping covers all classification statuses."""
    statuses = [
        "bounded_trainability_observed",
        "shared_loss_behavior_conflict",
        "objective_untrainable",
        "method_specific_failure",
        "inconclusive",
    ]
    for status in statuses:
        code = run_stage0_trainability._compute_exit_code(status)
        if status == "bounded_trainability_observed":
            assert code == 0, f"Status {status} should map to exit code 0"
        else:
            assert code == 1, f"Status {status} should map to exit code 1"


def test_command_routes_investigation_result_through_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command invokes its runner and routes the result to both output files."""
    stability_report = tmp_path / "stability.json"
    stability_report.write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    output = tmp_path / "output.json"
    progress = tmp_path / "progress.jsonl"
    investigation_called = False

    def run_fixture_investigation(
        _args: object,
        writer: run_stage0_trainability.TrainabilityProgressWriter,
    ) -> tuple[str, list[str]]:
        nonlocal investigation_called
        investigation_called = True
        writer.write_record(
            TrainabilityProgressRecord(
                schema_version=1,
                sequence_number=0,
                kind="arm_checkpoint",
                probe_or_arm="fixture_test",
                consumed_examples=0,
                optimizer_call_count=0,
                elapsed_seconds=0.05,
                eta_seconds=0.1,
            )
        )
        return "inconclusive", ["fixture_outcome"]

    monkeypatch.setattr(
        run_stage0_trainability,
        "configure_deterministic_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        run_stage0_trainability,
        "canonical_trainability_implementation_identity",
        lambda _root: ImplementationIdentity(
            sha256="b" * 64,
            sources=(("test_source.py", "c" * 64),),
        ),
    )
    monkeypatch.setattr(
        run_stage0_trainability,
        "_run_investigation",
        run_fixture_investigation,
    )

    code = run_stage0_trainability.main(
        [
            "--stability-report",
            str(stability_report),
            "--final-output",
            str(output),
            "--progress-output",
            str(progress),
        ]
    )

    assert investigation_called is True
    assert code == 1
    assert output.exists(), "Final JSON report should be created"
    assert progress.exists(), "JSONL progress file should be created"

    output_data = json.loads(output.read_text(encoding="utf-8"))
    assert output_data["overall_status"] == "inconclusive"
    assert output_data["schema_version"] == 1
    assert "elapsed_seconds" in output_data
    assert output_data["failed_conditions"] == ["fixture_outcome"]

    progress_text = progress.read_text(encoding="utf-8")
    assert progress_text.strip()
    progress_lines = progress_text.strip().split("\n")
    assert len(progress_lines) == 1
    progress_data = json.loads(progress_lines[0])
    assert progress_data["kind"] == "arm_checkpoint"
    assert progress_data["sequence_number"] == 0


def test_main_writes_inconclusive_report_for_runtime_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Operational investigation errors produce the required fallback report."""
    stability_report = tmp_path / "stability.json"
    stability_report.write_text('{"status": "failed"}', encoding="utf-8")
    output = tmp_path / "output.json"

    monkeypatch.setattr(run_stage0_trainability, "configure_deterministic_runtime", lambda: None)
    monkeypatch.setattr(
        run_stage0_trainability,
        "canonical_trainability_implementation_identity",
        lambda _root: ImplementationIdentity(
            sha256="b" * 64,
            sources=(("test_source.py", "c" * 64),),
        ),
    )
    monkeypatch.setattr(
        run_stage0_trainability,
        "_run_investigation",
        lambda _args, _writer: (_ for _ in ()).throw(RuntimeError("cuda setup failed")),
    )

    code = run_stage0_trainability.main(
        [
            "--stability-report",
            str(stability_report),
            "--final-output",
            str(output),
        ]
    )

    assert code == 1
    output_data = json.loads(output.read_text(encoding="utf-8"))
    assert output_data["overall_status"] == "inconclusive"
    assert output_data["failed_conditions"] == ["investigation_exception: cuda setup failed"]


def test_arm_without_evaluation_becomes_typed_inconclusive_result() -> None:
    """An arm that cannot produce a baseline remains serializable and identifiable."""
    records = tuple((f"Q{index}", f"A{index}") for index in range(32))
    arm = MethodArm(
        arm_kind="gradient_only",
        training_records=records,
        final_status="stopped",
        final_stop_reason="RuntimeError: evaluator failed",
    )

    result = run_stage0_trainability._arm_result(
        arm,
        overfit_status="passed",
        causal_status="passed",
    )

    assert result.arm == "gradient_only"
    assert result.method == "gradient"
    assert result.status == "inconclusive"
    assert result.baseline_checkpoint is None
    assert result.to_dict()["baseline"] is None


def test_real_runner_builds_typed_report_without_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real orchestration path writes typed identity and probe evidence."""
    training_records = tuple(
        AsdivRecord(id=f"train-{index}", split="train", question=f"Q{index}", target=f"A{index}") for index in range(32)
    )
    held_out_records = tuple(
        AsdivRecord(id=f"validation-{index}", split="validation", question=f"V{index}", target=f"B{index}")
        for index in range(64)
    )

    class FakeDataset:
        def training_records(self, *, mode: str, epoch: int) -> tuple[AsdivRecord, ...]:
            assert mode == "eggroll"
            assert epoch == 1
            return training_records

        def held_out_records(self) -> tuple[AsdivRecord, ...]:
            return held_out_records

    validation_calls: list[tuple[tuple[tuple[str, str], ...], dict[str, object]]] = []

    class FakeTrainer:
        def __init__(self, **_kwargs: object) -> None:
            self.state = SimpleNamespace(trainable_params={"fixture": torch.zeros(1)})
            self.optimizer = None

        def train_step(self, _question: str, _answer: str) -> SimpleNamespace:
            return SimpleNamespace(language_model_loss=1.0, total_objective=1.0)

    monkeypatch.setattr(run_stage0_trainability, "LatentCoreTrainer", FakeTrainer)
    monkeypatch.setattr(run_stage0_trainability, "load_stage0_dataset", FakeDataset)
    monkeypatch.setattr(
        run_stage0_trainability,
        "build_trainability_record_selections",
        lambda _dataset: (
            (("overfit", "d" * 64),),
            tuple((f"train-{index}", "e" * 64) for index in range(32)),
            tuple((f"validation-{index}", "f" * 64) for index in range(64)),
        ),
    )
    monkeypatch.setattr(
        run_stage0_trainability,
        "load_and_validate_stability_report",
        lambda _path, _implementation, held_out, **kwargs: (
            validation_calls.append((held_out, kwargs)) or ("a" * 64, {"fixture": True})
        ),
    )
    monkeypatch.setattr(
        run_stage0_trainability,
        "evaluate_single_record_loss",
        lambda *_args, **_kwargs: (1.0, 1.0),
    )
    monkeypatch.setattr(
        "train.stage0_trainability.evaluate_single_record_loss",
        lambda *_args, **_kwargs: (1.0, 1.0),
    )
    monkeypatch.setattr(
        run_stage0_trainability,
        "evaluate_held_out_metrics",
        lambda *_args, **_kwargs: StabilityMetrics(
            problem_count=64,
            parameter_rms=(),
            language_model_loss=1.0,
            exact_accuracy=0.0,
            first_token_accuracy=0.0,
            valid_answer_rate=1.0,
            output_diversity=0.5,
            output_dominance=0.5,
            shared_slot_variance=0.5,
            student_teacher_mse=1.0,
            student_cross_problem_cosine=0.5,
            teacher_cross_problem_cosine=0.5,
            separation_retention=1.0,
        ),
    )

    stability_report = tmp_path / "stability.json"
    stability_report.write_text('{"status": "failed"}', encoding="utf-8")
    progress_path = tmp_path / "progress.jsonl"
    output_path = tmp_path / "report.json"
    code = run_stage0_trainability.main(
        [
            "--stability-report",
            str(stability_report),
            "--final-output",
            str(output_path),
            "--progress-output",
            str(progress_path),
            "--max-decode-tokens",
            "4",
        ]
    )

    assert code == 1
    assert len(validation_calls) == 2
    assert [record_id for record_id, _content_hash in validation_calls[0][0]] == list(
        run_stage0_trainability.STAGE0_HELD_OUT_ITEM_IDS
    )
    assert validation_calls[0][1]["require_complete"] is True
    report_data = json.loads(output_path.read_text(encoding="utf-8"))
    assert report_data["configuration"]["asset_identity"]["stability_report_digest"] == "a" * 64
    assert report_data["initial_state_digest"] != "0" * 64
    assert report_data["configuration"]["asset_identity"]["held_out_record_count"] == 64
    assert report_data["overfit_probe"]["attempt_count"] == 3
    assert "investigation_incomplete" not in report_data.get("failed_conditions", [])
    progress_lines = progress_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(progress_lines) == 20
    assert sum(json.loads(line)["kind"] == "overfit_checkpoint" for line in progress_lines) == 12
