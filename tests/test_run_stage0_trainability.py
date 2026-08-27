from __future__ import annotations

import json
from pathlib import Path

import pytest

from train import run_stage0_trainability
from train.stage0_trainability import (
    ImplementationIdentity,
    OverfitProbeResult,
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
        run_stage0_trainability.main([
            "--stability-report",
            str(missing_report),
            "--final-output",
            str(output),
        ])


def test_command_rejects_non_json_stability_report(tmp_path: Path) -> None:
    """Argument validation rejects malformed stability report."""
    malformed_report = tmp_path / "malformed.json"
    malformed_report.write_text("not valid json", encoding="utf-8")
    output = tmp_path / "output.json"
    with pytest.raises(ValueError, match="stability report malformed"):
        run_stage0_trainability.main([
            "--stability-report",
            str(malformed_report),
            "--final-output",
            str(output),
        ])


def test_command_rejects_non_object_stability_report(tmp_path: Path) -> None:
    """Argument validation rejects stability report that is not an object."""
    report = tmp_path / "report.json"
    report.write_text('["array", "not", "object"]', encoding="utf-8")
    output = tmp_path / "output.json"
    with pytest.raises(ValueError, match="root must be an object"):
        run_stage0_trainability.main([
            "--stability-report",
            str(report),
            "--final-output",
            str(output),
        ])


def test_command_rejects_passing_stability_report(tmp_path: Path) -> None:
    """Argument validation rejects stability report with status=passed."""
    report = tmp_path / "report.json"
    report_data = {"status": "passed"}
    report.write_text(json.dumps(report_data), encoding="utf-8")
    output = tmp_path / "output.json"
    with pytest.raises(ValueError, match="must have status=failed"):
        run_stage0_trainability.main([
            "--stability-report",
            str(report),
            "--final-output",
            str(output),
        ])


def test_command_rejects_missing_status_in_stability_report(tmp_path: Path) -> None:
    """Argument validation rejects stability report without status field."""
    report = tmp_path / "report.json"
    report_data = {"configuration": {}}
    report.write_text(json.dumps(report_data), encoding="utf-8")
    output = tmp_path / "output.json"
    with pytest.raises(ValueError, match="must have status=failed"):
        run_stage0_trainability.main([
            "--stability-report",
            str(report),
            "--final-output",
            str(output),
        ])


def test_command_rejects_existing_output_path(tmp_path: Path) -> None:
    """Argument validation rejects when output file already exists."""
    report = tmp_path / "report.json"
    report_data = {"status": "failed"}
    report.write_text(json.dumps(report_data), encoding="utf-8")
    output = tmp_path / "output.json"
    output.write_text("pre-existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_stage0_trainability.main([
            "--stability-report",
            str(report),
            "--final-output",
            str(output),
        ])


def test_command_rejects_existing_progress_path(tmp_path: Path) -> None:
    """Argument validation rejects when progress file already exists."""
    report = tmp_path / "report.json"
    report_data = {"status": "failed"}
    report.write_text(json.dumps(report_data), encoding="utf-8")
    output = tmp_path / "output.json"
    progress = tmp_path / "progress.jsonl"
    progress.write_text("pre-existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_stage0_trainability.main([
            "--stability-report",
            str(report),
            "--final-output",
            str(output),
            "--progress-output",
            str(progress),
        ])


def test_command_rejects_same_output_and_progress_path(tmp_path: Path) -> None:
    """Argument validation rejects when output and progress paths are identical."""
    report = tmp_path / "report.json"
    report_data = {"status": "failed"}
    report.write_text(json.dumps(report_data), encoding="utf-8")
    same_path = tmp_path / "output.json"
    with pytest.raises(ValueError, match="must use distinct paths"):
        run_stage0_trainability.main([
            "--stability-report",
            str(report),
            "--final-output",
            str(same_path),
            "--progress-output",
            str(same_path),
        ])


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
    code = run_stage0_trainability.main([
        "--stability-report",
        str(report),
        "--final-output",
        str(output),
    ])
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


def test_pre_existing_output_rejected_before_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        run_stage0_trainability.main([
            "--stability-report",
            str(report),
            "--final-output",
            str(output),
        ])

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


def test_command_output_structure_on_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Command produces correct JSON and JSONL output structure on fixture."""
    report = tmp_path / "stability.json"
    report_data = {"status": "failed"}
    report.write_text(json.dumps(report_data), encoding="utf-8")

    output = tmp_path / "output.json"
    progress = tmp_path / "progress.jsonl"

    final_report = TrainabilityReport(
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
            failed_conditions=("fixture",),
        ),
        causal_probes=(),
        arms=(),
        elapsed_seconds=0.1,
    )

    progress_record = TrainabilityProgressRecord(
        schema_version=1,
        sequence_number=0,
        kind="arm_checkpoint",
        probe_or_arm="fixture_test",
        consumed_examples=0,
        optimizer_call_count=0,
        elapsed_seconds=0.05,
        eta_seconds=0.1,
    )

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

    run_stage0_trainability._write_report_to_file(final_report, output)
    with run_stage0_trainability.TrainabilityProgressWriter(progress) as writer:
        writer.write_record(progress_record)

    assert output.exists(), "Final JSON report should be created"
    assert progress.exists(), "JSONL progress file should be created"

    output_data = json.loads(output.read_text(encoding="utf-8"))
    assert output_data["overall_status"] == "inconclusive"
    assert output_data["schema_version"] == 1
    assert "elapsed_seconds" in output_data

    progress_lines = progress.read_text(encoding="utf-8").strip().split("\n")
    assert len(progress_lines) == 1
    progress_data = json.loads(progress_lines[0])
    assert progress_data["kind"] == "arm_checkpoint"
    assert progress_data["sequence_number"] == 0
