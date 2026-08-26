from __future__ import annotations

import json
from pathlib import Path

import pytest

from train import run_stage0_trainability


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
        from train.stage0_trainability import TrainabilityProgressRecord
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
    from train.stage0_trainability import TrainabilityProgressRecord
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
    from train.stage0_trainability import (
        TrainabilityReport,
        TrainabilityConfiguration,
        TrainabilityAssetIdentity,
        ImplementationIdentity,
        OverfitProbeResult,
    )

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
    run_stage0_trainability._write_final_report(output_path, report)
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
