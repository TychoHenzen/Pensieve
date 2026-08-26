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
