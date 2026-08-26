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
