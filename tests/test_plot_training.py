from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from train.plot_training import (
    ProgressRecord,
    read_progress_records,
    write_metrics_csv,
    write_training_graph,
)


def _training(step: int, method: str, loss: float, variance: float) -> dict[str, object]:
    return {
        "record_type": "training",
        "update_method": method,
        "cycle": 1,
        "global_step": step,
        "epoch": 1,
        "example_position": step - 1,
        "phase_step": step,
        "language_model_loss": loss,
        "shared_variance": variance,
    }


def _evaluation(step: int, exact_match: float) -> dict[str, object]:
    return {
        "record_type": "evaluation",
        "update_method": "gradient",
        "cycle": 2,
        "global_step": step,
        "epoch": 1,
        "example_position": step - 1,
        "phase_step": 1,
        "boundaries": ["epoch"],
        "language_model_loss": 7.5,
        "shared_variance": 0.015,
        "answer_exact_match": exact_match,
    }


def test_reads_structured_records_from_mixed_utf16_training_log(tmp_path: Path) -> None:
    path = tmp_path / "training.log"
    lines = [
        "[12:00:00] starting",
        json.dumps(_training(50, "eggroll", 9.0, 0.001)),
        "unrelated stderr",
        json.dumps(_evaluation(50, 0.25)),
        json.dumps({"record_type": "checkpoint", "paths": ["epoch-1.ckpt"]}),
    ]
    path.write_text("\n".join(lines), encoding="utf-16")

    records = read_progress_records(path)

    assert records == [
        ProgressRecord("training", 50, 1, 49, 50, 1, "eggroll", 9.0, 0.001, None),
        ProgressRecord("evaluation", 50, 1, 49, 1, 2, "gradient", 7.5, 0.015, 0.25),
    ]


def test_rejects_log_without_training_or_evaluation_records(tmp_path: Path) -> None:
    path = tmp_path / "empty.log"
    path.write_text("[12:00:00] no structured metrics\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no training or evaluation records"):
        read_progress_records(path)


def test_writes_portable_csv_and_self_contained_graph(tmp_path: Path) -> None:
    records = [
        ProgressRecord("training", 50, 1, 49, 50, 1, "eggroll", 9.0, 0.001, None),
        ProgressRecord("training", 100, 1, 99, 50, 2, "gradient", 8.0, 0.03, None),
        ProgressRecord("evaluation", 100, 1, 99, 50, 2, "gradient", 7.5, 0.02, 0.25),
    ]
    csv_path = tmp_path / "metrics.csv"
    html_path = tmp_path / "metrics.html"

    write_metrics_csv(records, csv_path)
    write_training_graph(
        records,
        html_path,
        title="Epoch 10 training",
        variance_lower_threshold=0.01,
        variance_upper_threshold=0.02,
        baseline_exact_match=0.4962,
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    html = html_path.read_text(encoding="utf-8")

    assert [row["record_type"] for row in rows] == ["training", "training", "evaluation"]
    assert rows[-1]["answer_exact_match"] == "0.25"
    assert "Epoch 10 training" in html
    assert "Language-model loss" in html
    assert "Shared slot variance (log scale)" in html
    assert "Held-out exact match" in html
    assert "49.62% baseline" in html
    assert "Eggroll" in html and "Gradient" in html and "Evaluation" in html
    assert "http://" not in html and "https://" not in html
