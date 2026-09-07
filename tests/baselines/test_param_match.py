from __future__ import annotations

import pytest

from eval.baselines.model import MLP
from eval.baselines.param_match import ArchReport, check_tolerance, compare, count_parameters


def test_count_parameters_matches_manual_calculation() -> None:
    model = MLP(input_dim=10, output_dim=5, hidden_layers=2, hidden_units=20)

    expected = (10 * 20 + 20) + (20 * 20 + 20) + (20 * 5 + 5)

    assert count_parameters(model) == expected


def test_check_tolerance_true_within_bound() -> None:
    assert check_tolerance(subject_params=100_000, baseline_params=105_000, tolerance=0.1) is True


def test_check_tolerance_false_beyond_bound() -> None:
    assert check_tolerance(subject_params=100_000, baseline_params=500_000, tolerance=0.1) is False


def test_compare_raises_when_parameters_mismatch_beyond_tolerance() -> None:
    subject_report = ArchReport(name="subject", param_count=100_000, depth=2, width=400, optimizer="adam")
    baseline_report = ArchReport(name="baseline", param_count=500_000, depth=2, width=400, optimizer="adam")

    with pytest.raises(ValueError):
        compare(subject_report, baseline_report, tolerance=0.1)


def test_compare_returns_matched_report_within_tolerance() -> None:
    subject_report = ArchReport(name="subject", param_count=100_000, depth=2, width=400, optimizer="adam")
    baseline_report = ArchReport(name="baseline", param_count=105_000, depth=2, width=400, optimizer="adam")

    result = compare(subject_report, baseline_report, tolerance=0.1)

    assert result["subject"] is subject_report
    assert result["baseline"] is baseline_report
    assert result["matched"] is True


def test_arch_report_contains_all_expected_fields() -> None:
    report = ArchReport(name="subject", param_count=123, depth=2, width=400, optimizer="adam")

    assert report.name == "subject"
    assert report.param_count == 123
    assert report.depth == 2
    assert report.width == 400
    assert report.optimizer == "adam"
