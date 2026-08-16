from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_tolerance(subject_params: int, baseline_params: int, tolerance: float) -> bool:
    """Return True if baseline_params is within `tolerance` fraction of subject_params."""
    if subject_params == 0:
        return baseline_params == 0
    diff = abs(subject_params - baseline_params) / subject_params
    return diff <= tolerance


@dataclass
class ArchReport:
    """Architecture summary for a single model."""

    name: str
    param_count: int
    depth: int
    width: int
    optimizer: str


def compare(subject_report: ArchReport, baseline_report: ArchReport, tolerance: float) -> dict[str, Any]:
    """Compare a subject's and a baseline's architecture reports.

    Raises ValueError if the parameter counts differ by more than `tolerance`.
    """
    matched = check_tolerance(subject_report.param_count, baseline_report.param_count, tolerance)
    if not matched:
        raise ValueError(
            "Parameter counts differ by more than tolerance "
            f"({tolerance:.1%}): subject={subject_report.param_count}, "
            f"baseline={baseline_report.param_count}"
        )
    return {
        "subject": subject_report,
        "baseline": baseline_report,
        "matched": matched,
    }
