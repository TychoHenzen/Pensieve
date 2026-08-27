"""Configuration validation for alternating training schedules."""

from __future__ import annotations

import math

DEFAULT_VARIANCE_LOWER_THRESHOLD = 0.01
DEFAULT_VARIANCE_UPPER_THRESHOLD = 0.02


def validate_scheduler_config(
    *,
    phase_steps: int,
    epochs: int,
    log_every: int = 50,
    variance_lower_threshold: float = DEFAULT_VARIANCE_LOWER_THRESHOLD,
    variance_upper_threshold: float = DEFAULT_VARIANCE_UPPER_THRESHOLD,
) -> None:
    """Reject alternating schedule values that cannot produce training work.

    Commands call this before constructing models or loading a dataset.
    """
    if phase_steps < 1:
        raise ValueError("phase_steps must be at least 1")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if log_every < 1:
        raise ValueError("--log-every must be at least 1")
    if (
        not math.isfinite(variance_lower_threshold)
        or variance_lower_threshold < 0.0
        or not math.isfinite(variance_upper_threshold)
        or variance_upper_threshold <= variance_lower_threshold
    ):
        raise ValueError(
            "variance thresholds must be finite and satisfy "
            "0 <= lower < upper"
        )
