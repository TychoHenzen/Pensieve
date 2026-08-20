"""Configuration validation for alternating training schedules."""

from __future__ import annotations


def validate_scheduler_config(
    *, phase_steps: int, epochs: int, log_every: int = 50
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
