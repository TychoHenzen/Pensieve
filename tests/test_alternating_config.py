"""Tests for alternating training scheduler configuration."""

import pytest

from train.alternating_config import validate_scheduler_config


@pytest.mark.parametrize("phase_steps", [0, -1])
def test_rejects_non_positive_phase_budget(phase_steps: int) -> None:
    with pytest.raises(ValueError, match="phase_steps must be at least 1"):
        validate_scheduler_config(phase_steps=phase_steps, epochs=1)


@pytest.mark.parametrize("epochs", [0, -1])
def test_rejects_non_positive_epoch_count(epochs: int) -> None:
    with pytest.raises(ValueError, match="epochs must be at least 1"):
        validate_scheduler_config(phase_steps=1, epochs=epochs)


def test_accepts_positive_phase_budget_and_epoch_count() -> None:
    validate_scheduler_config(phase_steps=500, epochs=5, log_every=50)


@pytest.mark.parametrize("log_every", [0, -1])
def test_rejects_non_positive_log_interval(log_every: int) -> None:
    with pytest.raises(ValueError, match="--log-every must be at least 1"):
        validate_scheduler_config(phase_steps=500, epochs=5, log_every=log_every)
