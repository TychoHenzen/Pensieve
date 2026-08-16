from __future__ import annotations

import dataclasses

import pytest

from eval.instrumentation import InstrumentationFields


def test_defaults_are_none():
    fields = InstrumentationFields()

    assert fields.halting_steps is None
    assert fields.memory_write_magnitude is None
    assert fields.channel_bandwidth is None
    assert fields.consolidation_gain is None


def test_frozen_dataclass_rejects_mutation():
    fields = InstrumentationFields()

    with pytest.raises(dataclasses.FrozenInstanceError):
        fields.halting_steps = 1
