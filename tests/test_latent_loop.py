from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

from core.latent_loop import LatentLoop
from workspace.concept_slots import SLOT_DIM, Workspace

FAKE_HIDDEN_DIM = SLOT_DIM


class _FakeOutputs:
    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states


class _FakeConfig:
    hidden_size = FAKE_HIDDEN_DIM


class _FakeModel(nn.Module):
    """Stand-in for AutoModelForCausalLM that records how it was called.

    Mirrors the shape contract of a real causal LM forward pass without
    downloading or running Pythia-160M, so tests stay fast.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeConfig()
        self.linear = nn.Linear(FAKE_HIDDEN_DIM, FAKE_HIDDEN_DIM)
        self.calls: list[dict[str, Any]] = []

    def forward(self, **kwargs: Any) -> _FakeOutputs:
        self.calls.append(kwargs)
        inputs_embeds = kwargs["inputs_embeds"]
        hidden = self.linear(inputs_embeds)
        return _FakeOutputs(hidden_states=(hidden, hidden))

    def __call__(self, **kwargs: Any) -> _FakeOutputs:
        return self.forward(**kwargs)


@pytest.fixture
def loop() -> LatentLoop:
    fake_model = _FakeModel()
    with patch(
        "core.latent_loop.AutoModelForCausalLM.from_pretrained",
        return_value=fake_model,
    ):
        latent_loop = LatentLoop(num_steps=3, device="cpu")
    return latent_loop


def test_step_passes_only_inputs_embeds_not_input_ids(loop: LatentLoop) -> None:
    workspace = Workspace(slot_count=4)
    workspace.write_slots(torch.randn(4, SLOT_DIM))

    loop.step(workspace)

    calls = loop.model.calls
    assert len(calls) == 1
    call_kwargs = calls[0]
    assert "inputs_embeds" in call_kwargs
    assert "input_ids" not in call_kwargs


def test_step_output_shape_matches_input_slots(loop: LatentLoop) -> None:
    workspace = Workspace(slot_count=4)
    workspace.write_slots(torch.randn(4, SLOT_DIM))

    result = loop.step(workspace)

    assert result.shape == (4, SLOT_DIM)
    assert result.dtype.is_floating_point
    assert workspace.read_slots().shape == (4, SLOT_DIM)


def test_step_with_context_embeds_output_shape(loop: LatentLoop) -> None:
    workspace = Workspace(slot_count=4)
    workspace.write_slots(torch.randn(4, SLOT_DIM))
    context = torch.randn(10, SLOT_DIM)

    result = loop.step(workspace, context_embeds=context)

    assert result.shape == (4, SLOT_DIM)
    assert workspace.read_slots().shape == (4, SLOT_DIM)


def test_step_with_context_embeds_prepends_context(loop: LatentLoop) -> None:
    workspace = Workspace(slot_count=4)
    workspace.write_slots(torch.randn(4, SLOT_DIM))
    context = torch.randn(10, SLOT_DIM)

    loop.step(workspace, context_embeds=context)

    call_kwargs = loop.model.calls[0]
    assert call_kwargs["inputs_embeds"].shape == (1, 14, SLOT_DIM)


def test_run_calls_step_exactly_num_steps_times(loop: LatentLoop) -> None:
    workspace = Workspace(slot_count=4)
    workspace.write_slots(torch.randn(4, SLOT_DIM))

    loop.run(workspace)

    assert len(loop.model.calls) == loop.num_steps == 3
    for call_kwargs in loop.model.calls:
        assert "input_ids" not in call_kwargs


def test_get_cost_steps_matches_num_steps_after_run(loop: LatentLoop) -> None:
    workspace = Workspace(slot_count=4)
    workspace.write_slots(torch.randn(4, SLOT_DIM))

    loop.run(workspace)

    assert loop.get_cost().steps == loop.num_steps == 3


def test_reset_cost_zeroes_counters(loop: LatentLoop) -> None:
    workspace = Workspace(slot_count=4)
    workspace.write_slots(torch.randn(4, SLOT_DIM))

    loop.run(workspace)
    loop.reset_cost()

    cost = loop.get_cost()
    assert cost.steps == 0
    assert cost.flops == 0
