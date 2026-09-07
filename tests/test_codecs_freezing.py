from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from torch import nn

from codecs_module.encoder import SlotEncoder
from core.latent_loop import LatentLoop
from workspace.concept_slots import SLOT_DIM

FAKE_HIDDEN_DIM = SLOT_DIM


class _FakeConfig:
    hidden_size = FAKE_HIDDEN_DIM


class _FakeModel(nn.Module):
    """Stand-in for AutoModelForCausalLM, mirrors test_latent_loop.py's fixture."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeConfig()
        self.linear = nn.Linear(FAKE_HIDDEN_DIM, FAKE_HIDDEN_DIM)

    def forward(self, **kwargs: Any) -> Any:
        raise NotImplementedError("not needed for freezing tests")

    def __call__(self, **kwargs: Any) -> Any:
        return self.forward(**kwargs)


@pytest.fixture(scope="module")
def encoder() -> SlotEncoder:
    return SlotEncoder(
        slot_count=4,
        device="cpu",
        manifest_verifier=lambda *_: None,
    )


@pytest.fixture
def loop() -> LatentLoop:
    fake_model = _FakeModel()
    with patch(
        "core.latent_loop.AutoModelForCausalLM.from_pretrained",
        return_value=fake_model,
    ):
        latent_loop = LatentLoop(num_steps=3, device="cpu")
    return latent_loop


def test_encoder_sentence_model_params_frozen(encoder: SlotEncoder) -> None:
    params = list(encoder._sentence_model.parameters())
    assert len(params) > 0
    assert all(not param.requires_grad for param in params)


def test_encoder_projection_params_trainable(encoder: SlotEncoder) -> None:
    assert encoder.projection.weight.requires_grad
    assert encoder.projection.bias.requires_grad


def test_encoder_slot_queries_trainable(encoder: SlotEncoder) -> None:
    assert encoder.slot_queries.requires_grad


def test_latent_loop_model_params_frozen(loop: LatentLoop) -> None:
    params = list(loop.model.parameters())
    assert len(params) > 0
    assert all(not param.requires_grad for param in params)


def test_latent_loop_projection_params_trainable(loop: LatentLoop) -> None:
    assert loop.projection.weight.requires_grad
    assert loop.projection.bias.requires_grad
