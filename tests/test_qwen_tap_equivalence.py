"""Opt-in equivalence checks against the pinned real Qwen assets."""

from __future__ import annotations

import os

import pytest
import torch

from core.qwen_tap import QwenTapAdapter
from eval.stage0_identity import (
    LATENT_TAP_LAYER,
    QWEN_MODEL,
    QWEN_REVISION,
    STAGE0_IDENTITY,
    WORKSPACE_DIMENSION,
    FrozenQwenBackbone,
    load_frozen_qwen_backbone,
)

PINNED_QWEN_TESTS_ENABLED = os.environ.get("PENSIVE_RUN_PINNED_QWEN_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not PINNED_QWEN_TESTS_ENABLED,
    reason="set PENSIVE_RUN_PINNED_QWEN_TESTS=1 to load pinned Qwen assets",
)


@pytest.fixture(scope="module")
def pinned_backbone() -> FrozenQwenBackbone:
    return load_frozen_qwen_backbone(device="cpu")


def _deterministic_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1729)
    context = torch.randn(
        (5, WORKSPACE_DIMENSION),
        generator=generator,
        dtype=torch.float32,
    ) * 0.02
    slots = torch.randn(
        (4, WORKSPACE_DIMENSION),
        generator=generator,
        dtype=torch.float32,
    ) * 0.02
    return context, slots


# covers: core/latent-loop::Latent loop feeds hidden state back as input::Cached latent state matches full execution
def test_pinned_qwen_cached_state_and_slot_gradient_match_full_execution(
    pinned_backbone: FrozenQwenBackbone,
) -> None:
    assert QWEN_MODEL == "Qwen/Qwen2.5-0.5B-Instruct"
    assert QWEN_REVISION == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert STAGE0_IDENTITY["model"] == QWEN_MODEL
    assert STAGE0_IDENTITY["model_revision"] == QWEN_REVISION
    assert pinned_backbone.config.hidden_size == WORKSPACE_DIMENSION == 896
    assert pinned_backbone.model.config.hidden_size == WORKSPACE_DIMENSION
    assert LATENT_TAP_LAYER == 12
    assert pinned_backbone.model.config._attn_implementation == "eager"
    assert all(
        parameter.dtype == torch.float32
        for parameter in pinned_backbone.model.parameters()
    )

    adapter = QwenTapAdapter(pinned_backbone.model)
    context, initial_slots = _deterministic_embeddings()
    reference_slots = initial_slots.detach().clone().requires_grad_(True)
    reference_state = adapter.full_reference(context, reference_slots)
    output_weights = torch.linspace(
        -0.5,
        0.5,
        reference_state.numel(),
        dtype=torch.float32,
    ).reshape_as(reference_state)
    reference_gradient = torch.autograd.grad(
        (reference_state * output_weights).sum(),
        reference_slots,
    )[0]

    prefix = adapter.prepare_prefix(context)
    cached_slots = initial_slots.detach().clone().requires_grad_(True)
    cached_state = adapter.cached_partial(prefix, cached_slots)
    cached_gradient = torch.autograd.grad(
        (cached_state * output_weights).sum(),
        cached_slots,
    )[0]

    torch.testing.assert_close(
        cached_state,
        reference_state,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        cached_gradient,
        reference_gradient,
        rtol=1e-4,
        atol=1e-4,
    )
