"""Fake-only contracts for Qwen hidden-state latent-loop feedback."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

from core.latent_loop import LatentLoop
from core.qwen_tap import QwenTapAdapter
from workspace.concept_slots import Workspace


WIDTH = 896
CONTEXT_LENGTH = 3
SLOT_COUNT = 4
SEQUENCE_LENGTH = CONTEXT_LENGTH + SLOT_COUNT


class _FakeQwen(nn.Module):
    """A Qwen-shaped backbone that records only latent-loop operations."""

    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=WIDTH, num_hidden_layers=24)
        self.hidden_states = hidden_states
        self.calls: list[dict[str, Any]] = []
        self.generated = 0
        self.decoded = 0
        self.embedding_lookups = 0

    def forward(self, **kwargs: Any) -> Any:
        self.calls.append(
            {
                name: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
        )
        return SimpleNamespace(hidden_states=self.hidden_states)

    def generate(self, *_: Any, **__: Any) -> None:
        self.generated += 1
        raise AssertionError("latent steps must not generate tokens")

    def decode(self, *_: Any, **__: Any) -> None:
        self.decoded += 1
        raise AssertionError("latent steps must not decode tokens")

    def get_input_embeddings(self) -> nn.Module:
        self.embedding_lookups += 1
        raise AssertionError("latent steps must not look up token embeddings")


class _DifferentiableQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=WIDTH, num_hidden_layers=24)
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.calls: list[dict[str, Any]] = []

    def forward(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        hidden = kwargs["inputs_embeds"] * self.scale
        return SimpleNamespace(hidden_states=tuple(hidden for _ in range(25)))


class _CacheProducingQwen(nn.Module):
    def __init__(
        self,
        *,
        cache_layer_count: int = 24,
        cached_sequence_delta: int = 0,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=WIDTH, num_hidden_layers=24)
        self.cache_layer_count = cache_layer_count
        self.cached_sequence_delta = cached_sequence_delta
        self.calls: list[dict[str, Any]] = []

    def forward(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        context = kwargs["inputs_embeds"]
        cache = kwargs["past_key_values"]
        cached_length = context.shape[1] + self.cached_sequence_delta
        for layer_index in range(self.cache_layer_count):
            values = torch.arange(
                2 * cached_length * 4,
                dtype=context.dtype,
                device=context.device,
            ).reshape(1, 2, cached_length, 4)
            cache.update(values + layer_index, values - layer_index, layer_index)
        return SimpleNamespace(past_key_values=cache)


def _hidden_states(
    hidden_at_tap: torch.Tensor | None = None,
    *,
    count: int = 13,
) -> tuple[torch.Tensor, ...]:
    default = torch.full((1, SEQUENCE_LENGTH, WIDTH), -3.0)
    states = [default.clone() for _ in range(count)]
    if count > 12:
        states[12] = (
            hidden_at_tap.clone() if hidden_at_tap is not None else default.clone()
        )
    return tuple(states)


def _loop(hidden_states: tuple[torch.Tensor, ...], *, num_steps: int = 1) -> LatentLoop:
    model = _FakeQwen(hidden_states)
    return LatentLoop(
        backbone=SimpleNamespace(model=model, tokenizer=object()),
        num_steps=num_steps,
        device="cpu",
    )


def _workspace() -> Workspace:
    workspace = Workspace(slot_count=SLOT_COUNT)
    values = torch.arange(SLOT_COUNT * WIDTH, dtype=torch.float32).reshape(
        SLOT_COUNT, WIDTH
    )
    workspace.write_slots(values / WIDTH)
    return workspace


def test_qwen_step_uses_one_unpadded_context_and_slot_batch() -> None:
    loop = _loop(_hidden_states())
    workspace = _workspace()
    context = torch.full((CONTEXT_LENGTH, WIDTH), 2.0)

    loop.step(workspace, context)

    call = loop.model.calls[0]
    assert len(loop.model.calls) == 1
    assert torch.equal(
        call["inputs_embeds"], torch.cat((context, _workspace().read_slots())).unsqueeze(0)
    )
    assert call["inputs_embeds"].shape == (1, SEQUENCE_LENGTH, WIDTH)
    assert call["attention_mask"].shape == (1, SEQUENCE_LENGTH)
    assert torch.all(call["attention_mask"] == 1)
    assert torch.equal(
        call["position_ids"], torch.arange(SEQUENCE_LENGTH).unsqueeze(0)
    )
    assert call["output_hidden_states"] is True
    assert "input_ids" not in call


def test_qwen_tap_full_reference_preserves_unbatched_slot_autograd() -> None:
    model = _DifferentiableQwen()
    adapter = QwenTapAdapter(model)
    context = torch.ones(CONTEXT_LENGTH, WIDTH)
    slots = torch.linspace(-1.0, 1.0, SLOT_COUNT * WIDTH).reshape(
        SLOT_COUNT, WIDTH
    )
    slots.requires_grad_(True)

    selected = adapter.full_reference(context, slots)
    selected.square().sum().backward()

    assert selected.shape == slots.shape
    assert torch.equal(selected, slots.detach() * 2.0)
    assert slots.grad is not None
    assert torch.count_nonzero(slots.grad).item() > 0
    assert model.scale.requires_grad is False
    call = model.calls[0]
    assert call["inputs_embeds"].shape == (1, SEQUENCE_LENGTH, WIDTH)
    assert torch.equal(
        call["position_ids"], torch.arange(SEQUENCE_LENGTH).unsqueeze(0)
    )
    assert torch.all(call["attention_mask"] == 1)
    assert call["output_hidden_states"] is True


def test_qwen_tap_full_reference_accepts_candidate_slots_and_shared_context() -> None:
    candidate_count = 3
    model = _DifferentiableQwen()
    adapter = QwenTapAdapter(model)
    context = torch.ones(CONTEXT_LENGTH, WIDTH)
    slots = torch.arange(
        candidate_count * SLOT_COUNT * WIDTH,
        dtype=torch.float32,
    ).reshape(candidate_count, SLOT_COUNT, WIDTH)

    selected = adapter.full_reference(context, slots)

    assert selected.shape == slots.shape
    assert torch.equal(selected, slots * 2.0)
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["inputs_embeds"].shape == (
        candidate_count,
        SEQUENCE_LENGTH,
        WIDTH,
    )
    assert call["attention_mask"].shape == (candidate_count, SEQUENCE_LENGTH)
    assert call["position_ids"].shape == (candidate_count, SEQUENCE_LENGTH)


def test_qwen_tap_full_reference_validates_tapped_hidden_shape() -> None:
    adapter = QwenTapAdapter(
        _FakeQwen(_hidden_states(torch.zeros(1, SEQUENCE_LENGTH, WIDTH - 1)))
    )

    with pytest.raises(
        ValueError,
        match=(
            f"expected hidden state shape \\(1, {SEQUENCE_LENGTH}, {WIDTH}\\), "
            f"actual \\(1, {SEQUENCE_LENGTH}, {WIDTH - 1}\\)"
        ),
    ):
        adapter.full_reference(
            torch.ones(CONTEXT_LENGTH, WIDTH),
            torch.ones(SLOT_COUNT, WIDTH),
        )


# covers: core/latent-loop::Latent loop feeds hidden state back as input::Cached context is immutable
def test_qwen_tap_prefix_is_detached_and_candidate_caches_are_independent() -> None:
    model = _CacheProducingQwen()
    adapter = QwenTapAdapter(model)
    context = torch.arange(
        CONTEXT_LENGTH * WIDTH,
        dtype=torch.float32,
    ).reshape(CONTEXT_LENGTH, WIDTH)
    context.requires_grad_(True)

    prefix = adapter.prepare_prefix(context)
    original = tuple(
        (key.clone(), value.clone()) for key, value in prefix.layer_key_values
    )
    first_cache = adapter.fresh_candidate_cache(prefix, candidate_batch_size=3)

    assert prefix.context_length == CONTEXT_LENGTH
    assert len(prefix.layer_key_values) == 12
    assert all(
        not tensor.requires_grad and tensor.grad_fn is None
        for layer in prefix.layer_key_values
        for tensor in layer
    )
    assert context.grad is None
    call = model.calls[0]
    assert call["use_cache"] is True
    assert torch.equal(
        call["attention_mask"], torch.ones(1, CONTEXT_LENGTH, dtype=torch.long)
    )
    assert torch.equal(
        call["position_ids"], torch.arange(CONTEXT_LENGTH).unsqueeze(0)
    )
    for layer_index, (stored_key, stored_value) in enumerate(
        prefix.layer_key_values
    ):
        assert torch.equal(
            first_cache.layers[layer_index].keys,
            stored_key.expand(3, -1, -1, -1),
        )
        assert torch.equal(
            first_cache.layers[layer_index].values,
            stored_value.expand(3, -1, -1, -1),
        )
        assert (
            first_cache.layers[layer_index].keys.untyped_storage().data_ptr()
            == stored_key.untyped_storage().data_ptr()
        )
        assert (
            first_cache.layers[layer_index].values.untyped_storage().data_ptr()
            == stored_value.untyped_storage().data_ptr()
        )

    appended_key = torch.zeros(3, 2, 1, 4)
    appended_value = torch.ones(3, 2, 1, 4)
    first_cache.update(appended_key, appended_value, layer_idx=0)
    second_cache = adapter.fresh_candidate_cache(prefix, candidate_batch_size=3)

    assert first_cache.layers[0].keys.shape[-2] == CONTEXT_LENGTH + 1
    assert (
        first_cache.layers[0].keys.untyped_storage().data_ptr()
        != prefix.layer_key_values[0][0].untyped_storage().data_ptr()
    )
    for layer_index, (original_key, original_value) in enumerate(original):
        stored_key, stored_value = prefix.layer_key_values[layer_index]
        assert torch.equal(stored_key, original_key)
        assert torch.equal(stored_value, original_value)
        assert torch.equal(
            second_cache.layers[layer_index].keys,
            original_key.expand(3, -1, -1, -1),
        )
        assert torch.equal(
            second_cache.layers[layer_index].values,
            original_value.expand(3, -1, -1, -1),
        )


def test_qwen_tap_candidate_layout_uses_full_visibility_and_slot_positions() -> None:
    adapter = QwenTapAdapter(_CacheProducingQwen())
    prefix = adapter.prepare_prefix(torch.ones(CONTEXT_LENGTH, WIDTH))

    attention_mask, position_ids = adapter.candidate_layout(
        prefix,
        candidate_batch_size=3,
        slot_count=SLOT_COUNT,
    )

    assert attention_mask.shape == (3, SEQUENCE_LENGTH)
    assert torch.all(attention_mask == 1)
    assert torch.equal(
        position_ids,
        torch.arange(CONTEXT_LENGTH, SEQUENCE_LENGTH).unsqueeze(0).expand(3, -1),
    )


@pytest.mark.parametrize(
    ("context", "model", "expected"),
    [
        (
            torch.ones(2, CONTEXT_LENGTH, WIDTH),
            _CacheProducingQwen(),
            "context embeddings must be unbatched or have batch size one",
        ),
        (
            torch.ones(CONTEXT_LENGTH, WIDTH - 1),
            _CacheProducingQwen(),
            f"expected context embedding width {WIDTH}, actual {WIDTH - 1}",
        ),
        (
            torch.ones(CONTEXT_LENGTH, WIDTH),
            _CacheProducingQwen(cache_layer_count=11),
            "expected cache for 12 layers, actual 11",
        ),
        (
            torch.ones(CONTEXT_LENGTH, WIDTH),
            _CacheProducingQwen(cached_sequence_delta=-1),
            "expected layer 0 cache shapes",
        ),
    ],
    ids=("candidate-context", "wrong-width", "missing-layer", "wrong-cache-shape"),
)
def test_qwen_tap_prefix_validates_context_and_cache_shapes(
    context: torch.Tensor,
    model: _CacheProducingQwen,
    expected: str,
) -> None:
    adapter = QwenTapAdapter(model)

    with pytest.raises(ValueError, match=expected):
        adapter.prepare_prefix(context)


# covers: core/latent-loop::Latent loop feeds hidden state back as input::hidden state feedback
def test_qwen_step_uses_hidden_state_12_final_slot_slice_and_declared_equation() -> None:
    hidden = torch.arange(
        SEQUENCE_LENGTH * WIDTH, dtype=torch.float32
    ).reshape(1, SEQUENCE_LENGTH, WIDTH) / 1000
    loop = _loop(_hidden_states(hidden))
    workspace = _workspace()
    slots = workspace.read_slots().clone()
    with torch.no_grad():
        loop.projection.weight.copy_(torch.eye(WIDTH))
        loop.projection.bias.copy_(torch.linspace(-0.3, 0.3, WIDTH))
        loop.proj_norm.weight.copy_(torch.linspace(0.2, 0.8, WIDTH))
        loop.proj_norm.bias.copy_(torch.linspace(-0.1, 0.1, WIDTH))
        loop.layer_norm.weight.copy_(torch.linspace(0.4, 1.0, WIDTH))
        loop.layer_norm.bias.copy_(torch.linspace(-0.2, 0.2, WIDTH))

    updated = loop.step(workspace, torch.ones(CONTEXT_LENGTH, WIDTH))

    h = loop.model.hidden_states[12][0, -SLOT_COUNT:, :]
    u = functional.layer_norm(
        functional.linear(h, loop.projection.weight, loop.projection.bias),
        (WIDTH,),
        loop.proj_norm.weight,
        loop.proj_norm.bias,
        eps=1e-5,
    )
    expected = functional.layer_norm(
        u + 0.5 * slots,
        (WIDTH,),
        loop.layer_norm.weight,
        loop.layer_norm.bias,
        eps=1e-5,
    )
    assert loop.proj_norm.eps == loop.layer_norm.eps == 1e-5
    assert torch.allclose(updated, expected)
    assert torch.equal(workspace.read_slots(), expected)


@pytest.mark.parametrize(
    ("hidden_states", "expected"),
    [
        (_hidden_states(count=12), "expected hidden-state tuple with index 12, actual length 12"),
        (
            _hidden_states(torch.zeros(2, SEQUENCE_LENGTH, WIDTH)),
            f"expected hidden state shape (1, {SEQUENCE_LENGTH}, {WIDTH}), actual (2, {SEQUENCE_LENGTH}, {WIDTH})",
        ),
        (
            _hidden_states(torch.zeros(1, SEQUENCE_LENGTH - 1, WIDTH)),
            f"expected hidden state shape (1, {SEQUENCE_LENGTH}, {WIDTH}), actual (1, {SEQUENCE_LENGTH - 1}, {WIDTH})",
        ),
        (
            _hidden_states(torch.zeros(1, SEQUENCE_LENGTH, WIDTH - 1)),
            f"expected hidden state shape (1, {SEQUENCE_LENGTH}, {WIDTH}), actual (1, {SEQUENCE_LENGTH}, {WIDTH - 1})",
        ),
    ],
    ids=("missing-tap", "wrong-batch", "wrong-sequence", "wrong-width"),
)
# covers: core/latent-loop::Latent loop feeds hidden state back as input::selected tap layer unavailable
def test_qwen_step_rejects_each_incompatible_hidden_state_shape(
    hidden_states: tuple[torch.Tensor, ...], expected: str
) -> None:
    loop = _loop(hidden_states)

    with pytest.raises(ValueError, match=expected):
        loop.step(_workspace(), torch.ones(CONTEXT_LENGTH, WIDTH))


# covers: core/latent-loop::Latent loop feeds hidden state back as input::no intermediate tokens
def test_qwen_run_reuses_updated_slots_without_intermediate_tokens() -> None:
    hidden = torch.arange(
        SEQUENCE_LENGTH * WIDTH, dtype=torch.float32
    ).reshape(1, SEQUENCE_LENGTH, WIDTH)
    loop = _loop(_hidden_states(hidden), num_steps=3)
    workspace = _workspace()
    context = torch.full((CONTEXT_LENGTH, WIDTH), 4.0)
    initial = workspace.read_slots().clone()
    first_update = loop.layer_norm(
        loop.proj_norm(loop.projection(hidden[0, -SLOT_COUNT:, :])) + 0.5 * initial
    )

    loop.run(workspace, context)

    calls = loop.model.calls
    assert len(calls) == loop.num_steps == 3
    for call in calls:
        assert "input_ids" not in call
        assert call["inputs_embeds"].shape == (1, SEQUENCE_LENGTH, WIDTH)
    assert torch.equal(
        calls[0]["inputs_embeds"][0, -SLOT_COUNT:, :],
        initial,
    )
    assert torch.equal(
        calls[1]["inputs_embeds"][0, -SLOT_COUNT:, :],
        first_update,
    )
    assert loop.model.generated == 0
    assert loop.model.decoded == 0
    assert loop.model.embedding_lookups == 0
