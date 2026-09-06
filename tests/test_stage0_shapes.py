"""Fake-only Stage 0 shape contracts for the Qwen 896-wide workspace."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

from codecs_module.decoder import SlotDecoder
from codecs_module.encoder import SlotEncoder
from codecs_module.narration import NarrationDecoder
from workspace.concept_slots import Workspace

SLOT_WIDTH = 896
ABLATION_SLOT_COUNTS = (1, 4, 8, 16, 32, 64)


class _FakeTokenizer:
    eos_token_id = None

    def __call__(self, texts: list[str], **_: Any) -> dict[str, torch.Tensor]:
        token_count = max(1, len(texts[0].split()))
        return {
            "input_ids": torch.arange(token_count).unsqueeze(0),
            "attention_mask": torch.ones(1, token_count, dtype=torch.long),
        }

    def decode(self, token_ids: list[int], **_: Any) -> str:
        return " ".join(f"token-{token_id}" for token_id in token_ids)


class _FakeSentenceTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = SimpleNamespace(tokenizer=_FakeTokenizer(), auto_model=_FakeSentenceModel())

    def __getitem__(self, index: int) -> Any:
        assert index == 0
        return self.transformer


class _FakeSentenceModel:
    def __call__(self, *, input_ids: torch.Tensor, **_: Any) -> Any:
        token_count = input_ids.shape[1]
        values = torch.arange(token_count * 384, dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=values.reshape(1, token_count, 384))


class _FakeInputEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, SLOT_WIDTH)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_embeddings = _FakeInputEmbeddings()
        self.forwarded_embeddings: list[torch.Tensor] = []

    def get_input_embeddings(self) -> nn.Module:
        return self.input_embeddings

    def forward(self, *, inputs_embeds: torch.Tensor, **_: Any) -> Any:
        self.forwarded_embeddings.append(inputs_embeds.detach().clone())
        logits = torch.zeros(inputs_embeds.shape[0], inputs_embeds.shape[1], 8)
        token_id = 2 if inputs_embeds[0, 0, 0] < 0 else 3
        logits[:, -1, token_id] = 1.0
        return SimpleNamespace(logits=logits)


class _CachingFakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_embeddings = _FakeInputEmbeddings()
        self.calls: list[dict[str, Any]] = []

    def get_input_embeddings(self) -> nn.Module:
        return self.input_embeddings

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> Any:
        self.calls.append(
            {
                "input_shape": tuple(inputs_embeds.shape),
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )
        next_token_ids = (3, 4, 5)
        next_token_id = next_token_ids[len(self.calls) - 1]
        logits = torch.zeros(inputs_embeds.shape[0], inputs_embeds.shape[1], 8)
        logits[:, -1, next_token_id] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=f"cache-{len(self.calls)}")


class _WorkspaceState:
    def __init__(self, slots: torch.Tensor) -> None:
        self.slots = slots

    def read_slots(self) -> torch.Tensor:
        return self.slots


@pytest.fixture
def fake_sentence_transformer() -> Any:
    with patch(
        "codecs_module.encoder.SentenceTransformer",
        side_effect=lambda *_args, **_kwargs: _FakeSentenceTransformer(),
    ):
        yield


# covers: workspace/concept-slots::Workspace holds a configurable number of concept slots::default slot count
def test_workspace_default_slots_are_16_by_896() -> None:
    workspace = Workspace()

    assert workspace.read_slots().shape == (16, 896)


# covers: workspace/concept-slots::Workspace holds a configurable number of concept slots::configurable slot count
def test_workspace_configured_slots_are_8_by_896() -> None:
    workspace = Workspace(slot_count=8)

    assert workspace.read_slots().shape == (8, 896)


@pytest.mark.parametrize("slot_count", ABLATION_SLOT_COUNTS)
# covers: workspace/concept-slots::Workspace holds a configurable number of concept slots::ablation sweep values accepted
def test_workspace_accepts_each_unchanged_ablation_slot_count(slot_count: int) -> None:
    workspace = Workspace(slot_count=slot_count)

    assert workspace.slot_count == slot_count
    assert workspace.read_slots().shape[0] == slot_count


# covers: codecs/encoder::Encoder maps text to workspace slots::text encoded to slots
def test_encoder_maps_text_to_896_wide_workspace_slots(fake_sentence_transformer: Any) -> None:
    encoder = SlotEncoder(
        slot_count=8,
        device="cpu",
        manifest_verifier=lambda *_: None,
    )

    slots = encoder.encode("a small text input")

    assert slots.shape == (8, 896)


# covers: codecs/encoder::Encoder maps text to workspace slots::variable-length input accepted
def test_encoder_maps_short_and_long_text_to_same_896_wide_slot_shape(
    fake_sentence_transformer: Any,
) -> None:
    encoder = SlotEncoder(
        slot_count=8,
        device="cpu",
        manifest_verifier=lambda *_: None,
    )

    short_slots = encoder.encode("word")
    long_slots = encoder.encode("this is a paragraph with several distinct words")

    assert short_slots.shape == (8, 896)
    assert long_slots.shape == (8, 896)


# covers: codecs/decoder::Decoder maps workspace slots to text::slots decoded to text
def test_decoder_forwards_896_wide_slots_and_preserves_workspace_state() -> None:
    model = _FakeQwen()
    workspace = _WorkspaceState(torch.ones(8, 896))
    slots_before = workspace.read_slots().clone()
    decoder = SlotDecoder(model=model, tokenizer=_FakeTokenizer(), max_tokens=1)

    text = decoder.decode(workspace)  # type: ignore[arg-type]

    assert text == "token-3"
    assert model.forwarded_embeddings[0].shape == (1, 8, 896)
    assert torch.equal(model.forwarded_embeddings[0], slots_before.unsqueeze(0))
    assert torch.equal(workspace.read_slots(), slots_before)


def test_decoder_uses_kv_cache_after_forwarding_the_slot_prefix_once() -> None:
    model = _CachingFakeQwen()
    tokenizer = _FakeTokenizer()
    tokenizer.eos_token_id = 5
    workspace = _WorkspaceState(torch.ones(8, 896))
    decoder = SlotDecoder(model=model, tokenizer=tokenizer, max_tokens=5)

    text = decoder.decode(workspace)  # type: ignore[arg-type]

    assert text == "token-3 token-4"
    assert [call["input_shape"] for call in model.calls] == [
        (1, 8, 896),
        (1, 1, 896),
        (1, 1, 896),
    ]
    assert [call["past_key_values"] for call in model.calls] == [
        None,
        "cache-1",
        "cache-2",
    ]
    assert all(call["use_cache"] is True for call in model.calls)


def test_narration_forwards_896_wide_slots_and_preserves_workspace_state() -> None:
    model = _FakeQwen()
    workspace = _WorkspaceState(torch.ones(8, 896))
    slots_before = workspace.read_slots().clone()
    decoder = NarrationDecoder(model=model, tokenizer=_FakeTokenizer(), max_tokens=1, device="cpu")

    text = decoder.narrate(workspace)  # type: ignore[arg-type]

    assert text == "token-3"
    assert model.forwarded_embeddings[0].shape == (1, 8, 896)
    assert torch.equal(model.forwarded_embeddings[0], slots_before.unsqueeze(0))
    assert torch.equal(workspace.read_slots(), slots_before)


# covers: codecs/decoder::Decoder maps workspace slots to text::different slot states produce different text
def test_decoder_exposes_distinct_greedy_tokens_for_distinct_slot_states() -> None:
    model = _FakeQwen()
    decoder = SlotDecoder(model=model, tokenizer=_FakeTokenizer(), max_tokens=1)
    negative_state = _WorkspaceState(-torch.ones(8, 896))
    positive_state = _WorkspaceState(torch.ones(8, 896))

    negative_text = decoder.decode(negative_state)  # type: ignore[arg-type]
    positive_text = decoder.decode(positive_state)  # type: ignore[arg-type]

    assert negative_text == "token-2"
    assert positive_text == "token-3"
    assert torch.equal(model.forwarded_embeddings[0], negative_state.read_slots().unsqueeze(0))
    assert torch.equal(model.forwarded_embeddings[1], positive_state.read_slots().unsqueeze(0))


# covers: codecs/decoder::Decoder maps workspace slots to text::incompatible slot width
def test_decoder_rejects_slot_width_before_forwarding_to_qwen() -> None:
    model = _FakeQwen()
    decoder = SlotDecoder(model=model, tokenizer=_FakeTokenizer(), max_tokens=1)
    workspace = _WorkspaceState(torch.zeros(8, 768))

    with pytest.raises(ValueError, match=r"expected.*896.*actual.*768"):
        decoder.decode(workspace)  # type: ignore[arg-type]

    assert model.forwarded_embeddings == []
