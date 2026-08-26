from __future__ import annotations

import torch
from transformers import AutoTokenizer

from codecs_module.narration import NarrationDecoder
from core.latent_loop import LatentLoop
from eval.stream.events import Observe
from eval.subjects.latent_core import LatentCoreSubject
from workspace.concept_slots import Workspace
from codecs_module.decoder import SlotDecoder

TOKENIZER_NAME = "EleutherAI/pythia-160m"
SLOT_COUNT = 4


def test_slot_decoder_defaults_to_model_device() -> None:
    latent_loop = LatentLoop(num_steps=1, device="cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    decoder = SlotDecoder(model=latent_loop.model, tokenizer=tokenizer)

    assert decoder.device == next(latent_loop.model.parameters()).device


def _make_decoder() -> tuple[NarrationDecoder, LatentLoop]:
    latent_loop = LatentLoop(num_steps=1, device="cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    decoder = NarrationDecoder(
        model=latent_loop.model, tokenizer=tokenizer, device="cpu", max_tokens=8
    )
    return decoder, latent_loop


def test_narration_decoder_produces_non_empty_text() -> None:
    decoder, _ = _make_decoder()
    workspace = Workspace(slot_count=SLOT_COUNT)
    workspace.write_slots(torch.randn(SLOT_COUNT, workspace.slots.shape[1]))

    text = decoder.narrate(workspace)

    assert isinstance(text, str)
    assert len(text) > 0


def test_narration_decoder_differs_for_different_slot_states() -> None:
    decoder, _ = _make_decoder()

    generator_a = torch.Generator().manual_seed(0)
    generator_b = torch.Generator().manual_seed(1)

    workspace_a = Workspace(slot_count=SLOT_COUNT)
    workspace_a.write_slots(
        torch.randn(SLOT_COUNT, workspace_a.slots.shape[1], generator=generator_a)
    )

    workspace_b = Workspace(slot_count=SLOT_COUNT)
    workspace_b.write_slots(
        torch.randn(SLOT_COUNT, workspace_b.slots.shape[1], generator=generator_b)
    )

    text_a = decoder.narrate(workspace_a)
    text_b = decoder.narrate(workspace_b)

    assert text_a != text_b


def test_narration_decoder_does_not_modify_workspace_slots() -> None:
    decoder, _ = _make_decoder()
    workspace = Workspace(slot_count=SLOT_COUNT)
    slots_before = torch.randn(SLOT_COUNT, workspace.slots.shape[1])
    workspace.write_slots(slots_before.clone())

    decoder.narrate(workspace)

    assert torch.equal(workspace.read_slots(), slots_before)


def test_latent_core_subject_observe_returns_text_when_narration_enabled() -> None:
    subject = LatentCoreSubject(slot_count=SLOT_COUNT, num_steps=1, narration=True)
    event = Observe(position=0, payload={"text": "the quick brown fox"})

    result = subject.observe(event)

    assert isinstance(result, str)


def test_latent_core_subject_observe_returns_none_when_narration_disabled() -> None:
    subject = LatentCoreSubject(slot_count=SLOT_COUNT, num_steps=1, narration=False)
    event = Observe(position=0, payload={"text": "the quick brown fox"})

    result = subject.observe(event)

    assert result is None
