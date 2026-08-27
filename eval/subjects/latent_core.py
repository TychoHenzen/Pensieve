"""Stage 0 subject: wires the workspace, latent loop, encoder, and decoder.

`observe` encodes the rendered event text into workspace slots, then runs the
latent loop over the workspace. `answer` runs the latent loop over the
current workspace, then decodes the resulting slots into text. `idle` is a
no-op: the subject holds no wall-clock behavior for idle time. `snapshot`
and `restore` delegate to the workspace, the only stateful component.
`cost` reports the latent loop's cumulative step/flop counters, since the
encoder and decoder run frozen pretrained models outside that budget.
"""

from __future__ import annotations

import torch

from codecs_module.decoder import SlotDecoder
from codecs_module.encoder import SlotEncoder
from codecs_module.narration import NarrationDecoder
from core.latent_loop import LatentLoop
from eval.stage0_identity import load_frozen_qwen_backbone
from eval.stream.events import Event, Probe
from eval.stream.render import render_event
from eval.subject import CostCounters, Subject
from workspace.concept_slots import DEFAULT_SLOT_COUNT, Workspace

DEFAULT_NUM_STEPS = 2


class LatentCoreSubject(Subject):
    """A Stage 0 subject built from the workspace, latent loop, and codecs."""

    def __init__(
        self,
        slot_count: int = DEFAULT_SLOT_COUNT,
        num_steps: int = DEFAULT_NUM_STEPS,
        device: str = "cpu",
        narration: bool = False,
        backbone: object | None = None,
    ) -> None:
        self.backbone = backbone or load_frozen_qwen_backbone(device=device)
        self.workspace = Workspace(slot_count=slot_count)
        self.latent_loop = LatentLoop(
            num_steps=num_steps, device=device, backbone=self.backbone
        )
        self.encoder = SlotEncoder(slot_count=slot_count, device=device)
        self.tokenizer = self.backbone.tokenizer
        self._last_context_embeds: torch.Tensor | None = None
        self.decoder = SlotDecoder(
            model=self.latent_loop.model, tokenizer=self.tokenizer, device=device
        )
        self.narration_decoder = (
            NarrationDecoder(
                model=self.latent_loop.model, tokenizer=self.tokenizer, device=device
            )
            if narration
            else None
        )

    def _context_embeds(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        return self.latent_loop.embed_tokens(ids)

    def observe(self, event: Event) -> object | None:
        text = render_event(event)
        self.encoder.encode_to_workspace(text, self.workspace)
        self._last_context_embeds = self._context_embeds(text)
        self.latent_loop.run(self.workspace, context_embeds=self._last_context_embeds)
        if self.narration_decoder is not None:
            return self.narration_decoder.narrate(self.workspace)
        return None

    def answer(self, probe: Probe) -> str:
        del probe
        self.latent_loop.run(self.workspace, context_embeds=self._last_context_embeds)
        return self.decoder.decode(self.workspace)

    def idle(self, budget: int) -> None:
        del budget

    def snapshot(self) -> object:
        return {
            "workspace": self.workspace.snapshot(),
            "last_context_embeds": self._last_context_embeds.clone()
            if self._last_context_embeds is not None
            else None,
        }

    def restore(self, state: object) -> None:
        s = state  # type: ignore[assignment]
        self.workspace.restore(s["workspace"])  # type: ignore[index]
        self._last_context_embeds = s["last_context_embeds"]  # type: ignore[index]

    def cost(self) -> CostCounters:
        return self.latent_loop.get_cost()
