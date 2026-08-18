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

from transformers import AutoTokenizer

from codecs_module.decoder import SlotDecoder
from codecs_module.encoder import SlotEncoder
from codecs_module.narration import NarrationDecoder
from core.latent_loop import LatentLoop
from eval.stream.events import Event, Probe
from eval.stream.render import render_event
from eval.subject import CostCounters, Subject
from workspace.concept_slots import DEFAULT_SLOT_COUNT, Workspace

DEFAULT_NUM_STEPS = 8
TOKENIZER_NAME = "EleutherAI/pythia-160m"


class LatentCoreSubject(Subject):
    """A Stage 0 subject built from the workspace, latent loop, and codecs."""

    def __init__(
        self,
        slot_count: int = DEFAULT_SLOT_COUNT,
        num_steps: int = DEFAULT_NUM_STEPS,
        device: str = "cpu",
        narration: bool = False,
    ) -> None:
        self.workspace = Workspace(slot_count=slot_count)
        self.latent_loop = LatentLoop(num_steps=num_steps, device=device)
        self.encoder = SlotEncoder(slot_count=slot_count, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
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

    def observe(self, event: Event) -> object | None:
        text = render_event(event)
        self.encoder.encode_to_workspace(text, self.workspace)
        self.latent_loop.run(self.workspace)
        if self.narration_decoder is not None:
            return self.narration_decoder.narrate(self.workspace)
        return None

    def answer(self, probe: Probe) -> str:
        del probe
        self.latent_loop.run(self.workspace)
        return self.decoder.decode(self.workspace)

    def idle(self, budget: int) -> None:
        del budget

    def snapshot(self) -> object:
        return self.workspace.snapshot()

    def restore(self, state: object) -> None:
        self.workspace.restore(state)  # type: ignore[arg-type]

    def cost(self) -> CostCounters:
        return self.latent_loop.get_cost()
