from __future__ import annotations

import random

from eval.stream.events import Event, Probe
from eval.stream.vocab import VOCAB
from eval.subject import CostCounters, Subject


class ChanceOracle(Subject):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._steps: int = 0

    def observe(self, event: Event) -> object | None:
        return None

    def answer(self, probe: Probe) -> str:
        self._steps += 1
        return self._rng.choice(VOCAB)

    def idle(self, budget: int) -> None:
        pass

    def snapshot(self) -> object:
        return (self._rng.getstate(), self._steps)

    def restore(self, state: object) -> None:
        rng_state, steps = state  # type: ignore
        self._rng.setstate(rng_state)
        self._steps = steps

    def cost(self) -> CostCounters:
        return CostCounters(steps=self._steps)
