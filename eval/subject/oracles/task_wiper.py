from __future__ import annotations

import copy
import random

from eval.stream.events import Boundary, BoundaryKind, Event, Observe, Probe
from eval.stream.render import format_features
from eval.stream.vocab import VOCAB
from eval.subject import CostCounters, Subject


class TaskWiperOracle(Subject):
    def __init__(self, seed: int = 0) -> None:
        self._facts: dict[str, str] = {}
        self._rng = random.Random(seed)
        self._steps: int = 0

    def observe(self, event: Event) -> object | None:
        if isinstance(event, Boundary) and event.kind == BoundaryKind.TASK_SWITCH:
            self._facts.clear()
            return None
        if isinstance(event, Observe) and isinstance(event.payload, dict):
            key, value = _extract_kv(event.payload)
            if key is not None:
                self._facts[key] = value
                self._steps += 1
        return None

    def answer(self, probe: Probe) -> str:
        self._steps += 1
        if probe.query in self._facts:
            return self._facts[probe.query]
        return self._rng.choice(VOCAB)

    def idle(self, budget: int) -> None:
        pass

    def snapshot(self) -> object:
        return (copy.copy(self._facts), self._rng.getstate(), self._steps)

    def restore(self, state: object) -> None:
        facts, rng_state, steps = state  # type: ignore
        self._facts = dict(facts)
        self._rng.setstate(rng_state)
        self._steps = steps

    def cost(self) -> CostCounters:
        return CostCounters(steps=self._steps)


def _extract_kv(payload: dict) -> tuple[str | None, str]:
    if "key" in payload and "value" in payload:
        return str(payload["key"]), str(payload["value"])
    if "features" in payload and "label" in payload:
        return format_features(payload["features"]), str(payload["label"])
    return None, ""
