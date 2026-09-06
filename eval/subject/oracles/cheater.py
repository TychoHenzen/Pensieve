from __future__ import annotations

import copy

from eval.stream.events import Event, Probe
from eval.subject import CostCounters, Subject


class CheaterOracle(Subject):
    def __init__(self, truth_map: dict[str, str]) -> None:
        self._truth_map = truth_map
        self._learned: dict[str, str] = {}
        self._steps: int = 0

    def observe(self, event: Event) -> object | None:
        return None

    def answer(self, probe: Probe) -> str:
        self._steps += 1
        result = self._learned.get(probe.query, "")
        if probe.query in self._truth_map:
            self._learned[probe.query] = self._truth_map[probe.query]
        return result

    def idle(self, budget: int) -> None:
        pass

    def snapshot(self) -> object:
        return (copy.copy(self._learned), self._steps)

    def restore(self, state: object) -> None:
        learned, steps = state  # type: ignore
        self._learned = dict(learned)
        self._steps = steps

    def cost(self) -> CostCounters:
        return CostCounters(steps=self._steps)
