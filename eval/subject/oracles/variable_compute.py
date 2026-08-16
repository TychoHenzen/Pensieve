"""Positive control oracle: spends compute proportional to chain length.

`VariableComputeOracle` parses the arithmetic-chain queries rendered by
`eval.stream.generators.difficulty_mix` and actually executes each add or
subtract step, incrementing its step counter once per step. It never sees
`ProbeTruth.difficulty` (difficulty lives only on the truth side channel),
so a correlation between its reported compute and the difficulty label can
only arise from genuinely doing more work on longer chains.
"""

from __future__ import annotations

import copy
import re

from eval.stream.events import Event, Probe
from eval.subject import CostCounters, Subject

_START_RE = re.compile(r"Start at (-?\d+)\.")
_STEP_RE = re.compile(r"(Add|Subtract) (\d+)\.")
_MODULUS_RE = re.compile(r"modulo (\d+)\?")


class VariableComputeOracle(Subject):

    def __init__(self) -> None:
        self._steps: int = 0

    def observe(self, event: Event) -> object | None:
        return None

    def answer(self, probe: Probe) -> str:
        query = probe.query
        start_match = _START_RE.search(query)
        modulus_match = _MODULUS_RE.search(query)
        if start_match is None or modulus_match is None:
            return ""

        modulus = int(modulus_match.group(1))
        total = int(start_match.group(1)) % modulus
        for op, operand_str in _STEP_RE.findall(query):
            self._steps += 1
            operand = int(operand_str)
            total = (total + operand) % modulus if op == "Add" else (total - operand) % modulus
        return str(total)

    def idle(self, budget: int) -> None:
        pass

    def snapshot(self) -> object:
        return copy.copy(self._steps)

    def restore(self, state: object) -> None:
        self._steps = state  # type: ignore

    def cost(self) -> CostCounters:
        return CostCounters(steps=self._steps)
