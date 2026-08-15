from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from eval.stream.events import Event, Probe


@dataclass(frozen=True)
class CostCounters:
    steps: int = 0
    flops: int = 0
    wall_seconds: float = 0.0


class Subject(ABC):

    @abstractmethod
    def observe(self, event: Event) -> object | None:
        ...

    @abstractmethod
    def answer(self, probe: Probe) -> str:
        ...

    @abstractmethod
    def idle(self, budget: int) -> None:
        ...

    @abstractmethod
    def snapshot(self) -> object:
        ...

    @abstractmethod
    def restore(self, state: object) -> None:
        ...

    @abstractmethod
    def cost(self) -> CostCounters:
        ...
