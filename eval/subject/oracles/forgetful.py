from __future__ import annotations

from eval.stream.events import Event, Observe, Probe
from eval.stream.render import format_features
from eval.subject import CostCounters, Subject


class ForgetfulOracle(Subject):
    def __init__(self) -> None:
        self._last_key: str | None = None
        self._last_value: str = ""
        self._steps: int = 0

    def observe(self, event: Event) -> object | None:
        if isinstance(event, Observe) and isinstance(event.payload, dict):
            key, value = _extract_kv(event.payload)
            if key is not None:
                self._last_key = key
                self._last_value = value
                self._steps += 1
        return None

    def answer(self, probe: Probe) -> str:
        self._steps += 1
        if probe.query == self._last_key:
            return self._last_value
        return ""

    def idle(self, budget: int) -> None:
        pass

    def snapshot(self) -> object:
        return (self._last_key, self._last_value, self._steps)

    def restore(self, state: object) -> None:
        self._last_key, self._last_value, self._steps = state  # type: ignore

    def cost(self) -> CostCounters:
        return CostCounters(steps=self._steps)


def _extract_kv(payload: dict) -> tuple[str | None, str]:
    if "key" in payload and "value" in payload:
        return str(payload["key"]), str(payload["value"])
    if "features" in payload and "label" in payload:
        return format_features(payload["features"]), str(payload["label"])
    return None, ""
