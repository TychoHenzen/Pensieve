"""Frozen baseline: no learning at all, the no-adaptation floor.

`FrozenBaseline` never updates its weights. It stays at random
initialization for the whole run, so its accuracy sits at or near
chance on every task (`eval.stream.generators.split_classify.chance_rate`
names that floor). It still has to answer with a real label string, so
it tracks the label vocabulary as labels arrive through `observe`. That
tracking is bookkeeping, not learning: no gradient ever touches the
model.

The label vocabulary is capped at `output_dim`, one label per output
unit, matching the shared `MLP` architecture in `eval.baselines.model`.
A label seen past that cap reuses the last free slot rather than
raising, so an oversized stream degrades gracefully instead of failing
the run.
"""

from __future__ import annotations

import copy
import re

import torch

from eval.baselines.model import MLP
from eval.stream.events import Event, Observe, Probe
from eval.subject import CostCounters, Subject

_FEATURES_PATTERN = re.compile(r"features=\(([^)]*)\)")


def _parse_features(query: str) -> list[float]:
    """Parse the feature vector out of a probe query.

    Probe queries are written by `eval.stream.render.format_features`,
    always as `features=(v1, v2, ...)`. Raises `ValueError` when the
    query does not match that shape, naming the query so the failure is
    diagnosable rather than a bare parse error.
    """
    match = _FEATURES_PATTERN.search(query)
    if match is None:
        raise ValueError(f"probe query carries no features=(...) vector: {query!r}")
    inner = match.group(1).strip()
    if not inner:
        return []
    return [float(part) for part in inner.split(",")]


def _to_input_tensor(features: list[float], input_dim: int) -> torch.Tensor:
    """Fit a parsed feature vector to the model's fixed input width.

    A generator's feature vector width need not equal `input_dim` (a
    `split-classify` vector has one value per class, not per pixel). The
    frozen model never trains on shape, so it pads short vectors with
    zeros and truncates long ones rather than refusing to answer.
    """
    values = list(features[:input_dim])
    values.extend([0.0] * (input_dim - len(values)))
    return torch.tensor(values, dtype=torch.float32)


class FrozenBaseline(Subject):
    """A `Subject` that never learns: weights stay at random init throughout.

    Implements the full `Subject` protocol so the harness can run it
    like any other subject, but `observe` and `idle` never touch the
    model's parameters. Only the label vocabulary accumulates, because
    the model needs label strings to answer with even though it never
    learns which features go with which label.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 400,
        device: str = "cpu",
    ) -> None:
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._device = torch.device(device)
        self._model = MLP(input_dim, output_dim, hidden_layers, hidden_units).to(self._device)
        self._model.eval()
        self._label_by_index: dict[int, str] = {}
        self._index_by_label: dict[str, int] = {}
        self._steps = 0

    def observe(self, event: Event) -> object | None:
        self._steps += 1
        if isinstance(event, Observe) and isinstance(event.payload, dict):
            label = event.payload.get("label")
            if label is not None:
                self._register_label(str(label))
        return None

    def answer(self, probe: Probe) -> str:
        self._steps += 1
        if not self._label_by_index:
            return ""
        features = _parse_features(probe.query)
        x = _to_input_tensor(features, self._input_dim).to(self._device)
        with torch.no_grad():
            logits = self._model(x.unsqueeze(0))
        index = int(torch.argmax(logits, dim=-1).item())
        return self._label_by_index.get(index, next(iter(self._label_by_index.values())))

    def idle(self, budget: int) -> None:
        pass

    def snapshot(self) -> object:
        return {
            "model_state": copy.deepcopy(self._model.state_dict()),
            "label_by_index": dict(self._label_by_index),
            "index_by_label": dict(self._index_by_label),
            "steps": self._steps,
        }

    def restore(self, state: object) -> None:
        data = state  # type: ignore[assignment]
        self._model.load_state_dict(data["model_state"])
        self._label_by_index = dict(data["label_by_index"])
        self._index_by_label = dict(data["index_by_label"])
        self._steps = data["steps"]

    def cost(self) -> CostCounters:
        return CostCounters(steps=self._steps)

    def _register_label(self, label: str) -> None:
        if label in self._index_by_label:
            return
        index = len(self._index_by_label)
        if index >= self._output_dim:
            index = self._output_dim - 1
        self._index_by_label[label] = index
        self._label_by_index[index] = label
