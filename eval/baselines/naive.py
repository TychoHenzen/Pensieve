"""Naive sequential baseline: plain Adam SGD, no forgetting mitigation.

Trains on each `Observe` example as it arrives in stream order, with no
replay, no regularization, and no architectural protection against
catastrophic forgetting. It is the forgetting floor every mitigation
strategy is measured against. van de Ven & Tolias (2019) report 19.90%
accuracy for this baseline on class-incremental Split-MNIST, near chance
for ten classes, because later tasks overwrite what earlier tasks taught.

`Observe` payloads outside `split-classify`'s `{features, label, source}`
shape (facts, prose) are not classification examples, so `observe` skips
them. A `Boundary` carries no example either.
"""

from __future__ import annotations

import copy
import re
from typing import Mapping

import torch
from torch import nn

from eval.baselines.model import MLP
from eval.stream.events import Event, Observe, Probe
from eval.subject import CostCounters, Subject

_FEATURE_PATTERN = re.compile(r"-?\d+\.\d+")


def _parse_features(query: str) -> list[float]:
    """Parse a `features=(1.23, 4.56, ...)` query back into floats.

    `eval.stream.render.format_features` is the one place that writes
    this shape, for both a taught example and a probe. Parsing it back
    here keeps the baseline reading the exact text a subject sees,
    rather than reaching past the protocol into generator internals.
    """
    return [float(match) for match in _FEATURE_PATTERN.findall(query)]


class NaiveBaseline(Subject):
    """Sequential fine-tuning with Adam, no continual-learning mitigation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 400,
        lr: float = 0.001,
    ) -> None:
        self._model = MLP(input_dim, output_dim, hidden_layers, hidden_units)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._loss_fn = nn.CrossEntropyLoss()
        self._output_dim = output_dim
        self._label_to_idx: dict[str, int] = {}
        self._idx_to_label: list[str] = []
        self._steps = 0

    def _label_index(self, label: str) -> int | None:
        """Return `label`'s class index, assigning a new one if there is room.

        A label the model has never seen, arriving after every index up
        to `output_dim` is already assigned, has nowhere to go. That
        should not happen for a correctly sized baseline, but returns
        `None` rather than raising, so a misconfigured `output_dim`
        fails a probe instead of crashing mid-stream.
        """
        idx = self._label_to_idx.get(label)
        if idx is not None:
            return idx
        if len(self._idx_to_label) >= self._output_dim:
            return None
        idx = len(self._idx_to_label)
        self._label_to_idx[label] = idx
        self._idx_to_label.append(label)
        return idx

    def observe(self, event: Event) -> object | None:
        if isinstance(event, Observe) and isinstance(event.payload, Mapping):
            payload = event.payload
            if "features" in payload and "label" in payload:
                target_idx = self._label_index(str(payload["label"]))
                if target_idx is not None:
                    features = torch.tensor(
                        [float(v) for v in payload["features"]], dtype=torch.float32
                    ).unsqueeze(0)
                    target = torch.tensor([target_idx], dtype=torch.long)

                    self._model.train()
                    self._optimizer.zero_grad()
                    logits = self._model(features)
                    loss = self._loss_fn(logits, target)
                    loss.backward()
                    self._optimizer.step()
                    self._steps += 1
        return None

    def answer(self, probe: Probe) -> str:
        features = torch.tensor(_parse_features(probe.query), dtype=torch.float32).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(features)
        idx = int(torch.argmax(logits, dim=1).item())
        self._steps += 1
        if idx < len(self._idx_to_label):
            return self._idx_to_label[idx]
        return ""

    def idle(self, budget: int) -> None:
        pass

    def snapshot(self) -> object:
        return (
            copy.deepcopy(self._model.state_dict()),
            copy.deepcopy(self._optimizer.state_dict()),
            copy.copy(self._label_to_idx),
            copy.copy(self._idx_to_label),
            self._steps,
        )

    def restore(self, state: object) -> None:
        model_state, optimizer_state, label_to_idx, idx_to_label, steps = state  # type: ignore
        self._model.load_state_dict(model_state)
        self._optimizer.load_state_dict(optimizer_state)
        self._label_to_idx = dict(label_to_idx)
        self._idx_to_label = list(idx_to_label)
        self._steps = steps

    def cost(self) -> CostCounters:
        return CostCounters(steps=self._steps)
