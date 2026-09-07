"""Joint training baseline: the retention ceiling for continual learning.

`JointBaseline` accumulates every taught `split-classify` example into a
buffer and retrains an `eval.baselines.model.MLP` from scratch on the
whole buffer each time a `Boundary(TASK_TRAINED)` arrives. It is not an
online learner: nothing updates the model between task switches. Because
it sees all data from all tasks in every retrain pass, its accuracy stays
high on every task, unlike a baseline that only sees the current task's
examples. That gap is the point: joint training is the ceiling other
baselines are measured against, not a realistic continual learner.
"""

from __future__ import annotations

import copy
import random
import re
from collections.abc import Mapping

import torch
from torch import nn

from eval.baselines.model import MLP
from eval.stream.events import Boundary, BoundaryKind, Event, Observe, Probe
from eval.subject import CostCounters, Subject

_FEATURES_RE = re.compile(r"features=\(([^)]*)\)")


def _parse_features(query: str) -> list[float] | None:
    match = _FEATURES_RE.search(query)
    if match is None:
        return None
    inner = match.group(1).strip()
    if not inner:
        return []
    return [float(part) for part in inner.split(",")]


class JointBaseline(Subject):
    """Retrain-from-scratch-on-everything baseline for `split-classify` streams."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 400,
        lr: float = 0.001,
        train_iterations: int = 2000,
        batch_size: int = 128,
        device: str = "cpu",
    ) -> None:
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_layers = hidden_layers
        self._hidden_units = hidden_units
        self._lr = lr
        self._train_iterations = train_iterations
        self._batch_size = batch_size
        self._device = torch.device(device)

        self._model = MLP(input_dim, output_dim, hidden_layers, hidden_units).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._loss_fn = nn.CrossEntropyLoss()
        self._model.eval()

        self._examples: list[tuple[list[float], str]] = []
        self._label_to_index: dict[str, int] = {}
        self._steps: int = 0
        self._rng = random.Random(0)

    def observe(self, event: Event) -> object | None:
        if isinstance(event, Observe) and isinstance(event.payload, Mapping):
            features = event.payload.get("features")
            label = event.payload.get("label")
            if features is not None and label is not None:
                label = str(label)
                self._examples.append((list(features), label))
                if label not in self._label_to_index and len(self._label_to_index) < self._output_dim:
                    self._label_to_index[label] = len(self._label_to_index)
        elif isinstance(event, Boundary) and event.kind == BoundaryKind.TASK_TRAINED:
            self._retrain()
        return None

    def answer(self, probe: Probe) -> str:
        features = _parse_features(probe.query)
        if not features or not self._label_to_index:
            return ""
        self._steps += 1
        self._model.eval()
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32).to(self._device)
            logits = self._model(x)
            index = int(torch.argmax(logits, dim=1).item())
        index_to_label = {i: lbl for lbl, i in self._label_to_index.items()}
        return index_to_label.get(index, "")

    def idle(self, budget: int) -> None:
        pass

    def snapshot(self) -> object:
        return {
            "model_state": copy.deepcopy(self._model.state_dict()),
            "optimizer_state": copy.deepcopy(self._optimizer.state_dict()),
            "examples": copy.deepcopy(self._examples),
            "label_to_index": copy.deepcopy(self._label_to_index),
            "steps": self._steps,
        }

    def restore(self, state: object) -> None:
        self._model.load_state_dict(state["model_state"])  # type: ignore[index]
        self._optimizer.load_state_dict(state["optimizer_state"])  # type: ignore[index]
        self._examples = copy.deepcopy(state["examples"])  # type: ignore[index]
        self._label_to_index = copy.deepcopy(state["label_to_index"])  # type: ignore[index]
        self._steps = state["steps"]  # type: ignore[index]

    def cost(self) -> CostCounters:
        return CostCounters(steps=self._steps)

    def _retrain(self) -> None:
        if not self._examples:
            return
        self._model.train()
        features = [example[0] for example in self._examples]
        labels = [self._label_to_index[example[1]] for example in self._examples]
        x_all = torch.tensor(features, dtype=torch.float32).to(self._device)
        y_all = torch.tensor(labels, dtype=torch.long).to(self._device)
        n = len(self._examples)
        batch_size = min(self._batch_size, n)

        for _ in range(self._train_iterations):
            batch_indices = [self._rng.randrange(n) for _ in range(batch_size)]
            x_batch = x_all[batch_indices]
            y_batch = y_all[batch_indices]

            self._optimizer.zero_grad()
            logits = self._model(x_batch)
            loss = self._loss_fn(logits, y_batch)
            loss.backward()
            self._optimizer.step()
            self._steps += 1

        self._model.eval()
