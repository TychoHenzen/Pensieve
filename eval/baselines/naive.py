"""Naive sequential baseline: retrain-per-task Adam SGD, no forgetting mitigation.

Buffers each task's `Observe` examples and retrains the model from its
current weights when a `Boundary(TASK_TRAINED)` arrives, matching van de
Ven & Tolias (2019)'s methodology of 2000 iterations at batch 128 per
task. The buffer is cleared after each retrain, so the model has no
access to earlier tasks' examples once it moves on: no replay, no
regularization, and no architectural protection against catastrophic
forgetting. It is the forgetting floor every mitigation strategy is
measured against. The paper reports 19.90% accuracy for this baseline on
class-incremental Split-MNIST, near chance for ten classes, because later
tasks overwrite what earlier tasks taught.

`Observe` payloads outside `split-classify`'s `{features, label, source}`
shape (facts, prose) are not classification examples, so `observe` skips
them. A `Boundary` other than `TASK_TRAINED` carries no example either.
"""

from __future__ import annotations

import copy
import random
import re
from typing import Mapping

import torch
from torch import nn

from eval.baselines.model import MLP
from eval.stream.events import Boundary, BoundaryKind, Event, Observe, Probe
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
    """Buffer-and-retrain-per-task baseline with no continual-learning mitigation."""

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
        self._device = torch.device(device)
        self._output_dim = output_dim
        self._train_iterations = train_iterations
        self._batch_size = batch_size

        self._model = MLP(input_dim, output_dim, hidden_layers, hidden_units).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._loss_fn = nn.CrossEntropyLoss()
        self._model.eval()

        self._buffer: list[tuple[list[float], str]] = []
        self._label_to_idx: dict[str, int] = {}
        self._idx_to_label: list[str] = []
        self._steps = 0
        self._rng = random.Random(0)

    def observe(self, event: Event) -> object | None:
        if isinstance(event, Observe) and isinstance(event.payload, Mapping):
            payload = event.payload
            if "features" in payload and "label" in payload:
                label = str(payload["label"])
                self._buffer.append(([float(v) for v in payload["features"]], label))
                if label not in self._label_to_idx and len(self._idx_to_label) < self._output_dim:
                    self._label_to_idx[label] = len(self._idx_to_label)
                    self._idx_to_label.append(label)
        elif isinstance(event, Boundary) and event.kind == BoundaryKind.TASK_TRAINED:
            self._retrain()
            self._buffer.clear()
        return None

    def answer(self, probe: Probe) -> str:
        features = torch.tensor(
            _parse_features(probe.query), dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(features)
        idx = int(torch.argmax(logits, dim=1).item())
        if idx < len(self._idx_to_label):
            return self._idx_to_label[idx]
        return ""

    def idle(self, budget: int) -> None:
        pass

    def snapshot(self) -> object:
        return {
            "model_state": copy.deepcopy(self._model.state_dict()),
            "optimizer_state": copy.deepcopy(self._optimizer.state_dict()),
            "buffer": copy.deepcopy(self._buffer),
            "label_to_idx": copy.copy(self._label_to_idx),
            "idx_to_label": copy.copy(self._idx_to_label),
            "steps": self._steps,
            "rng_state": self._rng.getstate(),
        }

    def restore(self, state: object) -> None:
        state = state  # type: ignore[assignment]
        self._model.load_state_dict(state["model_state"])  # type: ignore[index]
        self._optimizer.load_state_dict(state["optimizer_state"])  # type: ignore[index]
        self._buffer = copy.deepcopy(state["buffer"])  # type: ignore[index]
        self._label_to_idx = dict(state["label_to_idx"])  # type: ignore[index]
        self._idx_to_label = list(state["idx_to_label"])  # type: ignore[index]
        self._steps = state["steps"]  # type: ignore[index]
        self._rng.setstate(state["rng_state"])  # type: ignore[index]

    def cost(self) -> CostCounters:
        return CostCounters(steps=self._steps)

    def _retrain(self) -> None:
        if not self._buffer:
            return
        self._model.train()
        features = [example[0] for example in self._buffer]
        labels = [self._label_to_idx[example[1]] for example in self._buffer]
        x_all = torch.tensor(features, dtype=torch.float32, device=self._device)
        y_all = torch.tensor(labels, dtype=torch.long, device=self._device)
        n = len(self._buffer)
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
