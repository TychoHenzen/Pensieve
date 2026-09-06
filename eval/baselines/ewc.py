"""Elastic Weight Consolidation baseline (Kirkpatrick et al. 2017).

Buffers every taught `split-classify` example and, at each
`Boundary(TASK_TRAINED)`, retrains the shared `eval.baselines.model.MLP`
on the buffer for `train_iterations` steps of batch `batch_size`. The
training loss adds a quadratic penalty, weighted by the diagonal Fisher
information matrix, that anchors each parameter to its value at the end
of the previous task. After training, the Fisher diagonal is
re-estimated from the same buffer and accumulated into the running
total, and the buffer is cleared. Unlike naive fine-tuning, EWC does not
start each task from scratch: the penalty keeps the model's weights near
where they mattered for earlier tasks. It still has no task oracle at
test time, so on class-incremental Split-MNIST it is expected to fail
near chance (~20%): van de Ven & Tolias (2019), Table 4.
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


class EWCBaseline(Subject):
    """Buffer-and-retrain baseline with an EWC penalty at task boundaries."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 400,
        lr: float = 0.001,
        ewc_lambda: float = 1e7,
        fisher_samples: int = 1000,
        train_iterations: int = 2000,
        batch_size: int = 128,
        device: str = "cpu",
    ) -> None:
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_layers = hidden_layers
        self._hidden_units = hidden_units
        self._lr = lr
        self._ewc_lambda = ewc_lambda
        self._fisher_samples = fisher_samples
        self._train_iterations = train_iterations
        self._batch_size = batch_size
        self._device = torch.device(device)

        self._model = MLP(input_dim, output_dim, hidden_layers, hidden_units).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._loss_fn = nn.CrossEntropyLoss()
        self._model.eval()

        self._label_to_idx: dict[str, int] = {}
        self._idx_to_label: list[str] = []
        self._steps = 0
        self._rng = random.Random(0)

        # Examples taught since the last task boundary. Retrained on in
        # full at `Boundary(TASK_TRAINED)`, and doubles as the sample
        # pool for that boundary's Fisher estimate, then cleared.
        self._examples: list[tuple[list[float], str]] = []

        # Diagonal Fisher information, accumulated across task boundaries,
        # keyed by parameter name.
        self._fisher: dict[str, torch.Tensor] = {}
        # Parameter checkpoint taken at the end of the previous task.
        self._theta_star: dict[str, torch.Tensor] = {}

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

    def _ewc_penalty(self) -> torch.Tensor:
        if not self._theta_star:
            return torch.tensor(0.0, device=self._device)
        penalty = torch.tensor(0.0, device=self._device)
        for name, param in self._model.named_parameters():
            fisher = self._fisher.get(name)
            theta_star = self._theta_star.get(name)
            if fisher is None or theta_star is None:
                continue
            penalty = penalty + (fisher * (param - theta_star) ** 2).sum()
        return self._ewc_lambda / 2 * penalty

    def _consolidate_task(self) -> None:
        """Estimate the Fisher diagonal from the buffered task's examples,
        accumulate it into the running total, and checkpoint the current
        parameters. Leaves `self._examples` untouched; the caller clears
        it once both retraining and consolidation are done.
        """
        examples = self._examples[: self._fisher_samples]
        if examples:
            new_fisher = {name: torch.zeros_like(param) for name, param in self._model.named_parameters()}
            self._model.train()
            for features, label in examples:
                target_idx = self._label_to_idx.get(label)
                if target_idx is None:
                    continue
                x = torch.tensor([features], dtype=torch.float32, device=self._device)
                target = torch.tensor([target_idx], dtype=torch.long, device=self._device)
                self._optimizer.zero_grad()
                logits = self._model(x)
                loss = self._loss_fn(logits, target)
                loss.backward()
                for name, param in self._model.named_parameters():
                    if param.grad is not None:
                        new_fisher[name] += param.grad.detach() ** 2
            for name, fisher_value in new_fisher.items():
                normalized = fisher_value / len(examples)
                new_fisher[name] = normalized
                if name in self._fisher:
                    self._fisher[name] = self._fisher[name] + normalized
                else:
                    self._fisher[name] = normalized

        self._theta_star = {name: param.detach().clone() for name, param in self._model.named_parameters()}
        self._model.eval()

    def _retrain(self) -> None:
        if not self._examples:
            return
        self._model.train()
        features = [example[0] for example in self._examples]
        labels = [self._label_to_idx[example[1]] for example in self._examples]
        x_all = torch.tensor(features, dtype=torch.float32, device=self._device)
        y_all = torch.tensor(labels, dtype=torch.long, device=self._device)
        n = len(self._examples)
        batch_size = min(self._batch_size, n)

        for _ in range(self._train_iterations):
            batch_indices = [self._rng.randrange(n) for _ in range(batch_size)]
            x_batch = x_all[batch_indices]
            y_batch = y_all[batch_indices]

            self._optimizer.zero_grad()
            logits = self._model(x_batch)
            loss = self._loss_fn(logits, y_batch) + self._ewc_penalty()
            loss.backward()
            self._optimizer.step()
            self._steps += 1

        self._model.eval()

    def observe(self, event: Event) -> object | None:
        if isinstance(event, Observe) and isinstance(event.payload, Mapping):
            payload = event.payload
            if "features" in payload and "label" in payload:
                label = str(payload["label"])
                target_idx = self._label_index(label)
                if target_idx is not None:
                    self._examples.append(([float(v) for v in payload["features"]], label))
        elif isinstance(event, Boundary) and event.kind is BoundaryKind.TASK_TRAINED:
            self._retrain()
            self._consolidate_task()
            self._examples = []
        return None

    def answer(self, probe: Probe) -> str:
        features = _parse_features(probe.query)
        if not features or not self._idx_to_label:
            return ""
        self._steps += 1
        self._model.eval()
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32, device=self._device)
            logits = self._model(x)
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
            "label_to_idx": copy.copy(self._label_to_idx),
            "idx_to_label": copy.copy(self._idx_to_label),
            "steps": self._steps,
            "fisher": copy.deepcopy(self._fisher),
            "theta_star": copy.deepcopy(self._theta_star),
            "examples": copy.deepcopy(self._examples),
            "rng_state": self._rng.getstate(),
        }

    def restore(self, state: object) -> None:
        self._model.load_state_dict(state["model_state"])  # type: ignore[index]
        self._optimizer.load_state_dict(state["optimizer_state"])  # type: ignore[index]
        self._label_to_idx = dict(state["label_to_idx"])  # type: ignore[index]
        self._idx_to_label = list(state["idx_to_label"])  # type: ignore[index]
        self._steps = state["steps"]  # type: ignore[index]
        self._fisher = dict(state["fisher"])  # type: ignore[index]
        self._theta_star = dict(state["theta_star"])  # type: ignore[index]
        self._examples = copy.deepcopy(state["examples"])  # type: ignore[index]
        self._rng.setstate(state["rng_state"])  # type: ignore[index]

    def cost(self) -> CostCounters:
        return CostCounters(steps=self._steps)
