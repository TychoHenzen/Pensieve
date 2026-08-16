"""Elastic Weight Consolidation baseline (Kirkpatrick et al. 2017).

Adds a quadratic penalty, weighted by the diagonal Fisher information
matrix, that anchors each parameter to its value at the end of the
previous task. The penalty slows learning on weights that mattered for
earlier tasks, but EWC alone has no task oracle at test time. On
class-incremental Split-MNIST it is expected to fail near chance
(~20%): van de Ven & Tolias (2019), Table 4.
"""

from __future__ import annotations

import copy
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


class EWCBaseline(Subject):
    """Sequential fine-tuning with an EWC penalty at task boundaries."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 400,
        lr: float = 0.001,
        ewc_lambda: float = 1e7,
        fisher_samples: int = 1000,
    ) -> None:
        self._model = MLP(input_dim, output_dim, hidden_layers, hidden_units)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._loss_fn = nn.CrossEntropyLoss()
        self._output_dim = output_dim
        self._ewc_lambda = ewc_lambda
        self._fisher_samples = fisher_samples
        self._label_to_idx: dict[str, int] = {}
        self._idx_to_label: list[str] = []
        self._steps = 0

        # Diagonal Fisher information, accumulated across task boundaries,
        # keyed by parameter name.
        self._fisher: dict[str, torch.Tensor] = {}
        # Parameter checkpoint taken at the end of the previous task.
        self._theta_star: dict[str, torch.Tensor] = {}
        # Examples seen since the last task boundary, used to estimate
        # the Fisher diagonal for that task.
        self._task_examples: list[tuple[torch.Tensor, int]] = []

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
            return torch.tensor(0.0)
        penalty = torch.tensor(0.0)
        for name, param in self._model.named_parameters():
            fisher = self._fisher.get(name)
            theta_star = self._theta_star.get(name)
            if fisher is None or theta_star is None:
                continue
            penalty = penalty + (fisher * (param - theta_star) ** 2).sum()
        return self._ewc_lambda / 2 * penalty

    def _consolidate_task(self) -> None:
        """Estimate the Fisher diagonal from buffered examples, checkpoint
        the current parameters, and accumulate the Fisher into the running
        total, then clear the example buffer for the next task."""
        examples = self._task_examples[: self._fisher_samples]
        if examples:
            new_fisher = {
                name: torch.zeros_like(param)
                for name, param in self._model.named_parameters()
            }
            self._model.train()
            for features, target_idx in examples:
                self._optimizer.zero_grad()
                logits = self._model(features)
                target = torch.tensor([target_idx], dtype=torch.long)
                loss = self._loss_fn(logits, target)
                loss.backward()
                for name, param in self._model.named_parameters():
                    if param.grad is not None:
                        new_fisher[name] += param.grad.detach() ** 2
            for name in new_fisher:
                new_fisher[name] /= len(examples)
                if name in self._fisher:
                    self._fisher[name] = self._fisher[name] + new_fisher[name]
                else:
                    self._fisher[name] = new_fisher[name]

        self._theta_star = {
            name: param.detach().clone()
            for name, param in self._model.named_parameters()
        }
        self._task_examples = []

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
                    loss = self._loss_fn(logits, target) + self._ewc_penalty()
                    loss.backward()
                    self._optimizer.step()
                    self._steps += 1

                    self._task_examples.append((features, target_idx))
        elif isinstance(event, Boundary) and event.kind is BoundaryKind.TASK_SWITCH:
            self._consolidate_task()
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
            copy.deepcopy(self._fisher),
            copy.deepcopy(self._theta_star),
            copy.deepcopy(self._task_examples),
        )

    def restore(self, state: object) -> None:
        (
            model_state,
            optimizer_state,
            label_to_idx,
            idx_to_label,
            steps,
            fisher,
            theta_star,
            task_examples,
        ) = state  # type: ignore
        self._model.load_state_dict(model_state)
        self._optimizer.load_state_dict(optimizer_state)
        self._label_to_idx = dict(label_to_idx)
        self._idx_to_label = list(idx_to_label)
        self._steps = steps
        self._fisher = dict(fisher)
        self._theta_star = dict(theta_star)
        self._task_examples = list(task_examples)

    def cost(self) -> CostCounters:
        return CostCounters(steps=self._steps)
