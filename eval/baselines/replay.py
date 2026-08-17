"""Generative replay baseline: a VAE stands in for storing raw past examples.

`ReplayBaseline` follows van de Ven & Tolias (2019) / van de Ven et al.
(2020) "brain-inspired replay": alongside the `eval.baselines.model.MLP`
solver, it trains a variational autoencoder (the "generator") with the
same hidden-layer sizes as the solver. At each `Boundary(TASK_TRAINED)`,
it retrains both networks on a mix of real examples from the
just-finished task and pseudo-examples sampled from the generator and
labelled by the solver's own predictions. Because the generator can
resynthesize approximate examples of every task seen so far, the solver
keeps rehearsing old tasks without the harness ever handing it a stored
old example, so accuracy on old tasks stays substantially above chance
instead of collapsing the way `eval.baselines.naive.NaiveBaseline` does.
"""

from __future__ import annotations

import copy
import random
import re
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

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


def _hidden_stack(in_dim: int, hidden_layers: int, hidden_units: int) -> tuple[nn.Sequential, int]:
    """Build a ReLU MLP trunk, mirroring `eval.baselines.model.MLP`'s shape.

    Returns the trunk and the width of its final layer, so a caller can
    attach whatever head (class logits, mu/logvar, reconstruction) it
    needs on top without duplicating the layer-building loop.
    """
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(d, hidden_units))
        layers.append(nn.ReLU())
        d = hidden_units
    return nn.Sequential(*layers), d


class VAE(nn.Module):
    """Symmetric variational autoencoder used as the replay generator.

    Encoder: input -> hidden trunk -> (mu, logvar). Decoder: latent `z`
    -> hidden trunk -> reconstructed input. Hidden-layer sizes match the
    solver's `MLP` by default (2 layers of 400 units), so the generator
    has comparable capacity to the classifier it is rehearsing.

    When `pixel_mode` is True the decoder applies sigmoid so its output
    stays in [0, 1], matching MNIST pixel intensities. The corresponding
    loss uses BCE instead of MSE.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 400,
        latent_dim: int = 100,
        pixel_mode: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.pixel_mode = pixel_mode

        encoder_trunk, enc_out = _hidden_stack(input_dim, hidden_layers, hidden_units)
        self.encoder_trunk = encoder_trunk
        self.fc_mu = nn.Linear(enc_out, latent_dim)
        self.fc_logvar = nn.Linear(enc_out, latent_dim)

        decoder_trunk, dec_out = _hidden_stack(latent_dim, hidden_layers, hidden_units)
        self.decoder_trunk = decoder_trunk
        self.fc_out = nn.Linear(dec_out, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_trunk(z)
        out = self.fc_out(h)
        if self.pixel_mode:
            return torch.sigmoid(out)
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def _vae_loss(
    recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
    pixel_mode: bool = False,
) -> torch.Tensor:
    """Reconstruction plus KL divergence to a standard normal prior.

    In pixel mode (features in [0, 1]), uses BCE on the sigmoid-clamped
    decoder output, matching van de Ven & Tolias 2019. Otherwise uses
    MSE for arbitrary-range synthetic features. Both terms are averaged
    over the batch so their relative weight does not drift with
    `batch_size`.
    """
    if pixel_mode:
        recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
    else:
        recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kld


class ReplayBaseline(Subject):
    """Generative-replay continual learner for `split-classify` streams."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 400,
        lr: float = 0.001,
        latent_dim: int = 100,
        train_iterations: int = 2000,
        batch_size: int = 128,
        replay_ratio: float = 1.0,
        device: str = "cpu",
        pixel_mode: bool = False,
    ) -> None:
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_layers = hidden_layers
        self._hidden_units = hidden_units
        self._latent_dim = latent_dim
        self._train_iterations = train_iterations
        self._batch_size = batch_size
        self._replay_ratio = replay_ratio
        self._device = torch.device(device)
        self._pixel_mode = pixel_mode

        self._solver = MLP(input_dim, output_dim, hidden_layers, hidden_units).to(self._device)
        self._solver_optimizer = torch.optim.Adam(self._solver.parameters(), lr=lr)
        self._loss_fn = nn.CrossEntropyLoss()
        self._solver.eval()

        self._generator = VAE(input_dim, hidden_layers, hidden_units, latent_dim, pixel_mode).to(self._device)
        self._generator_optimizer = torch.optim.Adam(self._generator.parameters(), lr=lr)
        self._generator.eval()

        self._buffer: list[tuple[list[float], str]] = []
        self._label_to_index: dict[str, int] = {}
        self._has_trained_generator = False
        self._steps = 0
        self._rng = random.Random(0)

    def observe(self, event: Event) -> object | None:
        if isinstance(event, Observe) and isinstance(event.payload, Mapping):
            features = event.payload.get("features")
            label = event.payload.get("label")
            if features is not None and label is not None:
                label = str(label)
                self._buffer.append((list(features), label))
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
        self._solver.eval()
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32).to(self._device)
            logits = self._solver(x)
            index = int(torch.argmax(logits, dim=1).item())
        index_to_label = {i: lbl for lbl, i in self._label_to_index.items()}
        return index_to_label.get(index, "")

    def idle(self, budget: int) -> None:
        pass

    def snapshot(self) -> object:
        return {
            "solver_state": copy.deepcopy(self._solver.state_dict()),
            "solver_optimizer_state": copy.deepcopy(self._solver_optimizer.state_dict()),
            "generator_state": copy.deepcopy(self._generator.state_dict()),
            "generator_optimizer_state": copy.deepcopy(self._generator_optimizer.state_dict()),
            "buffer": copy.deepcopy(self._buffer),
            "label_to_index": copy.deepcopy(self._label_to_index),
            "has_trained_generator": self._has_trained_generator,
            "steps": self._steps,
        }

    def restore(self, state: object) -> None:
        data = state  # type: ignore[assignment]
        self._solver.load_state_dict(data["solver_state"])  # type: ignore[index]
        self._solver_optimizer.load_state_dict(data["solver_optimizer_state"])  # type: ignore[index]
        self._generator.load_state_dict(data["generator_state"])  # type: ignore[index]
        self._generator_optimizer.load_state_dict(data["generator_optimizer_state"])  # type: ignore[index]
        self._buffer = copy.deepcopy(data["buffer"])  # type: ignore[index]
        self._label_to_index = copy.deepcopy(data["label_to_index"])  # type: ignore[index]
        self._has_trained_generator = data["has_trained_generator"]  # type: ignore[index]
        self._steps = data["steps"]  # type: ignore[index]

    def cost(self) -> CostCounters:
        return CostCounters(steps=self._steps)

    def _retrain(self) -> None:
        """Retrain the solver and generator on the finished task plus replay.

        Replay samples come from frozen copies of the previous task's
        generator and solver. Using the models being trained for replay
        causes a feedback loop: as the solver drifts toward the new
        task, it relabels replay as the new classes, and the generator
        follows, collapsing replay into the latest task only.
        """
        if not self._buffer:
            return

        prev_generator = copy.deepcopy(self._generator) if self._has_trained_generator else None
        prev_solver = copy.deepcopy(self._solver) if self._has_trained_generator else None
        if prev_generator is not None:
            prev_generator.eval()
        if prev_solver is not None:
            prev_solver.eval()

        features = [f for f, _ in self._buffer]
        labels = [self._label_to_index[label] for _, label in self._buffer]
        x_real_all = torch.tensor(features, dtype=torch.float32).to(self._device)
        y_real_all = torch.tensor(labels, dtype=torch.long).to(self._device)
        n_real = len(self._buffer)
        real_batch_size = min(self._batch_size, n_real)

        self._solver.train()
        self._generator.train()

        for _ in range(self._train_iterations):
            real_idx = [self._rng.randrange(n_real) for _ in range(real_batch_size)]
            x_real = x_real_all[real_idx]
            y_real = y_real_all[real_idx]

            if prev_generator is not None and prev_solver is not None:
                replay_size = max(1, round(real_batch_size * self._replay_ratio))
                with torch.no_grad():
                    z = torch.randn(replay_size, self._latent_dim).to(self._device)
                    x_replay = prev_generator.decode(z)
                    replay_logits = prev_solver(x_replay)
                    y_replay = torch.argmax(replay_logits, dim=1)
                x_train = torch.cat([x_real, x_replay], dim=0)
                y_train = torch.cat([y_real, y_replay], dim=0)
            else:
                x_train = x_real
                y_train = y_real

            self._solver_optimizer.zero_grad()
            solver_logits = self._solver(x_train)
            solver_loss = self._loss_fn(solver_logits, y_train)
            solver_loss.backward()
            self._solver_optimizer.step()

            self._generator_optimizer.zero_grad()
            recon, mu, logvar = self._generator(x_train.detach())
            generator_loss = _vae_loss(recon, x_train.detach(), mu, logvar, self._pixel_mode)
            generator_loss.backward()
            self._generator_optimizer.step()

            self._steps += 1

        self._solver.eval()
        self._generator.eval()
        self._has_trained_generator = True
        self._buffer = []
