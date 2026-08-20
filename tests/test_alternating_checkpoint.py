"""Tests for resumable alternating-training checkpoints."""

from __future__ import annotations

import random

import torch
from torch import nn

from train.alternating_checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointSchedule,
    capture_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_round_trips_resumption_state(tmp_path) -> None:
    model = nn.Linear(2, 1)
    eggroll_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    gradient_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    _step(eggroll_optimizer, model, gradient=1.0)
    _step(gradient_optimizer, model, gradient=2.0)
    schedule = CheckpointSchedule("gradient", 17, 517, 2, 16)
    model_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    random.seed(123)
    torch.manual_seed(456)
    expected_python_random = random.getstate()
    expected_torch_random = torch.get_rng_state()

    checkpoint = capture_checkpoint(
        model_state=model_state,
        eggroll_optimizer=eggroll_optimizer,
        gradient_optimizer=gradient_optimizer,
        schedule=schedule,
        run_config={"phase_steps": 500, "epochs": 5},
        held_out_selection=[{"question": "q", "answer": "1"}],
        metrics={"answer_exact_match": 0.25},
    )
    path = tmp_path / "boundary.pt"
    save_checkpoint(path, checkpoint)
    loaded = load_checkpoint(path)

    assert loaded.version == CHECKPOINT_VERSION
    assert loaded.schedule == schedule
    assert loaded.run_config == {"phase_steps": 500, "epochs": 5}
    assert loaded.held_out_selection == [{"question": "q", "answer": "1"}]
    assert loaded.metrics == {"answer_exact_match": 0.25}
    assert loaded.python_random_state == expected_python_random
    assert torch.equal(loaded.torch_random_state, expected_torch_random)
    assert set(loaded.model_state) == {"weight", "bias"}
    assert all(torch.equal(loaded.model_state[name], value) for name, value in model_state.items())
    assert _same_state(loaded.eggroll_optimizer_state, eggroll_optimizer.state_dict())
    assert _same_state(loaded.gradient_optimizer_state, gradient_optimizer.state_dict())


def _step(optimizer: torch.optim.Optimizer, model: nn.Module, *, gradient: float) -> None:
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, gradient)
    optimizer.step()
    optimizer.zero_grad()


def _same_state(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_state(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_state(*pair) for pair in zip(left, right))
    return left == right
