"""Contract tests for the injected, shared frozen Qwen Stage 0 backbone.

Public API assumption for task 2.2:
``eval.stage0_identity.load_frozen_qwen_backbone`` accepts injected model,
tokenizer, config, generation-config, and manifest-verifier callables.  It
returns an object with ``model`` and ``tokenizer`` attributes.  ``LatentLoop``
and ``TrainingState`` accept that object through a ``backbone=`` keyword.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from core.latent_loop import LatentLoop
from core.qwen_tap import QwenTapAdapter
from eval import stage0_identity as stage0
from train.eggroll_trainer import EggrollTrainer
from train.trainer import LatentCoreTrainer
from train.training_state import TrainingState


class _FakeQwen(nn.Module):
    def __init__(self, *, hidden_size: int = 896, hidden_layers: int = 24) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_hidden_layers=hidden_layers,
        )
        self.embedding = nn.Embedding(32, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.embedding_calls: list[torch.Tensor] = []

    def get_input_embeddings(self) -> nn.Module:
        return _RecordingEmbedding(self.embedding, self.embedding_calls)

    def forward(self, *, inputs_embeds: torch.Tensor, **_: Any) -> Any:
        hidden = self.projection(inputs_embeds)
        return SimpleNamespace(hidden_states=tuple(hidden for _ in range(self.config.num_hidden_layers + 1)))


class _RecordingEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, calls: list[torch.Tensor]) -> None:
        super().__init__()
        self.embedding = embedding
        self.calls = calls

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.calls.append(token_ids.detach().clone())
        return self.embedding(token_ids)


def _backbone(model: _FakeQwen | None = None) -> Any:
    return SimpleNamespace(model=model or _FakeQwen(), tokenizer=object())


def _optimizer_step(optimizer: torch.optim.Optimizer, parameters: list[nn.Parameter]) -> None:
    optimizer.zero_grad()
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()


def test_loader_pins_and_verifies_every_qwen_asset_before_loading(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    events: list[str] = []
    calls: dict[str, dict[str, Any]] = {}
    model = _FakeQwen()

    def require_deterministic_environment(name: str) -> None:
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        events.append(name)

    def verify_manifest(*args: Any, **kwargs: Any) -> None:
        require_deterministic_environment("manifest")
        calls["manifest"] = {"args": args, **kwargs}

    def model_loader(name: str, **kwargs: Any) -> Any:
        require_deterministic_environment("model")
        calls["model"] = {"name": name, **kwargs}
        return model

    def config_loader(name: str, **kwargs: Any) -> Any:
        require_deterministic_environment("config")
        calls["config"] = {"name": name, **kwargs}
        return model.config

    def generation_config_loader(name: str, **kwargs: Any) -> object:
        require_deterministic_environment("generation_config")
        calls["generation_config"] = {"name": name, **kwargs}
        return object()

    def tokenizer_loader(name: str, **kwargs: Any) -> object:
        require_deterministic_environment("tokenizer")
        calls["tokenizer"] = {"name": name, **kwargs}
        return object()

    backbone = stage0.load_frozen_qwen_backbone(
        model_loader=model_loader,
        tokenizer_loader=tokenizer_loader,
        config_loader=config_loader,
        generation_config_loader=generation_config_loader,
        manifest_verifier=verify_manifest,
    )

    assert events[0] == "manifest"
    assert calls["manifest"]["args"] == (
        stage0.QWEN_MODEL,
        stage0.QWEN_REVISION,
        stage0.QWEN_MANIFEST,
    )
    for asset in ("model", "tokenizer", "config", "generation_config"):
        assert calls[asset]["name"] == stage0.QWEN_MODEL
        assert calls[asset]["revision"] == stage0.QWEN_REVISION
        assert calls[asset]["trust_remote_code"] is False
    assert calls["model"]["use_safetensors"] is True
    assert calls["model"]["torch_dtype"] is torch.float32
    assert calls["model"]["attn_implementation"] == "eager"
    assert backbone.model is model
    assert backbone.model.training is False
    assert all(not parameter.requires_grad for parameter in backbone.model.parameters())


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (_FakeQwen(hidden_size=768), r"expected.*896.*actual.*768"),
        (_FakeQwen(hidden_layers=11), r"expected.*12.*actual.*11"),
    ],
)
# covers: core/latent-loop::Latent loop feeds hidden state back as input::selected tap layer unavailable
def test_latent_loop_rejects_an_incompatible_qwen_shape(model: _FakeQwen, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        LatentLoop(backbone=_backbone(model), num_steps=1, device="cpu")


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (_FakeQwen(hidden_size=768), r"expected.*896.*actual.*768"),
        (_FakeQwen(hidden_layers=11), r"expected.*12.*actual.*11"),
    ],
)
def test_qwen_tap_adapter_rejects_an_incompatible_model_shape(model: _FakeQwen, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        QwenTapAdapter(model)


# covers: core/latent-loop::Model-neutral token embedding access::Qwen context embedding
def test_latent_loop_uses_qwens_public_input_embedding_layer() -> None:
    model = _FakeQwen()
    loop = LatentLoop(backbone=_backbone(model), num_steps=1, device="cpu")
    token_ids = torch.tensor([[4, 7, 9]])

    embeddings = loop.embed_tokens(token_ids)

    assert len(model.embedding_calls) == 1
    assert torch.equal(model.embedding_calls[0], token_ids)
    assert embeddings.shape == (3, 896)


# covers: train/stage0-training::Shared frozen Qwen backbone::Shared model instance
def test_shared_state_uses_one_qwen_model_for_both_trainers() -> None:
    backbone = _backbone()
    state = TrainingState(
        backbone=backbone,
        sentence_model=nn.Identity(),
        slot_count=4,
        num_steps=1,
        device="cpu",
    )
    gradient = LatentCoreTrainer(state=state)
    eggroll = EggrollTrainer(state=state, pop_size=2)

    assert gradient.latent_loop.model is backbone.model
    assert eggroll.latent_loop.model is backbone.model
    assert gradient.latent_loop.model is eggroll.latent_loop.model


# covers: train/stage0-training::Shared frozen Qwen backbone::Backbone parameters remain frozen
def test_shared_state_keeps_qwen_parameters_frozen_and_out_of_optimizers() -> None:
    backbone = _backbone()
    state = TrainingState(
        backbone=backbone,
        sentence_model=nn.Identity(),
        slot_count=4,
        num_steps=1,
        device="cpu",
    )
    gradient = LatentCoreTrainer(state=state)
    eggroll = EggrollTrainer(state=state, pop_size=2)
    qwen_before = [parameter.detach().clone() for parameter in backbone.model.parameters()]

    _optimizer_step(gradient.optimizer, list(state.parameters()))
    _optimizer_step(eggroll.optimizer, list(state.parameters()))

    assert all(
        parameter not in optimizer.state
        and all(all(candidate is not parameter for candidate in group["params"]) for group in optimizer.param_groups)
        for optimizer in (gradient.optimizer, eggroll.optimizer)
        for parameter in backbone.model.parameters()
    )
    assert all(not parameter.requires_grad for parameter in backbone.model.parameters())
    assert all(
        torch.equal(parameter, before)
        for parameter, before in zip(backbone.model.parameters(), qwen_before, strict=True)
    )
