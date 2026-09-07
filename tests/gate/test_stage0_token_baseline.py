"""Contract tests for the Stage 0 frozen-Qwen token baseline.

Public API assumptions for task 5.2:

``eval.gate.token_cot_baseline.run_baseline`` accepts ``device`` and a
separately named ``development_limit``.  Tests can inject normalized
``AsdivRecord`` values, a frozen-backbone loader, and the deterministic
runtime configurer.  The function returns the version-2 token result schema:
``{schema_version, identity, items, correct, total}``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from eval.gate import token_cot_baseline as baseline
from eval.stage0_identity import QWEN_ANSWER_PREFILL, QWEN_MATH_PROMPT
from eval.stream.generators.asdiv_a import (
    AsdivRecord,
    select_asdiv_a_records,
)

EOS_TOKEN_ID = 151_645
FULL_TEST_COUNT = 520


class _FakeTensor:
    """Small tensor-shaped value whose ``to('cuda')`` never touches CUDA."""

    def __init__(self, rows: Sequence[Sequence[int]]) -> None:
        self._rows = [list(row) for row in rows]
        self.shape = (len(self._rows), len(self._rows[0]))

    def to(self, device: str | torch.device) -> _FakeTensor:
        del device
        return self

    def tolist(self) -> list[list[int]]:
        return [list(row) for row in self._rows]


class _FakeTokenizer:
    eos_token_id = EOS_TOKEN_ID

    def __init__(self, *, prompt_length: int = 3) -> None:
        self.prompt_length = prompt_length
        self.template_calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []
        self.decode_calls: list[tuple[list[int], dict[str, Any]]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, _FakeTensor]:
        self.template_calls.append((messages, kwargs))
        token_ids = list(range(1, self.prompt_length + 1))
        return {
            "input_ids": _FakeTensor([token_ids]),
            "attention_mask": _FakeTensor([[1] * self.prompt_length]),
        }

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str:
        self.decode_calls.append((list(token_ids), kwargs))
        return "0"


class _FakeModel:
    def __init__(self) -> None:
        self.generate_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def generate(self, *args: Any, **kwargs: Any) -> list[list[int]]:
        self.generate_calls.append((args, kwargs))
        input_ids = args[0] if args else kwargs["input_ids"]
        return [input_ids.tolist()[0] + [99]]


def _records(count: int = FULL_TEST_COUNT) -> tuple[AsdivRecord, ...]:
    return tuple(
        AsdivRecord(
            id=f"asdiv_a__test_{index:03d}",
            split="test",
            question=f"What is zero in problem {index}?",
            target="0",
        )
        for index in range(count)
    )


def _backbone_loader(tokenizer: _FakeTokenizer, model: _FakeModel) -> Callable[..., Any]:
    def load(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return SimpleNamespace(
            tokenizer=tokenizer,
            model=model,
            config=SimpleNamespace(hidden_size=896),
            generation_config=SimpleNamespace(eos_token_id=EOS_TOKEN_ID),
        )

    return load


def _run(
    records: Sequence[AsdivRecord],
    *,
    development_limit: int | None,
    tokenizer: _FakeTokenizer | None = None,
    model: _FakeModel | None = None,
    device: str = "cpu",
    runtime_configurer: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], _FakeTokenizer, _FakeModel]:
    tokenizer = tokenizer or _FakeTokenizer()
    model = model or _FakeModel()
    kwargs: dict[str, Any] = {
        "device": device,
        "development_limit": development_limit,
        "records": tuple(records),
        "backbone_loader": _backbone_loader(tokenizer, model),
    }
    if runtime_configurer is not None:
        kwargs["runtime_configurer"] = runtime_configurer
    result = baseline.run_baseline(**kwargs)
    return result, tokenizer, model


# covers: eval/stage0-gate::Frozen Qwen token baseline::Full token baseline
def test_full_baseline_persists_every_seed_zero_test_identifier_in_order() -> None:
    records = _records()
    expected = select_asdiv_a_records(
        {"test": records},
        split="test",
        seed=0,
        problem_count=None,
    ).ordered_item_ids

    result, _, model = _run(records, development_limit=None)

    assert set(result) == {"schema_version", "identity", "items", "correct", "total"}
    assert result["schema_version"] == 2
    assert result["total"] == FULL_TEST_COUNT
    assert len(model.generate_calls) == FULL_TEST_COUNT
    assert tuple(item["item_id"] for item in result["items"]) == expected
    assert result["identity"]["development_only"] is False


# covers: eval/stage0-gate::Frozen Qwen token baseline::Official prompt format
def test_official_qwen_prompt_template_and_greedy_generation_are_exact() -> None:
    records = _records()
    expected_selection = select_asdiv_a_records({"test": records}, split="test", seed=0, problem_count=None)

    result, tokenizer, model = _run(records, development_limit=1)

    selected = expected_selection.records[0]
    assert tokenizer.template_calls == [
        (
            [
                {
                    "role": "user",
                    "content": QWEN_MATH_PROMPT.format(question=selected.question),
                },
                {"role": "assistant", "content": QWEN_ANSWER_PREFILL},
            ],
            {
                "tokenize": True,
                "continue_final_message": True,
                "return_dict": True,
                "return_tensors": "pt",
                "padding": False,
                "truncation": False,
            },
        )
    ]
    args, kwargs = model.generate_calls[0]
    input_ids = args[0] if args else kwargs["input_ids"]
    assert input_ids.shape[0] == 1
    assert kwargs["do_sample"] is False
    assert kwargs["num_beams"] == 1
    assert kwargs["max_new_tokens"] == 16
    assert kwargs["use_cache"] is True
    assert kwargs["eos_token_id"] == EOS_TOKEN_ID
    assert kwargs["pad_token_id"] == EOS_TOKEN_ID
    assert result["items"] == [
        {
            "item_id": selected.id,
            "completion": "0",
            "generated_token_count": 1,
            "hit_token_limit": False,
            "prediction": "0",
            "target": "0",
            "correct": True,
        }
    ]


def test_pinned_multi_eos_config_selects_the_tokenizer_eos_deterministically() -> None:
    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    def load_backbone(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return SimpleNamespace(
            tokenizer=tokenizer,
            model=model,
            config=SimpleNamespace(hidden_size=896),
            generation_config=SimpleNamespace(eos_token_id=[151_645, 151_643]),
        )

    result = baseline.run_baseline(
        device="cpu",
        development_limit=1,
        records=_records(),
        backbone_loader=load_backbone,
    )

    _, generation = model.generate_calls[0]
    assert generation["eos_token_id"] == EOS_TOKEN_ID
    assert generation["pad_token_id"] == EOS_TOKEN_ID
    assert result["identity"]["token_generation"]["eos_token_id"] == EOS_TOKEN_ID
    assert result["identity"]["token_generation"]["pad_token_id"] == EOS_TOKEN_ID


def test_multi_eos_config_must_contain_the_tokenizer_eos() -> None:
    tokenizer = _FakeTokenizer()

    def load_backbone(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return SimpleNamespace(
            tokenizer=tokenizer,
            model=_FakeModel(),
            config=SimpleNamespace(hidden_size=896),
            generation_config=SimpleNamespace(eos_token_id=[151_643]),
        )

    with pytest.raises(ValueError, match=r"tokenizer.*generation config.*EOS"):
        baseline.run_baseline(
            device="cpu",
            development_limit=1,
            records=_records(),
            backbone_loader=load_backbone,
        )


def test_context_over_512_tokens_is_rejected_without_generation() -> None:
    tokenizer = _FakeTokenizer(prompt_length=513)
    model = _FakeModel()

    with pytest.raises(ValueError, match=r"512"):
        _run(
            _records(),
            development_limit=1,
            tokenizer=tokenizer,
            model=model,
        )

    assert model.generate_calls == []
    assert tokenizer.template_calls[0][1]["truncation"] is False


@pytest.mark.parametrize("development_limit", [False, 0, 521])
def test_development_limit_is_a_bounded_non_boolean_integer(
    development_limit: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match=r"development_limit.*1.*520"):
        _run(_records(), development_limit=development_limit)  # type: ignore[arg-type]


def test_limited_selection_is_explicitly_development_only_and_non_gating() -> None:
    records = _records()
    full_selection = select_asdiv_a_records({"test": records}, split="test", seed=0, problem_count=None)

    result, _, _ = _run(records, development_limit=7)

    assert tuple(item["item_id"] for item in result["items"]) == (full_selection.ordered_item_ids[:7])
    assert result["total"] == 7
    assert result["identity"]["development_only"] is True
    assert "pass" not in result
    assert "gate_decision" not in result


def test_deterministic_runtime_is_configured_before_cuda_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    events: list[str] = []
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_benchmark = torch.backends.cudnn.benchmark

    def configure_runtime() -> None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        events.append("runtime")

    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    def load_backbone(*args: Any, **kwargs: Any) -> Any:
        del args
        assert kwargs["device"] == "cuda"
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.benchmark is False
        events.append("cuda-model")
        return SimpleNamespace(
            tokenizer=tokenizer,
            model=model,
            config=SimpleNamespace(hidden_size=896),
            generation_config=SimpleNamespace(eos_token_id=EOS_TOKEN_ID),
        )

    try:
        result = baseline.run_baseline(
            device="cuda",
            development_limit=1,
            records=_records(),
            backbone_loader=load_backbone,
            runtime_configurer=configure_runtime,
        )
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.backends.cudnn.benchmark = previous_benchmark

    assert events == ["runtime", "cuda-model"]
    expected_runtime = {
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "tf32_enabled": False,
        "cudnn_benchmark": False,
    }
    for name, value in expected_runtime.items():
        assert result["identity"]["runtime"][name] == value
