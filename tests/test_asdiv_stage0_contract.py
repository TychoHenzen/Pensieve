"""Regression contract for the measured Calc-ASDiv_A Stage 0 replacement."""

from __future__ import annotations

import random
from typing import Any

from eval.gate import token_cot_baseline
from eval.stage0_identity import (
    ASDIV_CONFIGURATION,
    ASDIV_DATASET,
    ASDIV_PARTITION_COUNTS,
    ASDIV_REVISION,
    QWEN_ANSWER_PREFILL,
    QWEN_MATH_PROMPT,
    apply_qwen_chat_template,
)
from eval.stream.generators.asdiv_a import load_asdiv_records


def _rows() -> list[dict[str, object]]:
    return [
        {
            "id": f"asdiv_a__{index:04d}",
            "question": f"What is {index} plus one?",
            "result": str(index + 1),
            "result_float": float(index + 1),
        }
        for index in range(1_218)
    ]


def test_seed_zero_partitions_preserve_the_measured_gate_population() -> None:
    source = _rows()
    expected = list(source)
    random.Random(0).shuffle(expected)

    records = load_asdiv_records(source)

    assert {split: len(items) for split, items in records.items()} == dict(ASDIV_PARTITION_COUNTS)
    assert [record.id for record in records["test"]] == [row["id"] for row in expected[:520]]
    assert [record.id for record in records["train"]] == [row["id"] for row in expected[520:1_090]]
    assert [record.id for record in records["validation"]] == [row["id"] for row in expected[1_090:]]
    partition_ids = [{record.id for record in records[split]} for split in ("train", "validation", "test")]
    assert partition_ids[0].isdisjoint(partition_ids[1])
    assert partition_ids[0].isdisjoint(partition_ids[2])
    assert partition_ids[1].isdisjoint(partition_ids[2])


def test_asdiv_loader_uses_the_pinned_single_source_split() -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def dataset_loader(*args: Any, **kwargs: Any) -> list[dict[str, object]]:
        calls.append((args, kwargs))
        return _rows()

    records = load_asdiv_records(dataset_loader=dataset_loader)

    assert len(records["test"]) == 520
    assert calls == [
        (
            (ASDIV_DATASET, ASDIV_CONFIGURATION),
            {
                "split": "test",
                "revision": ASDIV_REVISION,
                "trust_remote_code": False,
            },
        )
    ]


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, list[list[int]]]:
        self.calls.append((messages, kwargs))
        return {"input_ids": [[1, 2, 3]]}


def test_answer_only_prompt_continues_the_assistant_prefill() -> None:
    tokenizer = _Tokenizer()

    apply_qwen_chat_template(tokenizer, "Two plus two?")

    assert tokenizer.calls == [
        (
            [
                {
                    "role": "user",
                    "content": QWEN_MATH_PROMPT.format(question="Two plus two?"),
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
    assert token_cot_baseline.DEFAULT_MAX_NEW_TOKENS == 16
