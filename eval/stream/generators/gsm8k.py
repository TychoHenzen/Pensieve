"""GSM8K generator: wrap the Hugging Face GSM8K dataset as a probed stream.

Each GSM8K item is a natural-language grade-school math word problem plus a
worked solution ending in `#### <number>`. This generator teaches the word
problem as one `Observe` and immediately follows it with one `Probe` that
asks for the numerical answer, so a subject sees the problem before being
asked to solve it.

The dataset loads once via the Hugging Face `datasets` library and its item
order is fixed by the library, so `derive(seed, "order")` reshuffles it
before drawing, keeping two runs at the same seed identical without relying
on the dataset's own iteration order.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import cast

from datasets import load_dataset

from datasets import load_dataset

from eval.stream.config import StreamConfig
from eval.stream.events import Observe, Probe
from eval.stream.generator import derive
from eval.stream.truth import ProbeTruth, StreamItem

_ANSWER_PATTERN = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")

# Caches the loaded HF dataset split so repeated `generate()` calls (for
# example across seeds in the same process) do not redownload or reparse it.
_dataset_cache: dict[str, list[dict[str, str]]] = {}


def _extract_answer(answer_field: str) -> str:
    match = _ANSWER_PATTERN.search(answer_field)
    if match is None:
        raise ValueError(f"could not find '#### <number>' in GSM8K answer: {answer_field!r}")
    return match.group(1).replace(",", "")


def _load_split(split: str) -> list[dict[str, str]]:
    cached = _dataset_cache.get(split)
    if cached is not None:
        return cached

    hf_dataset = load_dataset("openai/gsm8k", "main", split=split)
    items: list[dict[str, str]] = []
    for raw_row in hf_dataset:
        row = cast(Mapping[str, str], raw_row)
        items.append({"question": row["question"], "answer": row["answer"]})
    _dataset_cache[split] = items
    return items


class GSM8KGenerator:
    """Present GSM8K word problems, then probe for the numerical answer."""

    name = "gsm8k"
    version = "1"

    def chance_rate(self, config: StreamConfig) -> float:
        del config
        return 0.0

    def generate(self, config: StreamConfig, seed: int) -> Iterator[StreamItem]:
        params = config.params
        split = params.get("split", "test")
        problem_count = params.get("problem_count")

        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if problem_count is not None and problem_count < 1:
            raise ValueError(f"problem_count must be at least 1, got {problem_count}")

        problems = _load_split(split)

        order_source = derive(seed, "order")
        shuffled = list(problems)
        order_source.shuffle(shuffled)

        if problem_count is not None:
            shuffled = shuffled[:problem_count]

        position = 0
        for index, problem in enumerate(shuffled):
            question = problem["question"]
            answer = _extract_answer(problem["answer"])

            yield StreamItem(
                event=Observe(position=position, payload={"text": question}),
                truth=None,
            )
            position += 1

            yield StreamItem(
                event=Probe(
                    position=position,
                    probe_id=f"gsm8k-{index}-{position}",
                    task_id="gsm8k",
                    query="What is the numerical answer to the problem above?",
                    teaching_position=position - 1,
                ),
                truth=ProbeTruth(answer=answer),
            )
            position += 1
