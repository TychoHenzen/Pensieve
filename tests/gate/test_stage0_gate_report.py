"""Contract tests for the exact Stage 0 gate decision.

Public API assumptions for task 5.7:

``eval.gate.gate_report.evaluate_gate_results(token_result, latent_result)``
accepts already validated version-2 result mappings. It derives token accuracy
from ``correct/total`` and latent mean from each run's ``correct/total``. It
returns a report mapping with the existing ``criteria`` names and a boolean
``pass`` field. It never trusts cached floating-point ``accuracy`` or ``mean``
fields.

File loading and identity validation remain the responsibility of
``build_report`` and ``eval.gate.result_cache`` in task 5.7.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
import torch

from eval.gate import gate_report, latent_eval
from eval.stream.generators.calc_mawps import (
    CalcMawpsRecord,
    select_calc_mawps_records,
)


FULL_TEST_COUNT = 520
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def _items(correct: int) -> list[dict[str, Any]]:
    return [
        {
            "item_id": f"mawps__test_{index:03d}",
            "prediction": str(index) if index < correct else "wrong",
            "target": str(index),
            "correct": index < correct,
        }
        for index in range(FULL_TEST_COUNT)
    ]


def _token_result(correct: int) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "identity": {"development_only": False},
        "items": _items(correct),
        "correct": correct,
        "total": FULL_TEST_COUNT,
    }


def _latent_result(
    correct_by_seed: Sequence[int],
    *,
    seeds: Sequence[int | bool] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "identity": {"development_only": False, "seeds": list(seeds)},
        "checkpoint_sha256": "0" * 64,
        "seeds": list(seeds),
        "runs": [
            {
                "seed": seed,
                "items": _items(correct),
                "correct": correct,
                "total": FULL_TEST_COUNT,
            }
            for seed, correct in zip(seeds, correct_by_seed, strict=True)
        ],
    }


@pytest.mark.parametrize(
    ("token_correct", "expected_valid", "expected_pass"),
    [(25, False, False), (26, True, True)],
)
# covers: eval/stage0-gate::Stage 0 pass condition::Baseline below floor
def test_gate_uses_exact_26_of_520_baseline_floor(
    token_correct: int, expected_valid: bool, expected_pass: bool
) -> None:
    report = gate_report.evaluate_gate_results(
        _token_result(token_correct),
        _latent_result([26, 26, 26, 26, 26]),
    )

    assert report["criteria"]["criterion_baseline_meaningful"] is expected_valid
    assert report["pass"] is expected_pass


@pytest.mark.parametrize(
    "seeds",
    [
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 3],
        [1, 0, 2, 3, 4],
        [False, 1, 2, 3, 4],
    ],
    ids=["missing", "extra", "duplicate", "reordered", "boolean"],
)
# covers: eval/stage0-gate::Stage 0 pass condition::Too few latent seeds
def test_gate_requires_exact_ordered_non_boolean_default_seeds(
    seeds: list[int | bool],
) -> None:
    report = gate_report.evaluate_gate_results(
        _token_result(26),
        _latent_result([26] * len(seeds), seeds=seeds),
    )

    assert report["criteria"]["criterion_min_seeds"] is False
    assert report["pass"] is False


@pytest.mark.parametrize(
    ("latent_correct", "expected_meets", "expected_pass"),
    [
        ([0, 1, 33, 45, 51], True, True),
        ([0, 1, 33, 45, 50], False, False),
    ],
    ids=["exact-unrounded-equality", "rounded-display-equality-only"],
)
# covers: eval/stage0-gate::Stage 0 pass condition::Latent result meets baseline
def test_gate_compares_unrounded_mean_of_validated_per_seed_fractions(
    latent_correct: list[int], expected_meets: bool, expected_pass: bool
) -> None:
    report = gate_report.evaluate_gate_results(
        _token_result(26),
        _latent_result(latent_correct),
    )

    assert report["criteria"]["criterion_latent_meets_baseline"] is expected_meets
    assert report["pass"] is expected_pass


def _records() -> tuple[CalcMawpsRecord, ...]:
    return tuple(
        CalcMawpsRecord(
            id=f"mawps__test_{index:03d}",
            split="test",
            question=f"What is {index} plus zero?",
            target=str(index),
        )
        for index in range(FULL_TEST_COUNT)
    )


def _persisted_token_result(records: Sequence[CalcMawpsRecord]) -> dict[str, Any]:
    ordered = select_calc_mawps_records(
        {"test": tuple(records)}, split="test", seed=0, problem_count=None
    ).ordered_item_ids
    return {
        "schema_version": 2,
        "identity": {
            "selection": {
                "split": "test",
                "seed": 0,
                "problem_count": FULL_TEST_COUNT,
                "ordered_item_ids": list(ordered),
            },
            "development_only": False,
        },
        "items": [{"item_id": item_id} for item_id in ordered],
    }


def test_each_latent_seed_initializes_all_rngs_before_subject_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int]] = []

    monkeypatch.setattr(random, "seed", lambda seed: events.append(("python", seed)))
    monkeypatch.setattr(np.random, "seed", lambda seed: events.append(("numpy", seed)))
    monkeypatch.setattr(torch, "manual_seed", lambda seed: events.append(("torch", seed)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: events.append(("cuda", seed)),
    )

    def subject_loader(*args: Any, **kwargs: Any) -> object:
        del args, kwargs
        subjects = sum(event[0] == "subject" for event in events)
        seed = DEFAULT_SEEDS[subjects]
        events.append(("subject", seed))
        return object()

    records = _records()
    latent_eval.run_eval(
        checkpoint_path=None,
        seeds=list(DEFAULT_SEEDS),
        development_limit=1,
        slot_count=16,
        num_steps=2,
        device="cpu",
        token_result=_persisted_token_result(records),
        records=records,
        subject_loader=subject_loader,
        item_evaluator=lambda _subject, record: {
            "prediction": record.target,
            "input_ids": [1, 2, 3],
        },
        runtime_configurer=lambda: None,
    )

    assert events == [
        event
        for seed in DEFAULT_SEEDS
        for event in (
            ("python", seed),
            ("numpy", seed),
            ("torch", seed),
            ("cuda", seed),
            ("subject", seed),
        )
    ]
