"""Stage 0 latent-evaluation selection and result contract tests.

Public API assumptions for task 5.4:

``eval.gate.latent_eval.run_eval`` accepts keyword-only ``token_result`` and
``records`` inputs plus a separately named ``development_limit``.  Tests can
inject ``subject_loader`` and ``item_evaluator`` callables.  The item evaluator
receives ``(subject, record)`` and returns a mapping with the raw or canonical
``prediction`` and the exact rendered ``input_ids`` used for that record.

The function returns the version-2 latent result schema
``{schema_version, identity, checkpoint_sha256, seeds, runs}``.  Each run is
``{seed, items, correct, total}`` and each item uses the token result's shared
``{item_id, prediction, target, correct}`` schema.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from eval.gate import latent_eval
from eval.stage0_identity import canonical_json_bytes
from eval.stream.generators.asdiv_a import (
    AsdivRecord,
    select_asdiv_a_records,
)

FULL_TEST_COUNT = 520
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def _records() -> tuple[AsdivRecord, ...]:
    return tuple(
        AsdivRecord(
            id=f"asdiv_a__test_{index:03d}",
            split="test",
            question=f"What is {index} plus zero?",
            target=str(index),
        )
        for index in range(FULL_TEST_COUNT)
    )


def _persisted_order(records: Sequence[AsdivRecord]) -> tuple[str, ...]:
    return select_asdiv_a_records(
        {"test": tuple(records)},
        split="test",
        seed=0,
        problem_count=None,
    ).ordered_item_ids


def _input_ids(item_id: str) -> list[int]:
    ordinal = int(item_id.rsplit("_", 1)[1])
    return [17, ordinal, 23]


def _rendered_digest(item_ids: Sequence[str]) -> str:
    rendered = [{"item_id": item_id, "input_ids": _input_ids(item_id)} for item_id in item_ids]
    return hashlib.sha256(canonical_json_bytes(rendered)).hexdigest()


def _token_result(
    records: Sequence[AsdivRecord],
    *,
    ordered_item_ids: Sequence[str] | None = None,
    item_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    ordered = list(ordered_item_ids or _persisted_order(records))
    persisted_items = list(item_ids if item_ids is not None else ordered)
    targets = {record.id: record.target for record in records}
    return {
        "schema_version": 2,
        "identity": {
            "selection": {
                "schema_version": 1,
                "split": "test",
                "seed": 0,
                "problem_count": len(ordered),
                "ordered_item_ids": ordered,
                "identity_sha256": "persisted-token-selection",
            },
            "rendered_inputs_sha256": (
                _rendered_digest(ordered)
                if all(item_id in targets for item_id in ordered)
                else "unknown-token-rendering"
            ),
            "development_only": False,
        },
        "items": [
            {
                "item_id": item_id,
                "prediction": targets.get(item_id, "0"),
                "target": targets.get(item_id, "0"),
                "correct": True,
            }
            for item_id in persisted_items
        ],
        "correct": len(persisted_items),
        "total": len(persisted_items),
    }


def _run(
    *,
    records: Sequence[AsdivRecord],
    token_result: Mapping[str, Any],
    development_limit: int | None,
) -> tuple[dict[str, Any], list[list[str]]]:
    calls_by_subject: list[list[str]] = []

    def load_subject(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        calls_by_subject.append([])
        return SimpleNamespace(run_index=len(calls_by_subject) - 1)

    def evaluate_item(subject: Any, record: AsdivRecord) -> dict[str, Any]:
        calls_by_subject[subject.run_index].append(record.id)
        return {"prediction": record.target, "input_ids": _input_ids(record.id)}

    result = latent_eval.run_eval(
        checkpoint_path=None,
        seeds=list(DEFAULT_SEEDS),
        development_limit=development_limit,
        slot_count=16,
        num_steps=2,
        device="cpu",
        token_result=token_result,
        records=tuple(records),
        subject_loader=load_subject,
        item_evaluator=evaluate_item,
    )
    return result, calls_by_subject


# covers: eval/stage0-gate::Comparable test selection::Full evaluation identity
def test_full_latent_eval_reuses_every_persisted_token_identifier_and_schema() -> None:
    records = _records()
    ordered = _persisted_order(records)

    result, calls_by_subject = _run(
        records=records,
        token_result=_token_result(records),
        development_limit=None,
    )

    assert set(result) == {
        "schema_version",
        "identity",
        "checkpoint_sha256",
        "seeds",
        "runs",
    }
    assert result["schema_version"] == 2
    assert result["seeds"] == DEFAULT_SEEDS
    assert result["identity"]["selection"]["ordered_item_ids"] == list(ordered)
    assert result["identity"]["rendered_inputs_sha256"] == _rendered_digest(ordered)
    assert result["identity"]["development_only"] is False
    assert calls_by_subject == [list(ordered) for _ in DEFAULT_SEEDS]

    for run, seed in zip(result["runs"], DEFAULT_SEEDS, strict=True):
        assert set(run) == {"seed", "items", "correct", "total"}
        assert run["seed"] == seed
        assert run["total"] == FULL_TEST_COUNT
        assert tuple(item["item_id"] for item in run["items"]) == ordered
        assert all(set(item) == {"item_id", "prediction", "target", "correct"} for item in run["items"])


# covers: eval/stage0-gate::Comparable test selection::Limited evaluation identity
def test_limited_latent_eval_uses_persisted_prefix_and_subset_rendered_digest() -> None:
    records = _records()
    ordered = _persisted_order(records)
    expected_subset = ordered[:7]

    result, calls_by_subject = _run(
        records=records,
        token_result=_token_result(records),
        development_limit=7,
    )

    assert result["identity"]["selection"]["ordered_item_ids"] == list(expected_subset)
    assert result["identity"]["selection"]["problem_count"] == 7
    assert result["identity"]["rendered_inputs_sha256"] == _rendered_digest(expected_subset)
    assert result["identity"]["rendered_inputs_sha256"] != _rendered_digest(ordered)
    assert result["identity"]["development_only"] is True
    assert calls_by_subject == [list(expected_subset) for _ in DEFAULT_SEEDS]
    assert all(tuple(item["item_id"] for item in run["items"]) == expected_subset for run in result["runs"])
    assert "pass" not in result
    assert "gate_decision" not in result


def test_latent_eval_reports_each_completed_item_through_optional_callback() -> None:
    records = _records()
    events: list[Any] = []

    latent_eval.run_eval(
        checkpoint_path=None,
        seeds=[7, 11],
        development_limit=2,
        slot_count=16,
        num_steps=2,
        device="cpu",
        token_result=_token_result(records),
        records=records,
        subject_loader=lambda *_args, **_kwargs: object(),
        item_evaluator=lambda _subject, record: {
            "prediction": record.target,
            "input_ids": _input_ids(record.id),
        },
        on_item=events.append,
    )

    assert [event.seed for event in events] == [7, 7, 11, 11]
    assert [event.item_index for event in events] == [1, 2, 1, 2]
    assert [event.seed_correct for event in events] == [1, 2, 1, 2]
    assert [event.global_done for event in events] == [1, 2, 3, 4]
    assert [event.global_correct for event in events] == [1, 2, 3, 4]
    assert all(event.item_count == 2 for event in events)
    assert all(event.global_count == 4 for event in events)


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("unknown", r"unknown.*item|item.*unknown"),
        ("missing", r"missing.*item|item.*missing|coverage"),
        ("duplicate", r"duplicate.*item|item.*duplicate"),
    ],
)
def test_persisted_token_identifier_errors_reject_before_subject_construction(
    case: str,
    expected_error: str,
) -> None:
    records = _records()
    ordered = list(_persisted_order(records))
    item_ids = list(ordered)
    if case == "unknown":
        ordered[0] = "asdiv_a__unknown"
        item_ids[0] = "asdiv_a__unknown"
    elif case == "missing":
        item_ids.pop()
    else:
        ordered[1] = ordered[0]
        item_ids[1] = item_ids[0]

    constructed = 0

    def fail_if_constructed(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        nonlocal constructed
        constructed += 1
        raise AssertionError("invalid identifiers reached subject construction")

    with pytest.raises(ValueError, match=expected_error):
        latent_eval.run_eval(
            checkpoint_path=None,
            seeds=list(DEFAULT_SEEDS),
            development_limit=None,
            slot_count=16,
            num_steps=2,
            device="cpu",
            token_result=_token_result(
                records,
                ordered_item_ids=ordered,
                item_ids=item_ids,
            ),
            records=records,
            subject_loader=fail_if_constructed,
            item_evaluator=lambda subject, record: {
                "prediction": record.target,
                "input_ids": _input_ids(record.id),
            },
        )

    assert constructed == 0
