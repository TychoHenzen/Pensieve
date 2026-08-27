"""Injected contract tests for the Calc-ASDiv_A Stage 0 stream."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from eval.stage0_identity import ASDIV_PARTITION_COUNTS, ASDIV_REVISION
from eval.stream.config import StreamConfig
from eval.stream.events import Observe, Probe
from eval.stream.generators.asdiv_a import (
    AsdivGenerator,
    AsdivRecord,
    asdiv_a_selection_identity,
    load_asdiv_records,
    select_asdiv_a_records,
)
from eval.stream.truth import ProbeTruth


def _row(
    index: int,
    *,
    source_id: str | None = None,
    question: str | None = None,
    result: str | None = None,
    result_float: float | None = None,
) -> dict[str, object]:
    target = result if result is not None else str(index + 1)
    return {
        "id": f"asdiv_a__{index:04d}" if source_id is None else source_id,
        "question": question or f"What is {index} plus one?",
        "result": target,
        "result_float": float(target) if result_float is None else result_float,
    }


def _source_rows() -> list[dict[str, object]]:
    return [_row(index) for index in range(1_218)]


def _records() -> dict[str, tuple[AsdivRecord, ...]]:
    return load_asdiv_records(_source_rows())


def _find(records: dict[str, tuple[AsdivRecord, ...]], item_id: str) -> AsdivRecord:
    return next(record for split in records.values() for record in split if record.id == item_id)


def test_valid_row_uses_source_id_nfc_whitespace_and_canonical_target() -> None:
    rows = _source_rows()
    rows[0] = _row(
        0,
        source_id="source-001",
        question="  Cafe\u0301\t keeps punctuation!\n\nHow many?  ",
        result="003.5000",
        result_float=3.5,
    )

    record = _find(load_asdiv_records(rows), "source-001")

    assert record.question == "Caf\u00e9 keeps punctuation! How many?"
    assert record.target == "3.5"


@pytest.mark.parametrize(
    "replacement",
    [
        _row(0, source_id="", question="valid"),
        _row(0, source_id="empty-question", question=" \t\n "),
        _row(0, source_id="infinite", result_float=float("inf")),
        _row(0, source_id="bad-target", result="not-a-number", result_float=0.0),
    ],
)
def test_malformed_rows_name_the_available_identity(
    replacement: dict[str, object],
) -> None:
    rows = _source_rows()
    rows[0] = replacement

    with pytest.raises(ValueError) as error:
        load_asdiv_records(rows)

    identity = replacement.get("id")
    if identity:
        assert str(identity) in str(error.value)


def test_source_grouping_underscores_use_the_canonical_target_rules() -> None:
    rows = _source_rows()
    rows[0] = _row(
        0,
        source_id="grouped",
        result="3_381/1_450",
        result_float=2.3317241379310345,
    )

    assert _find(load_asdiv_records(rows), "grouped").target == "3381/1450"


def test_partition_counts_and_measured_gate_population_are_fixed() -> None:
    source = _source_rows()
    expected = list(source)
    random.Random(0).shuffle(expected)

    records = load_asdiv_records(source)

    assert {split: len(items) for split, items in records.items()} == dict(
        ASDIV_PARTITION_COUNTS
    )
    assert [record.id for record in records["test"]] == [
        row["id"] for row in expected[:520]
    ]


def test_duplicate_id_or_question_target_across_partitions_is_rejected() -> None:
    rows = _source_rows()
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError, match="duplicate source id"):
        load_asdiv_records(rows)

    rows = _source_rows()
    rows[1]["question"] = rows[0]["question"]
    rows[1]["result"] = rows[0]["result"]
    rows[1]["result_float"] = rows[0]["result_float"]
    with pytest.raises(ValueError, match="duplicate canonical question and target"):
        load_asdiv_records(rows)


def test_selection_is_repeatable_and_binds_every_identity_input() -> None:
    records = _records()
    baseline = select_asdiv_a_records(
        records, split="test", seed=17, problem_count=3
    )
    repeated = select_asdiv_a_records(
        records, split="test", seed=17, problem_count=3
    )
    changed_revision = asdiv_a_selection_identity(
        records=baseline.records,
        split="test",
        seed=17,
        problem_count=3,
        revision="different",
    )
    changed_records = asdiv_a_selection_identity(
        records=(replace(baseline.records[0], question="changed"), *baseline.records[1:]),
        split="test",
        seed=17,
        problem_count=3,
        revision=ASDIV_REVISION,
    )

    assert baseline == repeated
    assert baseline.identity != changed_revision
    assert baseline.identity != changed_records


def test_generator_emits_one_public_question_then_one_numerical_probe() -> None:
    generator = AsdivGenerator(records_by_split=_records())

    items = list(
        generator.generate(
            StreamConfig(
                generator="asdiv-a",
                params={"split": "test", "problem_count": 1},
            ),
            seed=17,
        )
    )

    assert len(items) == 2
    assert isinstance(items[0].event, Observe)
    assert isinstance(items[1].event, Probe)
    assert items[1].event.teaching_position == items[0].event.position
    assert isinstance(items[1].truth, ProbeTruth)
    assert items[1].truth.answer
