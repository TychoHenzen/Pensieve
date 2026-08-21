"""Injected contract tests for the Calc-MAWPS Stage 0 stream."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import random

import pytest

from eval.stage0_identity import (
    CALC_MAWPS_REVISION,
    CALC_MAWPS_VALIDATION_EXCLUDED_ID,
)
from eval.stream.config import StreamConfig
from eval.stream.events import Observe, Probe
from eval.stream.generators.calc_mawps import (
    CalcMawpsGenerator,
    CalcMawpsRecord,
    calc_mawps_selection_identity,
    load_calc_mawps_records,
    select_calc_mawps_records,
)
from eval.stream.render import render_event
from eval.stream.truth import ProbeTruth, subject_view


def _row(
    source_id: str,
    question: str = "What is two plus one?",
    result: str = "3",
    result_float: float = 3.0,
) -> dict[str, object]:
    return {
        "id": source_id,
        "question": question,
        "result": result,
        "result_float": result_float,
    }


def _alphabetical_index(index: int) -> str:
    value = index + 1
    characters: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        characters.append(chr(ord("a") + remainder))
    return "".join(reversed(characters))


def _pinned_rows() -> dict[str, list[dict[str, object]]]:
    train = [
        _row(f"train-{index}", question=f"Train question {_alphabetical_index(index)}")
        for index in range(1089)
    ]
    validation = [
        _row(
            f"validation-{index}",
            question=f"Validation question {_alphabetical_index(index)}",
        )
        for index in range(1039)
    ]
    validation.append(
        _row(
            CALC_MAWPS_VALIDATION_EXCLUDED_ID,
            question="Pinned duplicate question",
        )
    )
    test = [
        _row(f"test-{index}", question=f"Test question {_alphabetical_index(index)}")
        for index in range(520)
    ]
    test[0] = _row("mawps__yCG5jGSKjPM9koup", question="Pinned duplicate question")
    return {"train": train, "validation": validation, "test": test}


def _records() -> dict[str, tuple[CalcMawpsRecord, ...]]:
    return load_calc_mawps_records(_pinned_rows())


def _selection(
    records: dict[str, tuple[CalcMawpsRecord, ...]] | None = None,
    **overrides: object,
):
    arguments: dict[str, object] = {
        "split": "test",
        "seed": 17,
        "problem_count": 3,
    }
    arguments.update(overrides)
    return select_calc_mawps_records(records or _records(), **arguments)


# covers: eval/generators/calc-mawps::Canonical math problem records::Valid row normalization
def test_valid_row_uses_source_id_exact_nfc_and_split_join_question_normalization():
    rows = _pinned_rows()
    rows["train"][0] = _row(
        "source-001",
        "  Cafe\u0301\t keeps  punctuation!\n\nHow many?  ",
        "003.5000",
        3.5,
    )

    record = load_calc_mawps_records(rows)["train"][0]

    assert record.id == "source-001"
    assert record.split == "train"
    assert record.question == "Caf\u00e9 keeps punctuation! How many?"
    assert record.target == "3.5"


# covers: eval/generators/calc-mawps::Canonical math problem records::Valid row normalization
def test_source_id_and_source_fields_have_declared_bounds():
    rows = _pinned_rows()
    rows["train"][0] = _row("")
    with pytest.raises(ValueError, match=r"train.*0"):
        load_calc_mawps_records(rows)

    rows = _pinned_rows()
    rows["train"][0] = _row("x" * 513)
    with pytest.raises(ValueError, match=r"train.*0.*x"):
        load_calc_mawps_records(rows)

    rows = _pinned_rows()
    rows["train"][0] = _row("bounded", question="q" * 16_385)
    with pytest.raises(ValueError, match=r"train.*0.*bounded"):
        load_calc_mawps_records(rows)

    rows = _pinned_rows()
    rows["train"][0] = _row("target-bound", result="1" * 257, result_float=1.0)
    with pytest.raises(ValueError, match=r"train.*0.*target-bound"):
        load_calc_mawps_records(rows)


@pytest.mark.parametrize(
    ("replacement", "identity"),
    [
        (_row("empty-question", question=" \t\n "), "empty-question"),
        (_row("infinite-float", result_float=float("inf")), "infinite-float"),
        ({"question": "missing id", "result": "3", "result_float": 3.0}, "0"),
        (_row("bad-target", result="not a number"), "bad-target"),
    ],
)
# covers: eval/generators/calc-mawps::Canonical math problem records::Malformed row rejected
def test_malformed_row_names_its_split_position_and_available_identity(replacement, identity):
    rows = _pinned_rows()
    rows["train"][0] = replacement

    with pytest.raises(ValueError) as error:
        load_calc_mawps_records(rows)

    message = str(error.value)
    assert "train" in message
    assert "0" in message
    assert identity in message


# covers: eval/generators/calc-mawps::Canonical math problem records::Valid row normalization
def test_result_float_agrees_via_its_decimal_string_and_target_is_canonical():
    rows = _pinned_rows()
    source_float = 0.1000005
    rows["train"][0] = _row(
        "decimal-string", result="0.1000005", result_float=source_float
    )

    record = load_calc_mawps_records(rows)["train"][0]

    assert Decimal(str(source_float)) == Decimal("0.1000005")
    assert record.target == "0.1000005"


# covers: eval/generators/calc-mawps::Filtered Calc-MAWPS split binding::Default filtered splits load
def test_pinned_raw_counts_are_required_and_only_pinned_validation_id_is_removed():
    records = _records()

    assert {split: len(value) for split, value in records.items()} == {
        "train": 1089,
        "validation": 1039,
        "test": 520,
    }
    assert CALC_MAWPS_VALIDATION_EXCLUDED_ID not in {
        record.id for record in records["validation"]
    }

    rows = _pinned_rows()
    rows["validation"].pop()
    with pytest.raises(ValueError, match=r"validation.*1040"):
        load_calc_mawps_records(rows)


# covers: eval/generators/calc-mawps::Filtered Calc-MAWPS split binding::Default filtered splits load
def test_every_cross_split_id_or_question_target_duplicate_except_pinned_pair_is_rejected():
    rows = _pinned_rows()
    rows["test"][0] = _row("train-0", question="A different question", result="4", result_float=4.0)
    with pytest.raises(ValueError, match=r"test.*0.*train-0"):
        load_calc_mawps_records(rows)

    rows = _pinned_rows()
    rows["test"][0] = _row("distinct-id", question="What is two plus one?")
    with pytest.raises(ValueError, match=r"test.*0.*distinct-id"):
        load_calc_mawps_records(rows)

    rows = _pinned_rows()
    records = load_calc_mawps_records(rows)
    assert len(records["validation"]) == 1039


# covers: eval/generators/calc-mawps::Selection identity is deterministic::Repeated selection
def test_omitted_count_normalizes_to_usable_split_cardinality_and_selection_is_repeatable():
    records = _records()
    first = _selection(records, problem_count=None)
    second = _selection(records, problem_count=None)

    assert first.problem_count == 520
    assert len(first.records) == 520
    assert first.ordered_item_ids == second.ordered_item_ids
    assert first.identity == second.identity


# covers: eval/generators/calc-mawps::Selection identity is deterministic::Repeated selection
@pytest.mark.parametrize("problem_count", [0, -1, 521, 1.0, "1", True, False])
def test_selection_count_is_a_non_boolean_integer_inside_the_split_limit(problem_count):
    with pytest.raises((TypeError, ValueError)):
        _selection(problem_count=problem_count)


# covers: eval/generators/calc-mawps::Selection identity is deterministic::Repeated selection
def test_selection_uses_python_random_seeded_identifier_order_and_prefix_limit():
    records = _records()
    selection = _selection(records, problem_count=3, seed=19)
    expected_ids = [record.id for record in records["test"]]
    random.Random(19).shuffle(expected_ids)

    assert len(selection.records) == 3
    assert selection.ordered_item_ids == tuple(expected_ids[:3])
    assert selection.ordered_item_ids == tuple(record.id for record in selection.records)
    assert selection.records == tuple(selection.records)


# covers: eval/generators/calc-mawps::Selection identity is deterministic::Selection input changes
def test_selection_identity_changes_for_every_declared_identity_input():
    records = _records()
    baseline = _selection(records)
    changed_split = _selection(records, split="train")
    changed_count = _selection(records, problem_count=2)
    changed_seed = _selection(records, seed=18)
    changed_revision = calc_mawps_selection_identity(
        records=baseline.records,
        split="test",
        seed=17,
        problem_count=3,
        revision="different-revision",
    )
    changed_records = calc_mawps_selection_identity(
        records=(
            replace(baseline.records[0], question="changed question"),
            *baseline.records[1:],
        ),
        split="test",
        seed=17,
        problem_count=3,
        revision=CALC_MAWPS_REVISION,
    )

    assert baseline.identity != changed_split.identity
    assert baseline.identity != changed_count.identity
    assert baseline.identity != changed_seed.identity
    assert baseline.identity != changed_revision
    assert baseline.identity != changed_records


# covers: eval/generators/calc-mawps::Calc-MAWPS stream follows the v1 schema::Observe and probe pair
def test_generator_emits_one_question_observe_then_one_exact_numerical_probe():
    records = _records()
    generator = CalcMawpsGenerator(records_by_split=records)
    items = list(
        generator.generate(
            StreamConfig(generator="calc-mawps", params={"split": "test", "problem_count": 1}),
            seed=17,
        )
    )

    assert len(items) == 2
    assert isinstance(items[0].event, Observe)
    assert items[0].event.payload == {"text": items[0].event.payload["text"]}
    assert isinstance(items[1].event, Probe)
    assert items[1].event.query == "What is the numerical answer?"
    assert items[1].event.teaching_position == items[0].event.position
    assert isinstance(items[1].truth, ProbeTruth)
    assert items[1].truth.answer == "3"


# covers: eval/generators/calc-mawps::Calc-MAWPS stream follows the v1 schema::Truth remains isolated
def test_rendered_and_fake_model_facing_inputs_contain_only_public_strings():
    records = _records()
    generator = CalcMawpsGenerator(records_by_split=records)
    items = list(
        generator.generate(
            StreamConfig(generator="calc-mawps", params={"split": "test", "problem_count": 1}),
            seed=17,
        )
    )
    probe = items[1]
    assert isinstance(probe.event, Probe)
    assert probe.truth is not None

    model_facing_arguments: list[object] = []

    def fake_model_boundary(text: str) -> None:
        model_facing_arguments.append(text)

    for event in subject_view(items):
        fake_model_boundary(render_event(event))

    secret_values = {
        str(probe.truth.answer),
        items[0].event.payload["text"],
        probe.event.probe_id,
        probe.event.task_id,
    }
    assert all(isinstance(argument, str) for argument in model_facing_arguments)
    assert all(
        secret not in argument
        for argument in model_facing_arguments
        for secret in secret_values - {items[0].event.payload["text"]}
    )
