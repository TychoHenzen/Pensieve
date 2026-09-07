"""Contract tests for Stage 0 numerical answer scoring."""

from __future__ import annotations

import pytest

from eval.gate.answer_scoring import extract_predicted_number, score_numerical_answer


def test_final_delimiter_selects_its_first_candidate() -> None:
    assert extract_predicted_number("draft 9 #### 1/2 then 7") == "1/2"
    assert extract_predicted_number("#### 9 #### 1/2 then 7") == "1/2"
    assert score_numerical_answer("draft 9 #### 1/2 then 7", "0.5")
    assert not score_numerical_answer("draft 9 #### 1/2 then 7", "7")
    assert score_numerical_answer("#### 9 #### 1/2 then 7", "0.5")


def test_without_delimiter_selects_the_final_candidate() -> None:
    assert extract_predicted_number("first 12, final -3") == "-3"
    assert score_numerical_answer("first 12, final -3", "-3")
    assert not score_numerical_answer("first 12, final -3", "12")


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("#### +1,234", "1234"),
        ("#### -1,234", "-1234"),
        ("#### 1,234,567.25", "1234567.25"),
    ],
)
def test_valid_signs_and_grouped_numbers_are_supported(generated: str, expected: str) -> None:
    assert score_numerical_answer(generated, expected)


@pytest.mark.parametrize(
    "generated",
    [
        "#### 12,34",
        "#### 1,23,456",
        "#### 1,234,56",
        "#### 1,",
    ],
)
def test_invalid_comma_grouping_is_not_a_candidate(generated: str) -> None:
    assert extract_predicted_number(generated) is None
    assert not score_numerical_answer(generated, "1234")


# covers: eval/stage0-gate::Numerical answer equivalence::Equivalent fraction and decimal
def test_fraction_and_decimal_reduce_to_the_same_exact_value() -> None:
    assert score_numerical_answer("#### 1/2", "0.500")
    assert score_numerical_answer("#### 2/4", "0.5")


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("#### 1.000001", "1"),
        ("#### 0.000001", "0"),
        ("#### -0.000001", "0"),
    ],
)
# covers: eval/stage0-gate::Numerical answer equivalence::Repeating decimal tolerance
def test_tolerance_accepts_its_inclusive_boundary_including_near_zero(generated: str, expected: str) -> None:
    assert score_numerical_answer(generated, expected)


def test_tolerance_rejects_values_past_its_boundary() -> None:
    assert not score_numerical_answer("#### 1.0000011", "1")
    assert not score_numerical_answer("#### 0.0000011", "0")


@pytest.mark.parametrize("generated", ["#### 0", "#### -0"])
def test_zero_and_negative_zero_are_equal(generated: str) -> None:
    assert score_numerical_answer(generated, "0")


@pytest.mark.parametrize(
    "generated",
    [
        "#### 1/0",
        "#### 0/0",
        "#### NaN",
        "#### inf",
        "#### Infinity",
        "no numerical answer",
    ],
)
# covers: eval/stage0-gate::Numerical answer equivalence::Non-finite answer
def test_non_finite_or_missing_prediction_is_incorrect_without_raising(
    generated: str,
) -> None:
    assert not score_numerical_answer(generated, "1")


def test_missing_or_non_finite_target_is_incorrect_without_raising() -> None:
    for expected in ("", "NaN", "inf", "Infinity"):
        assert not score_numerical_answer("#### 1", expected)


def test_output_code_point_limit_is_inclusive() -> None:
    assert score_numerical_answer("x" * (4096 - len("#### 1")) + "#### 1", "1")
    assert not score_numerical_answer("x" * (4097 - len("#### 1")) + "#### 1", "1")


def test_selected_candidate_ascii_digit_limit_is_inclusive() -> None:
    digits = "1" * 256
    assert score_numerical_answer(f"#### {digits}", digits)
    assert not score_numerical_answer(f"#### {digits}1", f"{digits}1")


@pytest.mark.parametrize(
    "generated",
    [
        "#### 50%",
        "#### 1e3",
        "#### 1E3",
        "#### .5",
        "#### 5.",
        "#### 1 / 2",
        "#### 1/-2",
        "#### \u0661\u0662\u0663",
        "#### \uff11\uff12\uff13",
    ],
)
def test_unsupported_number_syntax_is_incorrect(generated: str) -> None:
    assert extract_predicted_number(generated) is None
    assert not score_numerical_answer(generated, "0.5")
