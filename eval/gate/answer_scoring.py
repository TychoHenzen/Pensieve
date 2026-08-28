"""Shared numerical-answer normalization for Stage 0 evaluations."""

from __future__ import annotations

import re
from fractions import Fraction

_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.,])[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)/(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)|(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:[.][0-9]+)?)(?![A-Za-z0-9_.,/%])"
)
_MAX_OUTPUT_CODE_POINTS = 4_096
_MAX_CANDIDATE_ASCII_DIGITS = 256
_TOLERANCE = Fraction(1, 1_000_000)
MIN_MEANINGFUL_ACCURACY = 0.05


def extract_predicted_number(text: str) -> str | None:
    """Return the selected finite-syntax numerical candidate, if one exists."""
    if not isinstance(text, str) or len(text) > _MAX_OUTPUT_CODE_POINTS:
        return None

    delimiter_index = text.rfind("####")
    matches = list(_NUMBER_PATTERN.finditer(text, delimiter_index + 4))
    if delimiter_index < 0:
        matches = list(_NUMBER_PATTERN.finditer(text))
    if not matches:
        return None

    match = matches[0] if delimiter_index >= 0 else matches[-1]
    if (
        text[: match.start()].rstrip().endswith("/")
        or text[match.end() :].lstrip().startswith("/")
    ):
        return None

    candidate = match.group(0)
    if sum(character.isascii() and character.isdigit() for character in candidate) > (
        _MAX_CANDIDATE_ASCII_DIGITS
    ):
        return None
    return candidate.replace(",", "")


def _parse_rational(text: str) -> Fraction | None:
    """Parse one complete supported numerical value into a reduced rational."""
    if not isinstance(text, str) or _NUMBER_PATTERN.fullmatch(text) is None:
        return None

    normalized = text.replace(",", "")
    try:
        value = Fraction(normalized)
    except (ValueError, ZeroDivisionError):
        return None
    return value


def score_numerical_answer(generated: str, expected: str) -> bool:
    """Compare generated and expected answers as exact rationals with tolerance."""
    candidate = extract_predicted_number(generated)
    if candidate is None:
        return False

    predicted = _parse_rational(candidate)
    target = _parse_rational(expected)
    if predicted is None or target is None:
        return False
    if predicted == target:
        return True

    tolerance = max(_TOLERANCE, _TOLERANCE * max(abs(predicted), abs(target)))
    return abs(predicted - target) <= tolerance
