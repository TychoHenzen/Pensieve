"""Calc-MAWPS records and deterministic Stage 0 stream generation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import random
import re
import unicodedata
from typing import Any

from eval.gate.answer_scoring import _TOLERANCE
from eval.stage0_identity import (
    CALC_MAWPS_CONFIGURATION,
    CALC_MAWPS_DATASET,
    CALC_MAWPS_RAW_COUNTS,
    CALC_MAWPS_REVISION,
    CALC_MAWPS_VALIDATION_EXCLUDED_ID,
)
from eval.stream.config import StreamConfig
from eval.stream.events import Observe, Probe
from eval.stream.truth import ProbeTruth, StreamItem


_NUMBER = re.compile(
    r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)/"
    r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)|"
    r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)"
)
_SPLITS = tuple(CALC_MAWPS_RAW_COUNTS)


@dataclass(frozen=True, kw_only=True)
class CalcMawpsRecord:
    """One normalized Calc-MAWPS source row."""

    id: str
    split: str
    question: str
    target: str


@dataclass(frozen=True, kw_only=True)
class CalcMawpsSelection:
    """The selected ordered records and their content identity."""

    records: tuple[CalcMawpsRecord, ...]
    split: str
    seed: int
    problem_count: int
    ordered_item_ids: tuple[str, ...]
    identity: str


def _row_error(split: str, position: int, row: object, detail: str) -> ValueError:
    source_id = row.get("id") if isinstance(row, Mapping) else None
    identity = source_id if isinstance(source_id, str) and source_id else str(position)
    return ValueError(f"Calc-MAWPS {split} row {position} ({identity}): {detail}")


def _parse_target(value: object) -> tuple[str, Fraction]:
    if not isinstance(value, str) or len(value) > 512 or _NUMBER.fullmatch(value) is None:
        raise ValueError("result is not a bounded integer, decimal, or fraction")
    normalized = value.replace(",", "")
    if sum(character.isdigit() for character in normalized) > 256:
        raise ValueError("result exceeds 256 digits")
    try:
        if "/" in normalized:
            rational = Fraction(normalized)
        elif "." in normalized:
            decimal = Decimal(normalized)
            if not decimal.is_finite():
                raise ValueError("result is not finite")
            rational = Fraction(decimal)
        else:
            rational = Fraction(int(normalized))
    except (InvalidOperation, ValueError, ZeroDivisionError) as error:
        raise ValueError("result is not a finite numerical value") from error
    if "/" in normalized:
        canonical = (
            str(rational.numerator)
            if rational.denominator == 1
            else f"{rational.numerator}/{rational.denominator}"
        )
    elif "." in normalized:
        canonical = format(Decimal(normalized), "f").rstrip("0").rstrip(".")
        canonical = canonical or "0"
    else:
        canonical = str(rational.numerator)
    return canonical, rational


def _validate_result_float(value: object, target: Fraction) -> None:
    if isinstance(value, bool):
        raise ValueError("result_float is not finite")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("result_float is not finite") from error
    if not decimal_value.is_finite():
        raise ValueError("result_float is not finite")
    float_value = Fraction(decimal_value)
    tolerance = max(_TOLERANCE, _TOLERANCE * max(abs(float_value), abs(target)))
    if abs(float_value - target) > tolerance:
        raise ValueError("result_float disagrees with result")


def _normalize_row(split: str, position: int, row: object) -> CalcMawpsRecord:
    if not isinstance(row, Mapping):
        raise _row_error(split, position, row, "row is not a mapping")
    source_id = row.get("id")
    if not isinstance(source_id, str) or not source_id or len(source_id) > 512:
        raise _row_error(split, position, row, "id must be a non-empty string of at most 512 code points")
    question = row.get("question")
    if not isinstance(question, str) or len(question) > 16_384:
        raise _row_error(split, position, row, "question must be a string of at most 16384 code points")
    question = " ".join(unicodedata.normalize("NFC", question).split())
    if not question:
        raise _row_error(split, position, row, "question is empty after normalization")
    try:
        target, rational = _parse_target(row.get("result"))
        _validate_result_float(row.get("result_float"), rational)
    except ValueError as error:
        raise _row_error(split, position, row, str(error)) from error
    return CalcMawpsRecord(id=source_id, split=split, question=question, target=target)


def _fingerprint(record: CalcMawpsRecord) -> str:
    payload = {"question": record.question, "target": record.target}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_calc_mawps_records(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]], *, strict_raw_counts: bool
) -> dict[str, tuple[CalcMawpsRecord, ...]]:
    """Normalize raw Calc-MAWPS rows and enforce the pinned split contract."""
    if set(rows_by_split) != set(_SPLITS):
        raise ValueError(f"Calc-MAWPS splits must be exactly {_SPLITS!r}")
    records: dict[str, tuple[CalcMawpsRecord, ...]] = {}
    seen_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    excluded_validation: CalcMawpsRecord | None = None
    for split in _SPLITS:
        rows = rows_by_split[split]
        expected = CALC_MAWPS_RAW_COUNTS[split]
        if strict_raw_counts and len(rows) != expected:
            raise ValueError(
                f"Calc-MAWPS {split} raw count mismatch: expected {expected}, found {len(rows)}"
            )
        source_rows = list(rows)
        excluded = [row for row in source_rows if row.get("id") == CALC_MAWPS_VALIDATION_EXCLUDED_ID]
        if split == "validation" and strict_raw_counts:
            if len(excluded) != 1:
                raise ValueError(f"Calc-MAWPS validation pinned exclusion mismatch for {CALC_MAWPS_VALIDATION_EXCLUDED_ID}")
            excluded_position = next(
                index for index, row in enumerate(source_rows)
                if row.get("id") == CALC_MAWPS_VALIDATION_EXCLUDED_ID
            )
            excluded_validation = _normalize_row(
                split, excluded_position, excluded[0]
            )
            source_rows = [row for row in source_rows if row.get("id") != CALC_MAWPS_VALIDATION_EXCLUDED_ID]
        elif excluded:
            raise ValueError(f"Calc-MAWPS {split} contains the validation exclusion id")
        normalized: list[CalcMawpsRecord] = []
        for position, row in enumerate(source_rows):
            record = _normalize_row(split, position, row)
            fingerprint = _fingerprint(record)
            if record.id in seen_ids:
                raise _row_error(split, position, row, "duplicate source id")
            if fingerprint in seen_fingerprints:
                raise _row_error(split, position, row, "duplicate canonical question and target")
            seen_ids.add(record.id)
            seen_fingerprints.add(fingerprint)
            normalized.append(record)
        records[split] = tuple(normalized)
    if strict_raw_counts:
        if excluded_validation is None:
            raise ValueError("Calc-MAWPS validation pinned exclusion is missing")
        paired = next(
            (record for record in records["test"] if record.id == "mawps__yCG5jGSKjPM9koup"),
            None,
        )
        if paired is None or _fingerprint(paired) != _fingerprint(excluded_validation):
            actual = records["test"][0].id if records["test"] else "0"
            raise ValueError(
                f"Calc-MAWPS test row 0 ({actual}): pinned duplicate fingerprint mismatch"
            )
    return records


def load_calc_mawps_records(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, tuple[CalcMawpsRecord, ...]]:
    """Normalize raw Calc-MAWPS rows and enforce the pinned split contract."""
    return _load_calc_mawps_records(rows_by_split, strict_raw_counts=True)


def calc_mawps_selection_identity(
    *, records: Sequence[CalcMawpsRecord], split: str, seed: int, problem_count: int, revision: str
) -> str:
    """Return the digest of the selected ordered records and selection inputs."""
    canonical_records = [
        {"id": record.id, "question": record.question, "target": record.target}
        for record in records
    ]
    records_sha256 = hashlib.sha256(
        json.dumps(canonical_records, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    identity = {
        "schema_version": 1,
        "dataset": CALC_MAWPS_DATASET,
        "configuration": CALC_MAWPS_CONFIGURATION,
        "revision": revision,
        "split": split,
        "seed": seed,
        "problem_count": problem_count,
        "ordered_item_ids": [record.id for record in records],
        "records_sha256": records_sha256,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def select_calc_mawps_records(
    records_by_split: Mapping[str, Sequence[CalcMawpsRecord]], *, split: str, seed: int, problem_count: int | None
) -> CalcMawpsSelection:
    """Shuffle pinned source order with the invocation seed and select its prefix."""
    if split not in _SPLITS:
        raise ValueError(f"unknown Calc-MAWPS split {split!r}")
    records = tuple(records_by_split[split])
    if problem_count is None:
        problem_count = len(records)
    if isinstance(problem_count, bool) or not isinstance(problem_count, int):
        raise TypeError("problem_count must be a non-boolean integer")
    if not 1 <= problem_count <= len(records):
        raise ValueError(f"problem_count must be from 1 through {len(records)}")
    ordered_ids = [record.id for record in records]
    random.Random(seed).shuffle(ordered_ids)
    by_id = {record.id: record for record in records}
    selected = tuple(by_id[item_id] for item_id in ordered_ids[:problem_count])
    return CalcMawpsSelection(
        records=selected,
        split=split,
        seed=seed,
        problem_count=problem_count,
        ordered_item_ids=tuple(ordered_ids[:problem_count]),
        identity=calc_mawps_selection_identity(
            records=selected, split=split, seed=seed, problem_count=problem_count, revision=CALC_MAWPS_REVISION
        ),
    )


class CalcMawpsGenerator:
    """Present Calc-MAWPS questions and probe their numerical answers."""

    name = "calc-mawps"
    version = "1"

    def __init__(self, *, records_by_split: Mapping[str, Sequence[CalcMawpsRecord]] | None = None) -> None:
        self._records_by_split = (
            {split: tuple(records) for split, records in records_by_split.items()}
            if records_by_split is not None
            else None
        )

    def chance_rate(self, config: StreamConfig) -> float:
        del config
        return 0.0

    def _records(self) -> Mapping[str, Sequence[CalcMawpsRecord]]:
        if self._records_by_split is None:
            import datasets

            rows = {
                split: list(
                    datasets.load_dataset(
                        CALC_MAWPS_DATASET,
                        CALC_MAWPS_CONFIGURATION,
                        split=split,
                        revision=CALC_MAWPS_REVISION,
                        trust_remote_code=False,
                    )
                )
                for split in _SPLITS
            }
            self._records_by_split = load_calc_mawps_records(rows)
        return self._records_by_split

    def generate(self, config: StreamConfig, seed: int) -> Iterator[StreamItem]:
        selection = select_calc_mawps_records(
            self._records(),
            split=config.params.get("split", "test"),
            seed=seed,
            problem_count=config.params.get("problem_count"),
        )
        position = 0
        for record in selection.records:
            yield StreamItem(event=Observe(position=position, payload={"text": record.question}), truth=None)
            yield StreamItem(
                event=Probe(
                    position=position + 1,
                    probe_id=f"calc-mawps-{record.id}-{position + 1}",
                    task_id="calc-mawps",
                    query="What is the numerical answer?",
                    teaching_position=position,
                ),
                truth=ProbeTruth(answer=record.target),
            )
            position += 2
