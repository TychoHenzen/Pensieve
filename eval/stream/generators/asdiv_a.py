"""Calc-ASDiv_A records and deterministic Stage 0 stream generation."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from eval.gate.answer_scoring import _TOLERANCE
from eval.stage0_identity import (
    ASDIV_CONFIGURATION,
    ASDIV_DATASET,
    ASDIV_PARTITION_COUNTS,
    ASDIV_PARTITION_SEED,
    ASDIV_REVISION,
    load_asdiv_source,
)
from eval.stream.config import StreamConfig
from eval.stream.events import Observe, Probe
from eval.stream.truth import ProbeTruth, StreamItem

_NUMBER = re.compile(
    r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)/"
    r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)|"
    r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)"
)
_SPLITS = tuple(ASDIV_PARTITION_COUNTS)


@dataclass(frozen=True, kw_only=True)
class AsdivRecord:
    """One normalized Calc-ASDiv_A source row."""

    id: str
    split: str
    question: str
    target: str


@dataclass(frozen=True, kw_only=True)
class AsdivSelection:
    """The selected ordered records and their content identity."""

    records: tuple[AsdivRecord, ...]
    split: str
    seed: int
    problem_count: int
    ordered_item_ids: tuple[str, ...]
    identity: str


def _row_error(split: str, position: int, row: object, detail: str) -> ValueError:
    source_id = row.get("id") if isinstance(row, Mapping) else None
    identity = source_id if isinstance(source_id, str) and source_id else str(position)
    return ValueError(f"Calc-ASDiv_A {split} row {position} ({identity}): {detail}")


def _require_finite_decimal(decimal: Decimal) -> None:
    if not decimal.is_finite():
        raise ValueError("result is not finite")


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
            _require_finite_decimal(decimal)
            rational = Fraction(decimal)
        else:
            rational = Fraction(int(normalized))
    except (InvalidOperation, ValueError, ZeroDivisionError) as error:
        raise ValueError("result is not a finite numerical value") from error
    if "/" in normalized:
        canonical = (
            str(rational.numerator) if rational.denominator == 1 else f"{rational.numerator}/{rational.denominator}"
        )
    elif "." in normalized:
        canonical = format(Decimal(normalized), "f").rstrip("0").rstrip(".")
        canonical = "0" if not canonical or Decimal(normalized).is_zero() else canonical
    else:
        canonical = str(rational.numerator)
    return canonical, rational


def canonicalize_numerical_target(value: str) -> str:
    """Return the exact canonical numerical text used by Calc-ASDiv_A."""
    canonical, _ = _parse_target(value)
    return canonical


def _parse_source_target(value: object) -> tuple[str, Fraction]:
    """Parse pinned Calc-ASDiv_A grouping underscores through the canonical grammar."""
    if isinstance(value, str) and "_" in value:
        if "," in value:
            raise ValueError("result is not a bounded integer, decimal, or fraction")
        value = value.replace("_", ",")
    return _parse_target(value)


def _validate_result_float(value: object, target: Fraction) -> None:
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


def _normalize_row(split: str, position: int, row: object) -> AsdivRecord:
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
        target, rational = _parse_source_target(row.get("result"))
        _validate_result_float(row.get("result_float"), rational)
    except ValueError as error:
        raise _row_error(split, position, row, str(error)) from error
    return AsdivRecord(id=source_id, split=split, question=question, target=target)


def _fingerprint(record: AsdivRecord) -> str:
    payload = {"question": record.question, "target": record.target}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def load_asdiv_records(
    source_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    dataset_loader: Callable[..., Any] | None = None,
) -> dict[str, tuple[AsdivRecord, ...]]:
    """Normalize and partition the pinned single Calc-ASDiv_A source split.

    The first 520 rows after the seed-0 shuffle are the exact population used
    by the measured frozen-Qwen preflight. The remaining rows become the
    disjoint training and validation partitions.
    """
    rows = list(source_rows if source_rows is not None else load_asdiv_source(dataset_loader=dataset_loader))
    expected_total = sum(ASDIV_PARTITION_COUNTS.values())
    if len(rows) != expected_total:
        raise ValueError(f"Calc-ASDiv_A source count mismatch: expected {expected_total}, found {len(rows)}")
    random.Random(ASDIV_PARTITION_SEED).shuffle(rows)
    test_end = ASDIV_PARTITION_COUNTS["test"]
    train_end = test_end + ASDIV_PARTITION_COUNTS["train"]
    rows_by_split = {
        "train": rows[test_end:train_end],
        "validation": rows[train_end:],
        "test": rows[:test_end],
    }

    records: dict[str, tuple[AsdivRecord, ...]] = {}
    seen_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    for split in _SPLITS:
        normalized: list[AsdivRecord] = []
        for position, row in enumerate(rows_by_split[split]):
            record = _normalize_row(split, position, row)
            fingerprint = _fingerprint(record)
            if record.id in seen_ids:
                raise _row_error(split, position, row, "duplicate source id")
            if fingerprint in seen_fingerprints:
                raise _row_error(
                    split,
                    position,
                    row,
                    "duplicate canonical question and target across partitions",
                )
            seen_ids.add(record.id)
            seen_fingerprints.add(fingerprint)
            normalized.append(record)
        expected = ASDIV_PARTITION_COUNTS[split]
        if len(normalized) != expected:
            raise ValueError(f"Calc-ASDiv_A {split} count mismatch: expected {expected}, found {len(normalized)}")
        records[split] = tuple(normalized)
    return records


def load_asdiv_a_records(
    source_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    dataset_loader: Callable[..., Any] | None = None,
) -> dict[str, tuple[AsdivRecord, ...]]:
    """Compatibility spelling for the public Calc-ASDiv_A record loader."""
    return load_asdiv_records(source_rows, dataset_loader=dataset_loader)


def load_asdiv_a_record_split(
    split: str,
    dataset_loader: Callable[..., Any] | None = None,
) -> tuple[AsdivRecord, ...]:
    if split not in _SPLITS:
        raise ValueError(f"unknown Calc-ASDiv_A split {split!r}")
    return load_asdiv_records(dataset_loader=dataset_loader)[split]


def asdiv_a_selection_identity(
    *, records: Sequence[AsdivRecord], split: str, seed: int, problem_count: int, revision: str
) -> str:
    """Return the digest of the selected ordered records and selection inputs."""
    canonical_records = [{"id": record.id, "question": record.question, "target": record.target} for record in records]
    records_sha256 = hashlib.sha256(
        json.dumps(
            canonical_records, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    identity = {
        "schema_version": 1,
        "dataset": ASDIV_DATASET,
        "configuration": ASDIV_CONFIGURATION,
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


def select_asdiv_a_records(
    records_by_split: Mapping[str, Sequence[AsdivRecord]], *, split: str, seed: int, problem_count: int | None
) -> AsdivSelection:
    """Shuffle pinned source order with the invocation seed and select its prefix."""
    if split not in _SPLITS:
        raise ValueError(f"unknown Calc-ASDiv_A split {split!r}")
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
    return AsdivSelection(
        records=selected,
        split=split,
        seed=seed,
        problem_count=problem_count,
        ordered_item_ids=tuple(ordered_ids[:problem_count]),
        identity=asdiv_a_selection_identity(
            records=selected, split=split, seed=seed, problem_count=problem_count, revision=ASDIV_REVISION
        ),
    )


class AsdivGenerator:
    """Present Calc-ASDiv_A questions and probe their numerical answers."""

    name = "asdiv-a"
    version = "1"

    def __init__(self, *, records_by_split: Mapping[str, Sequence[AsdivRecord]] | None = None) -> None:
        self._records_by_split = (
            {split: tuple(records) for split, records in records_by_split.items()}
            if records_by_split is not None
            else None
        )

    def chance_rate(self, config: StreamConfig) -> float:
        del config
        return 0.0

    def _records(self) -> Mapping[str, Sequence[AsdivRecord]]:
        if self._records_by_split is None:
            self._records_by_split = load_asdiv_records()
        return self._records_by_split

    def generate(self, config: StreamConfig, seed: int) -> Iterator[StreamItem]:
        selection = select_asdiv_a_records(
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
                    probe_id=f"asdiv-a-{record.id}-{position + 1}",
                    task_id="asdiv-a",
                    query="What is the numerical answer?",
                    teaching_position=position,
                ),
                truth=ProbeTruth(answer=record.target),
            )
            position += 2
