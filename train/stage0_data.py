"""Shared deterministic Calc-MAWPS selections for Stage 0 training."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from eval.stream.generators.calc_mawps import (
    CalcMawpsRecord,
    CalcMawpsSelection,
    load_calc_mawps_record_split,
    select_calc_mawps_records,
)


STAGE0_SEED = 0
HELD_OUT_COUNT = 128
TRAINING_MODES = frozenset({"gradient", "eggroll", "alternating"})
TrainingMode = Literal["gradient", "eggroll", "alternating"]


@dataclass(frozen=True)
class Stage0Dataset:
    """The persisted train order and validation-only held-out selection."""

    train_selection: CalcMawpsSelection
    held_out_selection: CalcMawpsSelection

    def training_records(
        self, *, mode: TrainingMode | str, epoch: int
    ) -> tuple[CalcMawpsRecord, ...]:
        self._validate_training_request(mode, epoch)
        return self.train_selection.records

    def training_item_ids(
        self, *, mode: TrainingMode | str, epoch: int
    ) -> tuple[str, ...]:
        self._validate_training_request(mode, epoch)
        return self.train_selection.ordered_item_ids

    def held_out_records(self) -> tuple[CalcMawpsRecord, ...]:
        return self.held_out_selection.records

    @staticmethod
    def _validate_training_request(mode: str, epoch: int) -> None:
        if mode not in TRAINING_MODES:
            raise ValueError(f"unknown Stage 0 training mode {mode!r}")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError(f"epoch must be a positive integer, got {epoch!r}")


def build_stage0_dataset(
    train_records: Sequence[CalcMawpsRecord],
    validation_records: Sequence[CalcMawpsRecord],
) -> Stage0Dataset:
    """Build the fixed seed-0 train order and 128-item validation prefix."""
    _validate_record_splits(train_records, validation_records)
    return Stage0Dataset(
        train_selection=select_calc_mawps_records(
            {"train": train_records},
            split="train",
            seed=STAGE0_SEED,
            problem_count=None,
        ),
        held_out_selection=select_calc_mawps_records(
            {"validation": validation_records},
            split="validation",
            seed=STAGE0_SEED,
            problem_count=HELD_OUT_COUNT,
        ),
    )


def load_stage0_dataset(
    dataset_loader: Callable[..., Any] | None = None,
) -> Stage0Dataset:
    """Load only the pinned train and validation splits for a Stage 0 run."""
    train_records = load_calc_mawps_record_split("train", dataset_loader)
    validation_records = load_calc_mawps_record_split("validation", dataset_loader)
    return build_stage0_dataset(train_records, validation_records)


def training_examples(
    dataset: Stage0Dataset,
    *,
    mode: TrainingMode,
    epoch: int,
    problem_count: int | None = None,
) -> list[tuple[str, str]]:
    """Return one mode's persisted epoch order as trainer input pairs."""
    records = dataset.training_records(mode=mode, epoch=epoch)
    if problem_count is not None:
        if isinstance(problem_count, bool) or not isinstance(problem_count, int):
            raise TypeError("problem_count must be a non-boolean integer")
        if not 1 <= problem_count <= len(records):
            raise ValueError(f"problem_count must be from 1 through {len(records)}")
        records = records[:problem_count]
    return [(record.question, record.target) for record in records]


def _validate_record_splits(
    train_records: Sequence[CalcMawpsRecord],
    validation_records: Sequence[CalcMawpsRecord],
) -> None:
    if any(record.split != "train" for record in train_records):
        raise ValueError("train_records must contain only train records")
    if any(record.split != "validation" for record in validation_records):
        raise ValueError("validation_records must contain only validation records")
    train_ids = {record.id for record in train_records}
    validation_ids = {record.id for record in validation_records}
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(f"Calc-MAWPS train and validation ids overlap: {min(overlap)}")
