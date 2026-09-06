"""Shared deterministic Calc-ASDiv_A selections for Stage 0 training."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from eval.stream.generators.asdiv_a import (
    AsdivRecord,
    AsdivSelection,
    load_asdiv_records,
    select_asdiv_a_records,
)

STAGE0_SEED = 0
HELD_OUT_COUNT = 128
TRAINING_MODES = frozenset({"gradient", "eggroll", "alternating"})
TrainingMode = Literal["gradient", "eggroll", "alternating"]


@dataclass(frozen=True)
class FitnessBatch:
    """One ordered EGGROLL update batch and its next dataset cursor."""

    records: tuple[AsdivRecord, ...]
    start_position: int
    next_position: int

    @property
    def ordered_item_ids(self) -> tuple[str, ...]:
        return tuple(record.id for record in self.records)

    @property
    def consumed_record_count(self) -> int:
        return len(self.records)

    def mean_candidate_fitnesses(
        self,
        per_record_fitnesses: Sequence[Sequence[float]],
    ) -> tuple[float, ...]:
        """Return each candidate's arithmetic mean in the batch record order."""
        if len(per_record_fitnesses) != self.consumed_record_count:
            raise ValueError("per-record fitness count must match the fitness batch")
        if not per_record_fitnesses:
            raise ValueError("fitness batches must contain at least one record")

        candidate_count = len(per_record_fitnesses[0])
        if candidate_count < 1:
            raise ValueError("each record must provide at least one candidate fitness")
        if any(len(fitnesses) != candidate_count for fitnesses in per_record_fitnesses):
            raise ValueError("each record must provide fitness for the same candidates")

        return tuple(
            sum(fitnesses[candidate_index] for fitnesses in per_record_fitnesses) / self.consumed_record_count
            for candidate_index in range(candidate_count)
        )


def plan_eggroll_fitness_batch(
    records: Sequence[AsdivRecord],
    *,
    cursor: int,
    configured_batch_size: int,
    records_until_epoch_boundary: int,
    records_until_observation_boundary: int,
    records_until_logging_boundary: int,
) -> FitnessBatch:
    """Plan one positive EGGROLL batch without crossing an active boundary."""
    _validate_positive_int("configured_batch_size", configured_batch_size)
    _validate_positive_int("records_until_epoch_boundary", records_until_epoch_boundary)
    _validate_positive_int("records_until_observation_boundary", records_until_observation_boundary)
    _validate_positive_int("records_until_logging_boundary", records_until_logging_boundary)
    if isinstance(cursor, bool) or not isinstance(cursor, int):
        raise TypeError("cursor must be a non-boolean integer")
    if not 0 <= cursor < len(records):
        raise ValueError(f"cursor must identify an unconsumed record, got {cursor}")

    record_count = min(
        configured_batch_size,
        len(records) - cursor,
        records_until_epoch_boundary,
        records_until_observation_boundary,
        records_until_logging_boundary,
    )
    return FitnessBatch(
        records=tuple(records[cursor : cursor + record_count]),
        start_position=cursor,
        next_position=cursor + record_count,
    )


@dataclass(frozen=True)
class Stage0Dataset:
    """The persisted train order and validation-only held-out selection."""

    train_selection: AsdivSelection
    held_out_selection: AsdivSelection

    def training_records(self, *, mode: TrainingMode | str, epoch: int) -> tuple[AsdivRecord, ...]:
        self._validate_training_request(mode, epoch)
        return self.train_selection.records

    def training_item_ids(self, *, mode: TrainingMode | str, epoch: int) -> tuple[str, ...]:
        self._validate_training_request(mode, epoch)
        return self.train_selection.ordered_item_ids

    def held_out_records(self) -> tuple[AsdivRecord, ...]:
        return self.held_out_selection.records

    @staticmethod
    def _validate_training_request(mode: str, epoch: int) -> None:
        if mode not in TRAINING_MODES:
            raise ValueError(f"unknown Stage 0 training mode {mode!r}")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError(f"epoch must be a positive integer, got {epoch!r}")


def build_stage0_dataset(
    train_records: Sequence[AsdivRecord],
    validation_records: Sequence[AsdivRecord],
) -> Stage0Dataset:
    """Build the fixed seed-0 train order and 128-item validation prefix."""
    _validate_record_splits(train_records, validation_records)
    return Stage0Dataset(
        train_selection=select_asdiv_a_records(
            {"train": train_records},
            split="train",
            seed=STAGE0_SEED,
            problem_count=None,
        ),
        held_out_selection=select_asdiv_a_records(
            {"validation": validation_records},
            split="validation",
            seed=STAGE0_SEED,
            problem_count=HELD_OUT_COUNT,
        ),
    )


def load_stage0_dataset(
    dataset_loader: Callable[..., Any] | None = None,
) -> Stage0Dataset:
    """Load the pinned source once and build its disjoint Stage 0 partitions."""
    records = load_asdiv_records(dataset_loader=dataset_loader)
    return build_stage0_dataset(records["train"], records["validation"])


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
    train_records: Sequence[AsdivRecord],
    validation_records: Sequence[AsdivRecord],
) -> None:
    if any(record.split != "train" for record in train_records):
        raise ValueError("train_records must contain only train records")
    if any(record.split != "validation" for record in validation_records):
        raise ValueError("validation_records must contain only validation records")
    train_ids = {record.id for record in train_records}
    validation_ids = {record.id for record in validation_records}
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(f"Calc-ASDiv_A train and validation ids overlap: {min(overlap)}")


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
