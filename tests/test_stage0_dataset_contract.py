"""Contracts for the shared Stage 0 Calc-MAWPS selections."""

from __future__ import annotations

from eval.stream.generators.calc_mawps import (
    CalcMawpsRecord,
    select_calc_mawps_records,
)
from train.stage0_data import build_stage0_dataset


TRAIN_COUNT = 1_089
VALIDATION_COUNT = 1_039
HELD_OUT_COUNT = 128


def _records(split: str, count: int) -> tuple[CalcMawpsRecord, ...]:
    return tuple(
        CalcMawpsRecord(
            id=f"mawps__{split}_{index}",
            split=split,
            question=f"{split} question {index}",
            target=str(index + 1),
        )
        for index in range(count)
    )


# covers: train/stage0-training :: Shared Stage 0 dataset contract :: Training modes share examples
def test_training_modes_replay_the_complete_seed_zero_train_order_each_epoch() -> None:
    train_records = _records("train", TRAIN_COUNT)
    stage0_data = build_stage0_dataset(
        train_records=train_records,
        validation_records=_records("validation", VALIDATION_COUNT),
    )

    expected = select_calc_mawps_records(
        {"train": train_records}, split="train", seed=0, problem_count=None
    )

    assert stage0_data.train_selection == expected
    for mode in ("gradient", "eggroll", "alternating"):
        for epoch in range(1, 4):
            assert stage0_data.training_records(mode=mode, epoch=epoch) == expected.records
            assert stage0_data.training_item_ids(mode=mode, epoch=epoch) == expected.ordered_item_ids


# covers: train/stage0-training :: Shared Stage 0 dataset contract :: Held-out split isolation
def test_held_out_selection_persists_only_the_seed_zero_validation_prefix() -> None:
    train_records = _records("train", TRAIN_COUNT)
    validation_records = _records("validation", VALIDATION_COUNT)
    stage0_data = build_stage0_dataset(
        train_records=train_records,
        validation_records=validation_records,
    )

    expected_validation = select_calc_mawps_records(
        {"validation": validation_records},
        split="validation",
        seed=0,
        problem_count=None,
    )

    assert stage0_data.held_out_selection.split == "validation"
    assert stage0_data.held_out_selection.seed == 0
    assert stage0_data.held_out_selection.problem_count == HELD_OUT_COUNT
    assert stage0_data.held_out_selection.ordered_item_ids == expected_validation.ordered_item_ids[:HELD_OUT_COUNT]
    assert stage0_data.held_out_records() == expected_validation.records[:HELD_OUT_COUNT]
    assert all(record.split == "validation" for record in stage0_data.held_out_records())
