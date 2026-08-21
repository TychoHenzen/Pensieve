"""Contracts for the shared Stage 0 Calc-MAWPS selections."""

from __future__ import annotations

from eval.stream.generators.calc_mawps import (
    CalcMawpsRecord,
    select_calc_mawps_records,
)
from eval.stage0_identity import (
    CALC_MAWPS_CONFIGURATION,
    CALC_MAWPS_DATASET,
    CALC_MAWPS_RAW_COUNTS,
    CALC_MAWPS_REVISION,
    CALC_MAWPS_VALIDATION_EXCLUDED_ID,
)
from train.stage0_data import build_stage0_dataset, load_stage0_dataset


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


def test_runtime_loader_uses_pinned_train_and_validation_without_test() -> None:
    loaded_splits: list[str] = []

    def dataset_loader(
        dataset: str,
        configuration: str,
        *,
        split: str,
        revision: str,
        trust_remote_code: bool,
    ) -> list[dict[str, object]]:
        assert dataset == CALC_MAWPS_DATASET
        assert configuration == CALC_MAWPS_CONFIGURATION
        assert revision == CALC_MAWPS_REVISION
        assert trust_remote_code is False
        assert split != "test"
        loaded_splits.append(split)
        rows = [
            {
                "id": f"mawps__{split}_{index}",
                "question": f"{split} question {index}",
                "result": str(index + 1),
                "result_float": float(index + 1),
            }
            for index in range(CALC_MAWPS_RAW_COUNTS[split])
        ]
        if split == "validation":
            rows[-1]["id"] = CALC_MAWPS_VALIDATION_EXCLUDED_ID
        return rows

    stage0_data = load_stage0_dataset(dataset_loader)

    assert loaded_splits == ["train", "validation"]
    assert len(stage0_data.train_selection.records) == TRAIN_COUNT
    assert len(stage0_data.held_out_records()) == HELD_OUT_COUNT
