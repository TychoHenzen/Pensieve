"""Contracts for the shared Stage 0 Calc-ASDiv_A selections."""

from __future__ import annotations

from eval.stage0_identity import (
    ASDIV_CONFIGURATION,
    ASDIV_DATASET,
    ASDIV_REVISION,
    ASDIV_SOURCE_COUNT,
)
from eval.stream.generators.asdiv_a import (
    AsdivRecord,
    select_asdiv_a_records,
)
from train.stage0_data import (
    build_stage0_dataset,
    load_stage0_dataset,
    plan_eggroll_fitness_batch,
)

TRAIN_COUNT = 570
VALIDATION_COUNT = 128
HELD_OUT_COUNT = 128


def _records(split: str, count: int) -> tuple[AsdivRecord, ...]:
    return tuple(
        AsdivRecord(
            id=f"asdiv_a__{split}_{index}",
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

    expected = select_asdiv_a_records({"train": train_records}, split="train", seed=0, problem_count=None)

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

    expected_validation = select_asdiv_a_records(
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


# covers: train/stage0-training :: Shared Stage 0 dataset contract :: Eggroll aggregates a deterministic fitness batch
def test_eggroll_fitness_batch_preserves_order_and_averages_candidate_fitnesses() -> None:
    train_records = _records("train", TRAIN_COUNT)
    stage0_data = build_stage0_dataset(
        train_records=train_records,
        validation_records=_records("validation", VALIDATION_COUNT),
    )
    records = stage0_data.training_records(mode="eggroll", epoch=1)

    batch = plan_eggroll_fitness_batch(
        records,
        cursor=11,
        configured_batch_size=8,
        records_until_epoch_boundary=len(records) - 11,
        records_until_observation_boundary=20,
        records_until_logging_boundary=10,
    )

    assert batch.ordered_item_ids == stage0_data.training_item_ids(mode="eggroll", epoch=1)[11:19]
    assert batch.start_position == 11
    assert batch.next_position == 19
    assert batch.consumed_record_count == 8
    assert batch.mean_candidate_fitnesses(
        (
            (1.0, -2.0, 5.0),
            (2.0, 4.0, 8.0),
            (3.0, 10.0, -1.0),
            (4.0, 0.0, 4.0),
            (5.0, 2.0, 7.0),
            (6.0, 6.0, 2.0),
            (7.0, 8.0, 6.0),
            (8.0, 12.0, 1.0),
        )
    ) == (4.5, 5.0, 4.0)


# covers: train/stage0-training :: Shared Stage 0 dataset contract :: Eggroll caps a boundary batch
def test_eggroll_fitness_batch_caps_each_boundary_and_keeps_final_partial_batch() -> None:
    records = _records("train", 13)

    epoch_batch = plan_eggroll_fitness_batch(
        records,
        cursor=2,
        configured_batch_size=8,
        records_until_epoch_boundary=3,
        records_until_observation_boundary=8,
        records_until_logging_boundary=8,
    )
    observation_batch = plan_eggroll_fitness_batch(
        records,
        cursor=5,
        configured_batch_size=8,
        records_until_epoch_boundary=8,
        records_until_observation_boundary=2,
        records_until_logging_boundary=8,
    )
    logging_batch = plan_eggroll_fitness_batch(
        records,
        cursor=8,
        configured_batch_size=8,
        records_until_epoch_boundary=5,
        records_until_observation_boundary=5,
        records_until_logging_boundary=1,
    )
    final_batch = plan_eggroll_fitness_batch(
        records,
        cursor=11,
        configured_batch_size=8,
        records_until_epoch_boundary=2,
        records_until_observation_boundary=2,
        records_until_logging_boundary=2,
    )

    assert epoch_batch.ordered_item_ids == tuple(record.id for record in records[2:5])
    assert observation_batch.ordered_item_ids == tuple(record.id for record in records[5:7])
    assert logging_batch.ordered_item_ids == (records[8].id,)
    assert final_batch.ordered_item_ids == tuple(record.id for record in records[11:13])
    assert final_batch.next_position == len(records)


def test_runtime_loader_reads_the_pinned_source_once() -> None:
    loaded_splits: list[str] = []

    def dataset_loader(
        dataset: str,
        configuration: str,
        *,
        split: str,
        revision: str,
        trust_remote_code: bool,
    ) -> list[dict[str, object]]:
        assert dataset == ASDIV_DATASET
        assert configuration == ASDIV_CONFIGURATION
        assert revision == ASDIV_REVISION
        assert trust_remote_code is False
        assert split == "test"
        loaded_splits.append(split)
        return [
            {
                "id": f"asdiv_a__source_{index}",
                "question": f"source question {index}",
                "result": str(index + 1),
                "result_float": float(index + 1),
            }
            for index in range(ASDIV_SOURCE_COUNT)
        ]

    stage0_data = load_stage0_dataset(dataset_loader)

    assert loaded_splits == ["test"]
    assert len(stage0_data.train_selection.records) == TRAIN_COUNT
    assert len(stage0_data.held_out_records()) == HELD_OUT_COUNT
