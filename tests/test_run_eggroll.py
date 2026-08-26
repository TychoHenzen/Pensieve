from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType, SimpleNamespace

from eval.stream.generators.asdiv_a import AsdivRecord
from train.stage0_data import plan_eggroll_fitness_batch


def test_main_constructs_and_runs_only_eggroll_trainer(monkeypatch) -> None:
    trainer_calls: list[dict[str, object]] = []
    trained_datasets: list[tuple[AsdivRecord, ...]] = []
    observed_batches: list[tuple[str, ...]] = []

    class FakeEggrollTrainer:
        def __init__(self, **kwargs: object) -> None:
            trainer_calls.append(kwargs)
            self.fitness_batch_size = kwargs["fitness_batch_size"]

        def trainable_param_count(self) -> int:
            return 7

        def train_epoch(self, dataset, on_step) -> SimpleNamespace:
            trained_datasets.append(dataset)
            cursor = 0
            optimizer_call_count = 0
            while cursor < len(dataset):
                batch = plan_eggroll_fitness_batch(
                    dataset,
                    cursor=cursor,
                    configured_batch_size=self.fitness_batch_size,
                    records_until_epoch_boundary=len(dataset) - cursor,
                    records_until_observation_boundary=len(dataset) - cursor,
                    records_until_logging_boundary=len(dataset) - cursor,
                )
                observed_batches.append(batch.ordered_item_ids)
                optimizer_call_count += 1
                on_step(
                    batch.next_position - 1,
                    len(dataset),
                    SimpleNamespace(
                        position=SimpleNamespace(
                            update_method="eggroll",
                            global_step=batch.next_position,
                            epoch=1,
                        ),
                        total_objective=1.5,
                        language_model_loss=1.25,
                        shared_variance=0.25,
                        consumed_record_count=batch.consumed_record_count,
                        next_example_position=batch.next_position,
                        optimizer_call_count=optimizer_call_count,
                    ),
                )
                cursor = batch.next_position
            return SimpleNamespace(
                avg_loss=1.5,
                avg_variance=0.25,
                min_variance=0.25,
                max_variance=0.25,
            )

    fake_trainer_module = ModuleType("train.eggroll_trainer")
    fake_trainer_module.DEFAULT_EVAL_BATCH_SIZE = 8
    fake_trainer_module.DEFAULT_FITNESS_BATCH_SIZE = 8
    fake_trainer_module.DEFAULT_LR = 0.001
    fake_trainer_module.DEFAULT_NUM_STEPS = 2
    fake_trainer_module.DEFAULT_POP_SIZE = 128
    fake_trainer_module.DEFAULT_RANK = 4
    fake_trainer_module.DEFAULT_SIGMA = 0.02
    fake_trainer_module.DEFAULT_VARIANCE_WEIGHT = 1.0
    fake_trainer_module.EggrollTrainer = FakeEggrollTrainer
    fake_trainer_module.StepResult = SimpleNamespace
    fake_trainer_module.validate_eggroll_config = lambda *_: None
    monkeypatch.setitem(sys.modules, "train.eggroll_trainer", fake_trainer_module)

    original_import = builtins.__import__

    def reject_alternating_scheduler(name, globals=None, locals=None, fromlist=(), level=0):
        if "alternating" in name.lower():
            raise AssertionError(f"Eggroll CLI imported alternating scheduler: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_alternating_scheduler)
    sys.modules.pop("train.run_eggroll", None)
    run_eggroll = importlib.import_module("train.run_eggroll")

    args = SimpleNamespace(
        epochs=1,
        slot_count=3,
        num_steps=4,
        pop_size=6,
        sigma=0.1,
        lr=0.2,
        rank=5,
        variance_weight=0.3,
        eval_batch_size=2,
        fitness_batch_size=8,
        use_amp=True,
        device="cpu",
        save_dir="unused",
        problem_count=None,
        log_every=1,
        resume=None,
    )
    monkeypatch.setattr(run_eggroll, "_parse_args", lambda: args)
    monkeypatch.setattr(
        run_eggroll,
        "_load_dataset_context",
        lambda _: (
            tuple(
                AsdivRecord(
                    id=f"record-{index}",
                    split="train",
                    question=f"q-{index}",
                    target=str(index),
                )
                for index in range(9)
            ),
            {},
            {},
        ),
    )
    monkeypatch.setattr(run_eggroll, "_save_checkpoint", lambda *_, **__: "unused.ckpt")
    monkeypatch.setattr(run_eggroll, "_log", lambda _: None)

    run_eggroll.main()

    assert trainer_calls == [
        {
            "slot_count": 3,
            "num_steps": 4,
            "pop_size": 6,
            "sigma": 0.1,
            "lr": 0.2,
            "rank": 5,
            "variance_weight": 0.3,
            "eval_batch_size": 2,
            "fitness_batch_size": 8,
            "use_amp": True,
            "device": "cpu",
        }
    ]
    assert [tuple(record.id for record in dataset) for dataset in trained_datasets] == [
        tuple(f"record-{index}" for index in range(9))
    ]
    assert observed_batches == [
        tuple(f"record-{index}" for index in range(8)),
        ("record-8",),
    ]


def test_standalone_progress_record_includes_batch_accounting(monkeypatch) -> None:
    sys.modules.pop("train.run_eggroll", None)
    run_eggroll = importlib.import_module("train.run_eggroll")
    result = SimpleNamespace(
        position=SimpleNamespace(update_method="eggroll", global_step=12, epoch=3),
        consumed_record_count=8,
        next_example_position=24,
        language_model_loss=1.25,
        total_objective=1.75,
        shared_variance=0.5,
        optimizer_call_count=3,
    )

    assert run_eggroll._training_progress_record(result) == {
        "update_method": "eggroll",
        "global_step": 12,
        "epoch": 3,
        "consumed_record_count": 8,
        "next_example_position": 24,
        "language_model_loss": 1.25,
        "total_objective": 1.75,
        "shared_variance": 0.5,
        "optimizer_call_count": 3,
    }
