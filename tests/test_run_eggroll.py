from __future__ import annotations

import builtins
import copy
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from eval.stream.generators.asdiv_a import AsdivRecord
from tests.test_stage0_checkpoint_resume import _metadata
from train.stage0_checkpoint import (
    ALLOWED_MODEL_PARAMETER_PATHS,
    EGGROLL_MODEL_PARAMETER_PATHS,
)
from train.stage0_data import plan_eggroll_fitness_batch
from train.standalone_checkpoint import load_checkpoint


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
                steps=optimizer_call_count,
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
        prompt_alignment_weight=0.4,
        eval_batch_size=2,
        fitness_batch_size=8,
        use_amp=True,
        device="cpu",
        save_dir="unused",
        problem_count=9,
        log_every=1,
        resume=None,
        stability_report=None,
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
            "prompt_alignment_weight": 0.4,
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


def test_standalone_resume_rejects_changed_fitness_batch_before_trainer_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("train.run_eggroll", None)
    run_eggroll = importlib.import_module("train.run_eggroll")
    args = SimpleNamespace(
        epochs=2,
        slot_count=16,
        num_steps=2,
        pop_size=128,
        sigma=0.001,
        lr=0.1,
        rank=4,
        variance_weight=1.0,
        prompt_alignment_weight=0.1,
        eval_batch_size=8,
        fitness_batch_size=8,
        use_amp=False,
        device="cpu",
        save_dir="unused",
        problem_count=3,
        log_every=1,
        resume="epoch-1.ckpt",
        stability_report=None,
    )
    current_config = run_eggroll._run_config(args, report_identity=None)
    saved_config = copy.deepcopy(current_config)
    saved_config["eggroll_population"]["fitness_batch_size"] = 4
    checkpoint = SimpleNamespace(
        metadata={
            "identity": {},
            "selections": {},
            "run_config": saved_config,
            "schedule": {"epoch": 1},
            "optimizer_manifests": [{"parameter_groups": [{"scalars": {"lr": 0.1}}]}],
        }
    )
    constructions: list[object] = []
    monkeypatch.setattr(run_eggroll, "configure_deterministic_runtime", lambda: None)
    monkeypatch.setattr(run_eggroll, "_parse_args", lambda: args)
    monkeypatch.setattr(
        run_eggroll,
        "_load_dataset_context",
        lambda _count: (("a", "b", "c"), {}, {}),
    )
    monkeypatch.setattr(
        run_eggroll,
        "load_checkpoint",
        lambda _path, *, expected_mode: checkpoint,
    )
    monkeypatch.setattr(
        run_eggroll,
        "EggrollTrainer",
        lambda **_kwargs: constructions.append(object()),
    )
    monkeypatch.setattr(run_eggroll, "_log", lambda _message: None)

    with pytest.raises(
        ValueError,
        match=r"\$\.run_config\.eggroll_population\.fitness_batch_size",
    ):
        run_eggroll.main()

    assert constructions == []


def test_public_command_writes_inspects_and_resumes_real_sgd_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sys.modules.pop("train.run_eggroll", None)
    run_eggroll = importlib.import_module("train.run_eggroll")
    fixture = _metadata()
    records = tuple(
        AsdivRecord(
            id=f"record-{index}",
            split="train",
            question=f"q-{index}",
            target=str(index),
        )
        for index in range(3)
    )
    visits: list[tuple[int, tuple[str, ...]]] = []
    trainers: list[object] = []

    class CheckpointTrainer:
        def __init__(self, **kwargs: object) -> None:
            parameters = {
                name: nn.Parameter(torch.tensor([float(index)]))
                for index, name in enumerate(ALLOWED_MODEL_PARAMETER_PATHS)
            }
            self.state = SimpleNamespace(trainable_params=parameters)
            self.optimizer = torch.optim.SGD(
                [parameters[name] for name in EGGROLL_MODEL_PARAMETER_PATHS],
                lr=float(kwargs["lr"]),
                momentum=0.0,
            )
            self.run_index = len(trainers)
            trainers.append(self)

        def trainable_param_count(self) -> int:
            return sum(parameter.numel() for parameter in self.state.trainable_params.values())

        def train_epoch(self, dataset, on_step) -> SimpleNamespace:
            visits.append((self.run_index, tuple(record.id for record in dataset)))
            with torch.no_grad():
                for name in EGGROLL_MODEL_PARAMETER_PATHS:
                    self.state.trainable_params[name].add_(1.0)
            on_step(
                len(dataset) - 1,
                len(dataset),
                SimpleNamespace(
                    position=SimpleNamespace(
                        update_method="eggroll",
                        global_step=len(dataset),
                        epoch=self.run_index + 1,
                    ),
                    total_objective=1.5,
                    language_model_loss=1.25,
                    shared_variance=0.25,
                    consumed_record_count=len(dataset),
                    next_example_position=len(dataset),
                    optimizer_call_count=self.run_index + 1,
                ),
            )
            return SimpleNamespace(
                avg_loss=1.5,
                avg_variance=0.25,
                min_variance=0.25,
                max_variance=0.25,
                steps=1,
            )

    args = SimpleNamespace(
        epochs=1,
        slot_count=16,
        num_steps=2,
        pop_size=128,
        sigma=0.001,
        lr=0.1,
        rank=4,
        variance_weight=1.0,
        prompt_alignment_weight=0.1,
        eval_batch_size=8,
        fitness_batch_size=8,
        use_amp=False,
        device="cpu",
        save_dir=str(tmp_path),
        problem_count=3,
        log_every=3,
        resume=None,
        stability_report=str(tmp_path / "stability.json"),
    )
    monkeypatch.setattr(run_eggroll, "configure_deterministic_runtime", lambda: None)
    monkeypatch.setattr(
        run_eggroll,
        "load_guarded_stability_report",
        lambda *_args, **_kwargs: SimpleNamespace(sha256="e" * 64),
    )
    monkeypatch.setattr(run_eggroll, "_parse_args", lambda: args)
    monkeypatch.setattr(
        run_eggroll,
        "_load_dataset_context",
        lambda _count: (records, fixture["identity"], fixture["selections"]),
    )
    monkeypatch.setattr(run_eggroll, "EggrollTrainer", CheckpointTrainer)
    monkeypatch.setattr(run_eggroll, "_log", lambda _message: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)

    run_eggroll.main()
    first_path = tmp_path / "epoch-1.ckpt"
    first = load_checkpoint(first_path, expected_mode="eggroll")

    assert first.metadata["optimizer_manifests"][0]["optimizer_type"] == "SGD"
    assert first.metadata["optimizer_manifests"][0]["parameter_groups"][0]["scalars"]["momentum"] == 0.0
    assert list(first.metadata["optimizer_manifests"][0]["parameter_names"]) == list(EGGROLL_MODEL_PARAMETER_PATHS)
    assert first.metadata["run_config"]["eggroll_population"]["fitness_batch_size"] == 8
    assert first.metadata["run_config"]["stability_report_identity"] == "e" * 64
    assert first.metadata["schedule"]["consumed_examples"] == 3
    assert first.metadata["schedule"]["eggroll_optimizer_calls"] == 1

    args.epochs = 2
    args.resume = str(first_path)
    run_eggroll.main()
    resumed = load_checkpoint(tmp_path / "epoch-2.ckpt", expected_mode="eggroll")

    assert visits == [
        (0, ("record-0", "record-1", "record-2")),
        (1, ("record-0", "record-1", "record-2")),
    ]
    assert resumed.metadata["schedule"]["consumed_examples"] == 6
    assert resumed.metadata["schedule"]["eggroll_optimizer_calls"] == 2
    for name in EGGROLL_MODEL_PARAMETER_PATHS:
        assert torch.equal(
            resumed.tensors[f"model.{name}"],
            first.tensors[f"model.{name}"] + 1.0,
        )
