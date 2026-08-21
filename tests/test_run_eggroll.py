from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType, SimpleNamespace


def test_main_constructs_and_runs_only_eggroll_trainer(monkeypatch) -> None:
    trainer_calls: list[dict[str, object]] = []
    trained_datasets: list[list[tuple[str, str]]] = []

    class FakeEggrollTrainer:
        def __init__(self, **kwargs: object) -> None:
            trainer_calls.append(kwargs)

        def trainable_param_count(self) -> int:
            return 7

        def train_epoch(self, dataset, on_step) -> SimpleNamespace:
            trained_datasets.append(dataset)
            on_step(
                0,
                len(dataset),
                SimpleNamespace(loss=1.5, variance=0.25, best_fitness=-1.0),
            )
            return SimpleNamespace(
                avg_loss=1.5,
                avg_variance=0.25,
                min_variance=0.25,
                max_variance=0.25,
            )

    fake_trainer_module = ModuleType("train.eggroll_trainer")
    fake_trainer_module.DEFAULT_EVAL_BATCH_SIZE = 8
    fake_trainer_module.DEFAULT_LR = 0.001
    fake_trainer_module.DEFAULT_NUM_STEPS = 2
    fake_trainer_module.DEFAULT_POP_SIZE = 128
    fake_trainer_module.DEFAULT_RANK = 4
    fake_trainer_module.DEFAULT_SIGMA = 0.02
    fake_trainer_module.DEFAULT_VARIANCE_WEIGHT = 1.0
    fake_trainer_module.EggrollTrainer = FakeEggrollTrainer
    fake_trainer_module.StepResult = SimpleNamespace
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
        lambda _: ([("q", "a")], {}, {}),
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
            "use_amp": True,
            "device": "cpu",
        }
    ]
    assert trained_datasets == [[("q", "a")]]
