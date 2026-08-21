from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import sys
from types import ModuleType, SimpleNamespace


def test_main_constructs_and_runs_only_gradient_trainer(monkeypatch) -> None:
    trainer_calls: list[dict[str, object]] = []
    trained_datasets: list[list[tuple[str, str]]] = []

    class FakeLatentCoreTrainer:
        def __init__(self, **kwargs: object) -> None:
            trainer_calls.append(kwargs)

        def trainable_param_count(self) -> int:
            return 7

        def train_epoch(self, dataset, on_step) -> SimpleNamespace:
            trained_datasets.append(dataset)
            on_step(
                0,
                len(dataset),
                SimpleNamespace(loss=1.5, variance=0.25),
            )
            return SimpleNamespace(
                avg_loss=1.5,
                avg_variance=0.25,
                min_variance=0.25,
                max_variance=0.25,
                avg_covariance=0.0,
                steps=len(dataset),
            )

    fake_trainer_module = ModuleType("train.trainer")
    fake_trainer_module.DEFAULT_LR = 0.001
    fake_trainer_module.DEFAULT_NUM_STEPS = 2
    fake_trainer_module.LatentCoreTrainer = FakeLatentCoreTrainer
    fake_trainer_module.StepResult = SimpleNamespace
    monkeypatch.setitem(sys.modules, "train.trainer", fake_trainer_module)

    original_import = builtins.__import__

    def reject_alternating_scheduler(name, globals=None, locals=None, fromlist=(), level=0):
        if "alternating" in name.lower():
            raise AssertionError(f"Gradient CLI imported alternating scheduler: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_alternating_scheduler)
    sys.modules.pop("train.run_training", None)
    run_training = importlib.import_module("train.run_training")

    args = SimpleNamespace(
        epochs=1,
        slot_count=3,
        num_steps=4,
        lr=0.2,
        device="cpu",
        save_dir="unused",
        problem_count=None,
        log_every=1,
        resume=None,
    )
    monkeypatch.setattr(run_training, "_parse_args", lambda: args)
    monkeypatch.setattr(
        run_training,
        "_load_dataset_context",
        lambda _: ([("q", "a")], {}, {}),
    )
    monkeypatch.setattr(run_training, "_save_checkpoint", lambda *_, **__: "unused.ckpt")
    monkeypatch.setattr(run_training, "_log", lambda _: None)

    run_training.main()

    assert trainer_calls == [
        {
            "slot_count": 3,
            "num_steps": 4,
            "lr": 0.2,
            "device": "cpu",
        }
    ]
    assert trained_datasets == [[("q", "a")]]


def test_gradient_cli_has_no_alternating_scheduler_import_or_call() -> None:
    import train.run_training as run_training

    tree = ast.parse(inspect.getsource(run_training))
    references = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call))
    ]

    assert "alternating" not in " ".join(ast.dump(node).lower() for node in references)
