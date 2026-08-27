from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pytest
import torch

from tests.eggroll_reference import EggrollStepSnapshot
from train import benchmark_eggroll
from train.training_results import ExperimentPosition, StepResult


class _FakeCuda:
    def __init__(self, events: list[tuple[str, Any]]) -> None:
        self.events = events
        self.active_path = ""

    def reset_peak_memory_stats(self, device: Any = None) -> None:
        self.events.append(("reset", device))

    def synchronize(self, device: Any = None) -> None:
        self.events.append(("synchronize", device))

    def max_memory_allocated(self, device: Any = None) -> int:
        self.events.append(("peak", device))
        return 400 if self.active_path == "reference" else 300


# covers: train/eggroll-execution::CUDA benchmark proves a material speed improvement::Performance gate passes
def test_benchmark_protocol_restores_and_alternates_complete_steps() -> None:
    events: list[tuple[str, Any]] = []
    cuda = _FakeCuda(events)
    clock_values = iter(float(value) for value in range(20))

    def restore() -> None:
        events.append(("restore", "state"))

    def run_step(path: benchmark_eggroll.BenchmarkPath) -> None:
        cuda.active_path = path
        events.append(("run", path))

    results = benchmark_eggroll.run_benchmark_protocol(
        run_step=run_step,
        restore_state=restore,
        consumed_examples_per_step=8,
        device="cuda:0",
        cuda=cuda,
        clock=lambda: next(clock_values),
    )

    runs = [value for event, value in events if event == "run"]
    assert runs[:4] == ["reference", "reference", "optimized", "optimized"]
    assert runs[4:] == ["reference", "optimized"] * 5
    assert sum(event == "restore" for event, _ in events) == 14
    assert sum(event == "reset" for event, _ in events) == 14
    assert sum(event == "synchronize" for event, _ in events) == 28
    assert results["reference"].durations_seconds == (1.0,) * 5
    assert results["optimized"].durations_seconds == (1.0,) * 5
    assert results["reference"].median_seconds == 1.0
    assert results["optimized"].median_seconds == 1.0
    assert results["reference"].peak_allocated_bytes == 400
    assert results["optimized"].peak_allocated_bytes == 300
    assert results["reference"].consumed_examples_per_step == 8
    assert results["optimized"].consumed_examples_per_step == 8
    assert results["reference"].durations_seconds_per_consumed_example == (
        0.125,
    ) * 5
    assert results["optimized"].median_seconds_per_consumed_example == 0.125
    assert results["reference"].peak_allocated_bytes_per_consumed_example == 50.0
    assert results["optimized"].peak_allocated_bytes_per_consumed_example == 37.5


def test_benchmark_protocol_rejects_invalid_consumed_example_count() -> None:
    with pytest.raises(
        ValueError,
        match="consumed_examples_per_step must be positive",
    ):
        benchmark_eggroll.run_benchmark_protocol(
            run_step=lambda _: None,
            restore_state=lambda: None,
            consumed_examples_per_step=0,
            device="cuda:0",
        )


def test_performance_gate_rejects_insufficient_speedup() -> None:
    measurements = {
        "reference": benchmark_eggroll.PathMeasurements((2.9,) * 5, 2.9, 400),
        "optimized": benchmark_eggroll.PathMeasurements((1.0,) * 5, 1.0, 300),
    }

    with pytest.raises(
        benchmark_eggroll.BenchmarkPerformanceError,
        match=re.escape("speedup_ratio=2.900000"),
    ):
        benchmark_eggroll._performance_gate(measurements)


def test_performance_gate_rejects_peak_memory_increase() -> None:
    measurements = {
        "reference": benchmark_eggroll.PathMeasurements((3.0,) * 5, 3.0, 300),
        "optimized": benchmark_eggroll.PathMeasurements((1.0,) * 5, 1.0, 301),
    }

    with pytest.raises(
        benchmark_eggroll.BenchmarkPerformanceError,
        match="optimized_peak_allocated_bytes=301",
    ):
        benchmark_eggroll._performance_gate(measurements)


# covers: train/eggroll-execution::CUDA benchmark proves a material speed improvement::Performance gate passes
def test_performance_gate_passes_at_three_x_speedup_and_no_memory_growth() -> None:
    measurements = {
        "reference": benchmark_eggroll.PathMeasurements((3.0,) * 5, 3.0, 400),
        "optimized": benchmark_eggroll.PathMeasurements((1.0,) * 5, 1.0, 400),
    }

    gate = benchmark_eggroll._performance_gate(measurements)

    assert gate == {
        "passed": True,
        "minimum_speedup_ratio": 3.0,
        "speed_passed": True,
        "memory_passed": True,
    }


def test_main_writes_one_json_object(monkeypatch: Any, capsys: Any) -> None:
    expected = {"schema_version": 1, "speedup_ratio": 3.5}
    monkeypatch.setattr(benchmark_eggroll, "run_cuda_benchmark", lambda: expected)

    benchmark_eggroll.main()

    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert json.loads(output) == expected


def _outcome(parameter_value: float) -> benchmark_eggroll.CompleteStepOutcome:
    result = StepResult(
        position=ExperimentPosition(
            update_method="eggroll",
            cycle=0,
            global_step=0,
            epoch=0,
            example_position=0,
            phase_step=0,
        ),
        language_model_loss=1.0,
        total_objective=2.0,
        regularizer_loss=1.0,
        shared_variance=0.5,
    )
    state = EggrollStepSnapshot(
        parameter_values=(torch.tensor([parameter_value]),),
        optimizer_state={
            "state": {0: {"exp_avg": torch.tensor([0.25])}},
            "param_groups": [{"params": [0], "lr": 1e-3}],
        },
        workspace_slots=torch.tensor([[0.75]]),
        python_rng_state=(3, (1, 2, 3), None),
        numpy_rng_state=(
            "MT19937",
            np.array([1, 2, 3], dtype=np.uint32),
            2,
            0,
            0.0,
        ),
        torch_cpu_rng_state=torch.tensor([1, 2, 3], dtype=torch.uint8),
        torch_cuda_rng_states=(torch.tensor([4, 5], dtype=torch.uint8),),
    )
    return benchmark_eggroll.CompleteStepOutcome(result=result, state=state)


# covers: train/eggroll-execution::CUDA benchmark proves a material speed improvement::Equivalence failure blocks the benchmark result
def test_equivalence_failure_exits_without_performance_json(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    reference = _outcome(1.0)
    optimized = _outcome(1.1)

    def fail_equivalence() -> dict[str, Any]:
        benchmark_eggroll.assert_complete_step_equivalence(reference, optimized)
        raise AssertionError("unreachable")

    monkeypatch.setattr(
        benchmark_eggroll,
        "run_cuda_benchmark",
        fail_equivalence,
    )

    with pytest.raises(SystemExit) as raised:
        benchmark_eggroll.main()

    captured = capsys.readouterr()
    assert raised.value.code == 1
    assert captured.out == ""
    assert "numerical equivalence failed at parameters[0]" in captured.err


# covers: train/eggroll-execution::CUDA benchmark proves a material speed improvement::CUDA is unavailable
def test_cuda_rejection_precedes_asset_loading(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    asset_loads: list[str] = []
    monkeypatch.setattr(benchmark_eggroll.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        benchmark_eggroll,
        "load_stage0_dataset",
        lambda: asset_loads.append("dataset"),
    )
    monkeypatch.setattr(
        benchmark_eggroll,
        "EggrollTrainer",
        lambda **_: asset_loads.append("model"),
    )

    with pytest.raises(SystemExit) as raised:
        benchmark_eggroll.main()

    captured = capsys.readouterr()
    assert raised.value.code == 1
    assert captured.out == ""
    assert "CUDA benchmark was not run" in captured.err
    assert asset_loads == []
