from __future__ import annotations

import json
from typing import Any

from train import benchmark_eggroll


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


def test_main_writes_one_json_object(monkeypatch: Any, capsys: Any) -> None:
    expected = {"schema_version": 1, "speedup_ratio": 3.5}
    monkeypatch.setattr(benchmark_eggroll, "run_cuda_benchmark", lambda: expected)

    benchmark_eggroll.main()

    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert json.loads(output) == expected
