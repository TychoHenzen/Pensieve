"""CUDA benchmark for materialized and optimized EGGROLL execution."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

import torch

from eval.stage0_identity import STAGE0_IDENTITY
from tests.eggroll_reference import (
    apply_reference_pair_loop_update,
    capture_eggroll_step_snapshot,
    materialized_linear,
    restore_eggroll_step_snapshot,
)
from train import eggroll_trainer as trainer_module
from train.eggroll_trainer import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_NUM_STEPS,
    DEFAULT_POP_SIZE,
    DEFAULT_RANK,
    DEFAULT_SIGMA,
    DEFAULT_VARIANCE_WEIGHT,
    EggrollTrainer,
)
from train.stage0_data import load_stage0_dataset
from train.standalone_checkpoint import (
    configure_deterministic_runtime,
    runtime_identity,
)
from workspace.concept_slots import DEFAULT_SLOT_COUNT

BenchmarkPath = Literal["reference", "optimized"]
PATHS: tuple[BenchmarkPath, BenchmarkPath] = ("reference", "optimized")
WARMUP_RUNS_PER_PATH = 2
MEASURED_RUNS_PER_PATH = 5


class CudaMeasurements(Protocol):
    """The CUDA timing surface used by the benchmark protocol."""

    def synchronize(self, device: torch.device | str | int | None = None) -> None: ...

    def reset_peak_memory_stats(
        self, device: torch.device | str | int | None = None
    ) -> None: ...

    def max_memory_allocated(
        self, device: torch.device | str | int | None = None
    ) -> int: ...


@dataclass(frozen=True)
class PathMeasurements:
    durations_seconds: tuple[float, ...]
    median_seconds: float
    peak_allocated_bytes: int


def _measurement_order() -> tuple[BenchmarkPath, ...]:
    return tuple(
        path
        for _ in range(MEASURED_RUNS_PER_PATH)
        for path in PATHS
    )


def run_benchmark_protocol(
    *,
    run_step: Callable[[BenchmarkPath], Any],
    restore_state: Callable[[], None],
    device: torch.device | str | int,
    cuda: CudaMeasurements = torch.cuda,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[BenchmarkPath, PathMeasurements]:
    """Run the pinned warm-up and alternating measurement schedule."""
    durations: dict[BenchmarkPath, list[float]] = {
        "reference": [],
        "optimized": [],
    }
    peaks: dict[BenchmarkPath, list[int]] = {
        "reference": [],
        "optimized": [],
    }

    for path in PATHS:
        for _ in range(WARMUP_RUNS_PER_PATH):
            restore_state()
            cuda.reset_peak_memory_stats(device)
            cuda.synchronize(device)
            run_step(path)
            cuda.synchronize(device)

    for path in _measurement_order():
        restore_state()
        cuda.reset_peak_memory_stats(device)
        cuda.synchronize(device)
        started = clock()
        run_step(path)
        cuda.synchronize(device)
        elapsed = clock() - started
        durations[path].append(elapsed)
        peaks[path].append(int(cuda.max_memory_allocated(device)))

    return {
        path: PathMeasurements(
            durations_seconds=tuple(durations[path]),
            median_seconds=statistics.median(durations[path]),
            peak_allocated_bytes=max(peaks[path]),
        )
        for path in PATHS
    }


@contextmanager
def _execution_path(path: BenchmarkPath) -> Any:
    """Select the benchmark-only materialized path for one complete step."""
    if path == "optimized":
        yield
        return
    if path != "reference":
        raise ValueError(f"unknown benchmark path {path!r}")

    original_linear = trainer_module.factorized_linear
    original_update = trainer_module.apply_factorized_update
    trainer_module.factorized_linear = materialized_linear
    trainer_module.apply_factorized_update = apply_reference_pair_loop_update
    try:
        yield
    finally:
        trainer_module.factorized_linear = original_linear
        trainer_module.apply_factorized_update = original_update


def _device_identity(device: torch.device) -> dict[str, Any]:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    major, minor = torch.cuda.get_device_capability(index)
    return {
        "type": "cuda",
        "index": index,
        "name": properties.name,
        "compute_capability": f"{major}.{minor}",
        "total_memory_bytes": int(properties.total_memory),
    }


def _example_identity(record: Any, selection_identity: str) -> dict[str, Any]:
    return {
        "selection_identity": selection_identity,
        "selection_index": 0,
        "item_id": record.id,
        "split": record.split,
        "question_sha256": hashlib.sha256(record.question.encode("utf-8")).hexdigest(),
        "target_sha256": hashlib.sha256(record.target.encode("utf-8")).hexdigest(),
    }


def _benchmark_config(device: torch.device) -> dict[str, Any]:
    return {
        "device": str(device),
        "dtype": "float32",
        "slot_count": DEFAULT_SLOT_COUNT,
        "num_steps": DEFAULT_NUM_STEPS,
        "pop_size": DEFAULT_POP_SIZE,
        "antithetic_pair_count": DEFAULT_POP_SIZE // 2,
        "eval_batch_size": DEFAULT_EVAL_BATCH_SIZE,
        "sigma": DEFAULT_SIGMA,
        "rank": DEFAULT_RANK,
        "learning_rate": DEFAULT_LR,
        "variance_weight": DEFAULT_VARIANCE_WEIGHT,
        "optimizer": "Adam",
        "use_amp": False,
        "warmup_runs_per_path": WARMUP_RUNS_PER_PATH,
        "measured_runs_per_path": MEASURED_RUNS_PER_PATH,
        "measurement_order": list(_measurement_order()),
    }


def run_cuda_benchmark() -> dict[str, Any]:
    """Load the pinned assets once and benchmark one persisted example."""
    configure_deterministic_runtime()
    device = torch.device("cuda:0")

    dataset = load_stage0_dataset()
    record = dataset.training_records(mode="eggroll", epoch=1)[0]
    trainer = EggrollTrainer(
        slot_count=DEFAULT_SLOT_COUNT,
        num_steps=DEFAULT_NUM_STEPS,
        pop_size=DEFAULT_POP_SIZE,
        sigma=DEFAULT_SIGMA,
        lr=DEFAULT_LR,
        rank=DEFAULT_RANK,
        variance_weight=DEFAULT_VARIANCE_WEIGHT,
        eval_batch_size=DEFAULT_EVAL_BATCH_SIZE,
        use_amp=False,
        device=str(device),
    )
    if any(parameter.dtype != torch.float32 for parameter in trainer.trainable_params):
        raise RuntimeError("benchmark trainable parameters must use float32")

    initial = capture_eggroll_step_snapshot(
        trainer.trainable_params,
        trainer.optimizer,
        trainer.workspace,
    )

    def restore_state() -> None:
        restore_eggroll_step_snapshot(
            initial,
            trainer.trainable_params,
            trainer.optimizer,
            trainer.workspace,
        )
        for parameter in trainer.trainable_params:
            parameter.grad = None

    def run_step(path: BenchmarkPath) -> Any:
        with _execution_path(path):
            return trainer.train_step(record.question, record.target)

    measurements = run_benchmark_protocol(
        run_step=run_step,
        restore_state=restore_state,
        device=device,
    )
    reference_median = measurements["reference"].median_seconds
    optimized_median = measurements["optimized"].median_seconds
    path_results = {
        path: asdict(measurements[path])
        for path in PATHS
    }
    return {
        "schema_version": 1,
        "device": _device_identity(device),
        "software": runtime_identity(),
        "assets": dict(STAGE0_IDENTITY),
        "example": _example_identity(record, dataset.train_selection.identity),
        "config": _benchmark_config(device),
        "paths": path_results,
        "speedup_ratio": reference_median / optimized_median,
    }


def main() -> None:
    result: Mapping[str, Any] = run_cuda_benchmark()
    sys.stdout.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
