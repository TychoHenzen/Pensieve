"""CUDA benchmark for materialized and optimized EGGROLL execution."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Literal, Protocol

import numpy as np
import torch

from eval.stage0_identity import STAGE0_IDENTITY
from tests.eggroll_reference import (
    EggrollStepSnapshot,
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
EQUIVALENCE_RTOL = 1e-4
EQUIVALENCE_ATOL = 1e-4


class BenchmarkError(RuntimeError):
    """A benchmark precondition or correctness check failed."""


class CudaBenchmarkUnavailable(BenchmarkError):
    """The pinned CUDA benchmark device cannot be used."""


class BenchmarkEquivalenceError(BenchmarkError):
    """Reference and optimized complete-step results differ."""


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


@dataclass(frozen=True)
class CompleteStepOutcome:
    result: Any
    state: EggrollStepSnapshot


def _measurement_order() -> tuple[BenchmarkPath, ...]:
    return tuple(
        path
        for _ in range(MEASURED_RUNS_PER_PATH)
        for path in PATHS
    )


def _equivalence_failure(path: str, detail: str) -> None:
    raise BenchmarkEquivalenceError(
        f"numerical equivalence failed at {path}: {detail}"
    )


def _assert_close(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            _equivalence_failure(path, "value is not a tensor")
        try:
            torch.testing.assert_close(
                actual,
                expected,
                rtol=EQUIVALENCE_RTOL,
                atol=EQUIVALENCE_ATOL,
            )
        except AssertionError as error:
            _equivalence_failure(path, str(error))
        return
    if isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not math.isclose(
            float(actual),
            expected,
            rel_tol=EQUIVALENCE_RTOL,
            abs_tol=EQUIVALENCE_ATOL,
        ):
            _equivalence_failure(path, f"expected {expected!r}, actual {actual!r}")
        return
    if is_dataclass(expected) and not isinstance(expected, type):
        if not isinstance(actual, type(expected)):
            _equivalence_failure(path, f"expected {type(expected).__name__}")
        for field in fields(expected):
            _assert_close(
                getattr(actual, field.name),
                getattr(expected, field.name),
                f"{path}.{field.name}",
            )
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or actual.keys() != expected.keys():
            _equivalence_failure(path, "mapping keys differ")
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            _equivalence_failure(path, "sequence shape differs")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_close(actual_item, expected_item, f"{path}[{index}]")
        return
    if actual != expected:
        _equivalence_failure(path, f"expected {expected!r}, actual {actual!r}")


def _assert_exact(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual, expected):
            _equivalence_failure(path, "tensor values differ")
        return
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not np.array_equal(actual, expected):
            _equivalence_failure(path, "array values differ")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            _equivalence_failure(path, "sequence shape differs")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_exact(actual_item, expected_item, f"{path}[{index}]")
        return
    if actual != expected:
        _equivalence_failure(path, f"expected {expected!r}, actual {actual!r}")


def assert_complete_step_equivalence(
    reference: CompleteStepOutcome,
    optimized: CompleteStepOutcome,
) -> None:
    """Require equivalent complete-step outputs and final mutable state."""
    _assert_close(optimized.result, reference.result, "step_result")
    _assert_close(
        optimized.state.parameter_values,
        reference.state.parameter_values,
        "parameters",
    )
    _assert_close(
        optimized.state.optimizer_state,
        reference.state.optimizer_state,
        "optimizer",
    )
    _assert_close(
        optimized.state.workspace_slots,
        reference.state.workspace_slots,
        "workspace",
    )
    _assert_exact(
        optimized.state.python_rng_state,
        reference.state.python_rng_state,
        "rng.python",
    )
    _assert_exact(
        optimized.state.numpy_rng_state,
        reference.state.numpy_rng_state,
        "rng.numpy",
    )
    _assert_exact(
        optimized.state.torch_cpu_rng_state,
        reference.state.torch_cpu_rng_state,
        "rng.torch_cpu",
    )
    _assert_exact(
        optimized.state.torch_cuda_rng_states,
        reference.state.torch_cuda_rng_states,
        "rng.torch_cuda",
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


def _require_cuda_device() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise CudaBenchmarkUnavailable(
            "CUDA benchmark was not run: CUDA device cuda:0 is unavailable"
        )
    device = torch.device("cuda:0")
    try:
        torch.cuda.get_device_properties(device)
    except Exception as error:
        raise CudaBenchmarkUnavailable(
            f"CUDA benchmark was not run: CUDA device cuda:0 cannot be used: {error}"
        ) from error
    return device


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
    device = _require_cuda_device()
    configure_deterministic_runtime()

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

    outcomes: dict[BenchmarkPath, CompleteStepOutcome] = {}
    for path in PATHS:
        restore_state()
        result = run_step(path)
        outcomes[path] = CompleteStepOutcome(
            result=result,
            state=capture_eggroll_step_snapshot(
                trainer.trainable_params,
                trainer.optimizer,
                trainer.workspace,
            ),
        )
    assert_complete_step_equivalence(
        outcomes["reference"],
        outcomes["optimized"],
    )

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
        "equivalence": {
            "passed": True,
            "rtol": EQUIVALENCE_RTOL,
            "atol": EQUIVALENCE_ATOL,
        },
        "paths": path_results,
        "speedup_ratio": reference_median / optimized_median,
    }


def main() -> None:
    try:
        result: Mapping[str, Any] = run_cuda_benchmark()
    except BenchmarkError as error:
        sys.stderr.write(f"{error}\n")
        raise SystemExit(1) from error
    sys.stdout.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
