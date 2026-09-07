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
    DEFAULT_FITNESS_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_NUM_STEPS,
    DEFAULT_POP_SIZE,
    DEFAULT_RANK,
    DEFAULT_SIGMA,
    DEFAULT_VARIANCE_WEIGHT,
    EggrollTrainer,
)
from train.stage0_data import FitnessBatch, load_stage0_dataset
from train.standalone_checkpoint import (
    configure_deterministic_runtime,
    runtime_identity,
)
from workspace.concept_slots import DEFAULT_SLOT_COUNT

BenchmarkPath = Literal["reference", "optimized"]
PATHS: tuple[BenchmarkPath, BenchmarkPath] = ("reference", "optimized")
WARMUP_RUNS_PER_PATH = 2
MEASURED_RUNS_PER_PATH = 5
STATE_PREPARATION_STEPS = 3
EQUIVALENCE_RTOL = 1e-4
EQUIVALENCE_ATOL = 1e-4
MINIMUM_SPEEDUP_RATIO = 3.0


class BenchmarkError(RuntimeError):
    """A benchmark precondition or correctness check failed."""


class CudaBenchmarkUnavailable(BenchmarkError):
    """The pinned CUDA benchmark device cannot be used."""


class BenchmarkEquivalenceError(BenchmarkError):
    """Reference and optimized complete-step results differ."""


class BenchmarkPerformanceError(BenchmarkError):
    """The optimized path misses the required speed or memory gate."""


class CudaMeasurements(Protocol):
    """The CUDA timing surface used by the benchmark protocol."""

    def synchronize(self, device: torch.device | str | int | None = None) -> None: ...

    def reset_peak_memory_stats(self, device: torch.device | str | int | None = None) -> None: ...

    def max_memory_allocated(self, device: torch.device | str | int | None = None) -> int: ...


@dataclass(frozen=True)
class PathMeasurements:
    durations_seconds: tuple[float, ...]
    median_seconds: float
    peak_allocated_bytes: int
    consumed_examples_per_step: int = 1

    @property
    def durations_seconds_per_consumed_example(self) -> tuple[float, ...]:
        return tuple(duration / self.consumed_examples_per_step for duration in self.durations_seconds)

    @property
    def median_seconds_per_consumed_example(self) -> float:
        return self.median_seconds / self.consumed_examples_per_step

    @property
    def peak_allocated_bytes_per_consumed_example(self) -> float:
        return self.peak_allocated_bytes / self.consumed_examples_per_step


@dataclass(frozen=True)
class CompleteStepOutcome:
    result: Any
    state: EggrollStepSnapshot
    fitnesses: torch.Tensor | None = None
    gradients: tuple[torch.Tensor, ...] = ()


@dataclass
class _StepTrace:
    fitnesses: torch.Tensor | None = None
    gradients: tuple[torch.Tensor, ...] = ()


class _FullModelReferenceTap:
    """Benchmark-only adapter that recomputes the full Qwen sequence."""

    def __init__(self, optimized_adapter: Any) -> None:
        self.optimized_adapter = optimized_adapter

    @staticmethod
    def prepare_prefix(context_embeddings: torch.Tensor) -> torch.Tensor:
        return context_embeddings.detach()

    def cached_partial(
        self,
        context_embeddings: torch.Tensor,
        slot_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.optimized_adapter.full_reference(
            context_embeddings,
            slot_embeddings,
        )


def _measurement_order() -> tuple[BenchmarkPath, ...]:
    return tuple(path for _ in range(MEASURED_RUNS_PER_PATH) for path in PATHS)


def _equivalence_failure(path: str, detail: str) -> None:
    raise BenchmarkEquivalenceError(f"numerical equivalence failed at {path}: {detail}")


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
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
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
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
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
        optimized.state.optimizer_state,
        reference.state.optimizer_state,
        "optimizer",
    )
    try:
        _assert_close(
            optimized.state.parameter_values,
            reference.state.parameter_values,
            "parameters",
        )
    except BenchmarkEquivalenceError as error:
        diagnostics: list[str] = []
        if reference.fitnesses is not None and optimized.fitnesses is not None:
            diagnostics.append(
                f"fitness_max_abs_difference={(optimized.fitnesses - reference.fitnesses).abs().max().item():.9g}"
            )
        if reference.gradients and optimized.gradients:
            gradient_difference = (optimized.gradients[0] - reference.gradients[0]).abs()
            sign_mismatches = (
                (torch.signbit(optimized.gradients[0]) != torch.signbit(reference.gradients[0])).sum().item()
            )
            diagnostics.append(f"gradient0_max_abs_difference={gradient_difference.max().item():.9g}")
            diagnostics.append(f"gradient0_sign_mismatches={sign_mismatches}")
        raise BenchmarkEquivalenceError(f"{error}; {'; '.join(diagnostics)}") from error
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


def _performance_gate(
    measurements: Mapping[BenchmarkPath, PathMeasurements],
) -> dict[str, Any]:
    reference = measurements["reference"]
    optimized = measurements["optimized"]
    speedup_ratio = reference.median_seconds / optimized.median_seconds
    memory_passed = optimized.peak_allocated_bytes <= reference.peak_allocated_bytes
    speed_passed = speedup_ratio >= MINIMUM_SPEEDUP_RATIO
    if not speed_passed or not memory_passed:
        raise BenchmarkPerformanceError(
            "performance gate failed: "
            f"speedup_ratio={speedup_ratio:.6f} "
            f"minimum={MINIMUM_SPEEDUP_RATIO:.6f}; "
            f"reference_peak_allocated_bytes={reference.peak_allocated_bytes}; "
            f"optimized_peak_allocated_bytes={optimized.peak_allocated_bytes}"
        )
    return {
        "passed": True,
        "minimum_speedup_ratio": MINIMUM_SPEEDUP_RATIO,
        "speed_passed": speed_passed,
        "memory_passed": memory_passed,
    }


def run_benchmark_protocol(
    *,
    run_step: Callable[[BenchmarkPath], Any],
    restore_state: Callable[[], None],
    consumed_examples_per_step: int,
    device: torch.device | str | int,
    cuda: CudaMeasurements = torch.cuda,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[BenchmarkPath, PathMeasurements]:
    """Run the pinned warm-up and alternating measurement schedule."""
    if consumed_examples_per_step < 1:
        raise ValueError("consumed_examples_per_step must be positive")
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
            consumed_examples_per_step=consumed_examples_per_step,
        )
        for path in PATHS
    }


@contextmanager
def _execution_path(
    path: BenchmarkPath,
    trainer: EggrollTrainer,
    trace: _StepTrace | None = None,
) -> Any:
    """Select the benchmark-only materialized path for one complete step."""
    if path not in PATHS:
        raise ValueError(f"unknown benchmark path {path!r}")
    if path == "optimized" and trace is None:
        yield
        return

    original_linear = trainer_module.factorized_linear
    original_update = trainer_module.apply_factorized_update
    original_adapter = trainer.latent_loop.tap_adapter
    selected_linear = materialized_linear if path == "reference" else original_linear
    selected_update = apply_reference_pair_loop_update if path == "reference" else original_update

    def traced_update(*args: Any, **kwargs: Any) -> Any:
        if trace is not None:
            trace.fitnesses = torch.as_tensor(kwargs["fitnesses"]).detach().clone()
        gradients = selected_update(*args, **kwargs)
        if trace is not None:
            trace.gradients = tuple(gradient.detach().clone() for gradient in gradients)
        return gradients

    trainer_module.factorized_linear = selected_linear
    trainer_module.apply_factorized_update = traced_update
    if path == "reference":
        trainer.latent_loop.tap_adapter = _FullModelReferenceTap(original_adapter)
    try:
        yield
    finally:
        trainer_module.factorized_linear = original_linear
        trainer_module.apply_factorized_update = original_update
        trainer.latent_loop.tap_adapter = original_adapter


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
        raise CudaBenchmarkUnavailable("CUDA benchmark was not run: CUDA device cuda:0 is unavailable")
    device = torch.device("cuda:0")
    try:
        torch.cuda.get_device_properties(device)
    except Exception as error:
        raise CudaBenchmarkUnavailable(
            f"CUDA benchmark was not run: CUDA device cuda:0 cannot be used: {error}"
        ) from error
    return device


def _example_identity(
    record: Any,
    selection_identity: str,
    selection_index: int,
) -> dict[str, Any]:
    return {
        "selection_identity": selection_identity,
        "selection_index": selection_index,
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
        "fitness_batch_size": DEFAULT_FITNESS_BATCH_SIZE,
        "sigma": DEFAULT_SIGMA,
        "rank": DEFAULT_RANK,
        "learning_rate": DEFAULT_LR,
        "variance_weight": DEFAULT_VARIANCE_WEIGHT,
        "optimizer": "SGD",
        "optimizer_momentum": 0.0,
        "use_amp": False,
        "warmup_runs_per_path": WARMUP_RUNS_PER_PATH,
        "measured_runs_per_path": MEASURED_RUNS_PER_PATH,
        "measurement_order": list(_measurement_order()),
        "state_preparation_steps": STATE_PREPARATION_STEPS,
    }


def run_cuda_benchmark() -> dict[str, Any]:
    """Load the pinned assets once and benchmark one persisted fitness batch."""
    device = _require_cuda_device()
    configure_deterministic_runtime()

    dataset = load_stage0_dataset()
    records = dataset.training_records(mode="eggroll", epoch=1)
    batch_records = tuple(records[:DEFAULT_FITNESS_BATCH_SIZE])
    if len(batch_records) != DEFAULT_FITNESS_BATCH_SIZE:
        raise RuntimeError(f"benchmark requires exactly {DEFAULT_FITNESS_BATCH_SIZE} training records")
    fitness_batch = FitnessBatch(
        records=batch_records,
        start_position=0,
        next_position=DEFAULT_FITNESS_BATCH_SIZE,
    )
    trainer = EggrollTrainer(
        slot_count=DEFAULT_SLOT_COUNT,
        num_steps=DEFAULT_NUM_STEPS,
        pop_size=DEFAULT_POP_SIZE,
        sigma=DEFAULT_SIGMA,
        lr=DEFAULT_LR,
        rank=DEFAULT_RANK,
        variance_weight=DEFAULT_VARIANCE_WEIGHT,
        eval_batch_size=DEFAULT_EVAL_BATCH_SIZE,
        fitness_batch_size=DEFAULT_FITNESS_BATCH_SIZE,
        use_amp=False,
        device=str(device),
    )
    if any(parameter.dtype != torch.float32 for parameter in trainer.trainable_params):
        raise RuntimeError("benchmark trainable parameters must use float32")

    for _ in range(STATE_PREPARATION_STEPS):
        trainer.train_fitness_batch(fitness_batch)
    for parameter in trainer.trainable_params:
        parameter.grad = None

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
        with _execution_path(path, trainer):
            return trainer.train_fitness_batch(fitness_batch)

    outcomes: dict[BenchmarkPath, CompleteStepOutcome] = {}
    for path in PATHS:
        restore_state()
        trace = _StepTrace()
        with _execution_path(path, trainer, trace):
            result = trainer.train_fitness_batch(fitness_batch)
        outcomes[path] = CompleteStepOutcome(
            result=result,
            state=capture_eggroll_step_snapshot(
                trainer.trainable_params,
                trainer.optimizer,
                trainer.workspace,
            ),
            fitnesses=trace.fitnesses,
            gradients=trace.gradients,
        )
    assert_complete_step_equivalence(
        outcomes["reference"],
        outcomes["optimized"],
    )

    measurements = run_benchmark_protocol(
        run_step=run_step,
        restore_state=restore_state,
        consumed_examples_per_step=fitness_batch.consumed_record_count,
        device=device,
    )
    reference_median = measurements["reference"].median_seconds
    optimized_median = measurements["optimized"].median_seconds
    gate = _performance_gate(measurements)
    path_results = {}
    for path in PATHS:
        path_measurements = measurements[path]
        path_results[path] = {
            **asdict(path_measurements),
            "durations_seconds_per_consumed_example": (path_measurements.durations_seconds_per_consumed_example),
            "median_seconds_per_consumed_example": (path_measurements.median_seconds_per_consumed_example),
            "peak_allocated_bytes_per_consumed_example": (path_measurements.peak_allocated_bytes_per_consumed_example),
        }
    return {
        "schema_version": 2,
        "device": _device_identity(device),
        "software": runtime_identity(),
        "assets": dict(STAGE0_IDENTITY),
        "fitness_batch": {
            "selection_identity": dataset.train_selection.identity,
            "start_position": fitness_batch.start_position,
            "next_position": fitness_batch.next_position,
            "consumed_record_count": fitness_batch.consumed_record_count,
            "examples": [
                _example_identity(
                    record,
                    dataset.train_selection.identity,
                    selection_index,
                )
                for selection_index, record in enumerate(fitness_batch.records)
            ],
        },
        "config": _benchmark_config(device),
        "equivalence": {
            "passed": True,
            "rtol": EQUIVALENCE_RTOL,
            "atol": EQUIVALENCE_ATOL,
        },
        "paths": path_results,
        "speedup_ratio": reference_median / optimized_median,
        "gate": gate,
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
