"""Alternating training on filtered Calc-ASDiv_A with frozen Qwen and MiniLM."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Protocol, TextIO

import torch

from eval.stage0_identity import (
    ASDIV_DATASET,
    ASDIV_REVISION,
    training_identity,
    stability_report_identity,
)
from eval.stream.generators.asdiv_a import (
    AsdivRecord,
    AsdivSelection,
    asdiv_a_selection_identity,
)
from train.alternating_config import (
    DEFAULT_VARIANCE_LOWER_THRESHOLD,
    DEFAULT_VARIANCE_UPPER_THRESHOLD,
    validate_scheduler_config,
)
from train.alternating_checkpoint import (
    AlternatingCheckpoint,
    CheckpointSchedule,
    build_alternating_checkpoint,
    latest_checkpoint,
    load_checkpoint,
    resume_config_conflicts,
    save_boundary_checkpoints,
    stage0_parameter_paths,
)
from train.alternating_evaluation import (
    DEFAULT_EVAL_PROBLEM_COUNT,
    HeldOutProblem,
    evaluate_unperturbed,
    load_held_out_problems,
)
from train.alternating_scheduler import (
    EvaluationRecord,
    VarianceHysteresisScheduler,
    PhaseEvaluator,
    TrainingEngine,
)
from train.answer_objective import (
    DEFAULT_PROMPT_ALIGNMENT_WEIGHT,
    validate_prompt_alignment_weight,
)
from train.eggroll_trainer import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_FITNESS_BATCH_SIZE,
    DEFAULT_LR as DEFAULT_EGGROLL_LR,
    DEFAULT_NUM_STEPS,
    DEFAULT_POP_SIZE,
    DEFAULT_RANK,
    DEFAULT_SIGMA,
    DEFAULT_VARIANCE_WEIGHT,
    EggrollTrainer,
    validate_eggroll_config,
)
from train.stage0_data import FitnessBatch
from train.trainer import DEFAULT_LR as DEFAULT_GRADIENT_LR, LatentCoreTrainer
from train.training_state import (
    EGGROLL_PARAMETER_PATHS,
    GRADIENT_PARAMETER_PATHS,
    TrainingState,
)
from train.training_results import EvaluationResult, ExperimentPosition, StepResult
from train.stage0_data import load_stage0_dataset, training_examples
from train.standalone_checkpoint import configure_deterministic_runtime
from workspace.concept_slots import DEFAULT_SLOT_COUNT


DEFAULT_EPOCHS = 5
DEFAULT_PHASE_STEPS = 50


def _dependency_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _runtime_identity() -> dict[str, object]:
    return {
        "initialization_seed": 0,
        "python_version": platform.python_version(),
        "numpy_version": _dependency_version("numpy"),
        "pytorch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda or "none"),
        "transformers_version": _dependency_version("transformers"),
        "datasets_version": _dependency_version("datasets"),
        "sentence_transformers_version": _dependency_version("sentence-transformers"),
        "device_topology": [f"cuda:{index}" for index in range(torch.cuda.device_count())],
        "dtype": "float32",
        "attention_implementation": "eager",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _selection_metadata(
    selection: AsdivSelection, count: int | None
) -> dict[str, object]:
    normalized_count = selection.problem_count if count is None else count
    records = selection.records[:normalized_count]
    identity = (
        selection.identity
        if normalized_count == selection.problem_count
        else asdiv_a_selection_identity(
            records=records,
            split=selection.split,
            seed=selection.seed,
            problem_count=normalized_count,
            revision=ASDIV_REVISION,
        )
    )
    return {
        "identity": identity,
        "split": selection.split,
        "seed": selection.seed,
        "problem_count": normalized_count,
        "ordered_item_ids": [record.id for record in records],
    }


def _validate_resume_metadata(
    checkpoint: AlternatingCheckpoint,
    *,
    identity: dict[str, object],
    selections: dict[str, object],
    run_config: dict[str, object],
) -> None:
    def plain(value: object) -> object:
        if isinstance(value, Mapping):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [plain(item) for item in value]
        return value

    def different_paths(saved: object, current: object, path: str) -> list[str]:
        saved = plain(saved)
        current = plain(current)
        if isinstance(saved, dict) and isinstance(current, dict):
            paths: list[str] = []
            for key in sorted(saved.keys() | current.keys()):
                if key not in saved or key not in current:
                    paths.append(f"{path}.{key}")
                else:
                    paths.extend(
                        different_paths(saved[key], current[key], f"{path}.{key}")
                    )
            return paths
        if saved != current:
            return [path]
        return []

    metadata = checkpoint.metadata
    conflicts = different_paths(metadata["identity"], identity, "$.identity")
    conflicts.extend(
        different_paths(metadata["selections"], selections, "$.selections")
    )
    conflicts.extend(
        resume_config_conflicts(
            checkpoint_config=metadata["run_config"],
            resume_config=run_config,
            completed_epochs=int(metadata["schedule"]["epoch"]),
        )
    )
    completed_phase_steps = int(metadata["schedule"]["completed_phase_steps"])
    metrics = metadata["metrics"]
    if completed_phase_steps > 0 and "phase_variance_sum" not in metrics:
        conflicts.append("$.metrics.phase_variance_sum")
    if conflicts:
        raise ValueError(
            "incompatible resume configuration: "
            + ", ".join(dict.fromkeys(conflicts))
        )


def _ts() -> str:
    t = time.localtime()
    return f"[{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}]"


def _log(message: str, output: TextIO = sys.stdout) -> None:
    print(f"{_ts()} {message}", file=output, flush=True)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


class _TrainerEngine:
    def __init__(self, trainer: object) -> None:
        self._trainer = trainer

    @property
    def max_consumed_records(self) -> int:
        """Expose each method's positive contiguous update capacity."""
        if isinstance(self._trainer, EggrollTrainer):
            return self._trainer.fitness_batch_size
        return 1

    def train_step(self, example: object, position: ExperimentPosition) -> StepResult:
        if isinstance(self._trainer, EggrollTrainer) and self._is_record_batch(example):
            records = tuple(example)
            start_position = position.example_position - len(records) + 1
            fitness_batch = FitnessBatch(
                records=tuple(
                    AsdivRecord(
                        id=f"alternating-{position.epoch}-{start_position + index}",
                        split="train",
                        question=question,
                        target=answer,
                    )
                    for index, (question, answer) in enumerate(records)
                ),
                start_position=start_position,
                next_position=position.example_position + 1,
            )
            return self._trainer.train_fitness_batch(fitness_batch, position)
        question, answer = example  # type: ignore[misc]
        return self._trainer.train_step(question, answer, position)  # type: ignore[attr-defined,no-any-return]

    @staticmethod
    def _is_record_batch(example: object) -> bool:
        return (
            isinstance(example, tuple)
            and bool(example)
            and isinstance(example[0], tuple)
        )


class _Evaluator:
    def __init__(self, model: object, problems: Sequence[HeldOutProblem]) -> None:
        self._model = model
        self._problems = problems

    def evaluate(self, position: ExperimentPosition) -> EvaluationResult:
        return evaluate_unperturbed(self._model, self._problems, position)  # type: ignore[arg-type]


class _BoundarySaver(Protocol):
    def __call__(
        self,
        schedule: CheckpointSchedule,
        *,
        phase_boundary: bool,
        epoch_boundary: bool,
        evaluation_record: EvaluationRecord[EvaluationResult],
    ) -> Sequence[str | Path]: ...


def _position_record(position: ExperimentPosition) -> dict[str, int | str]:
    return {
        "update_method": position.update_method,
        "cycle": position.cycle,
        "global_step": position.global_step,
        "epoch": position.epoch,
        "example_position": position.example_position,
        "phase_step": position.phase_step,
    }


def _training_record(result: StepResult) -> dict[str, float | int | str]:
    """Return the stable progress record for one completed update."""
    return {
        "record_type": "training",
        **_position_record(result.position),
        "language_model_loss": result.language_model_loss,
        "shared_variance": result.shared_variance,
    }


def _evaluation_record(
    record: EvaluationRecord[EvaluationResult],
) -> dict[str, float | int | str | list[str]]:
    """Return one deterministic held-out evaluation progress record."""
    return {
        "record_type": "evaluation",
        **_position_record(record.position),
        "boundaries": sorted(record.boundaries),
        "language_model_loss": record.result.language_model_loss,
        "shared_variance": record.result.shared_variance,
        "answer_exact_match": record.result.answer_exact_match,
    }


def _checkpoint_record(paths: Sequence[str | Path]) -> dict[str, str | list[str]]:
    """Return the progress record for checkpoint files written at a boundary."""
    return {
        "record_type": "checkpoint",
        "paths": [str(path) for path in paths],
    }


def _write_progress_record(
    record: dict[str, object], output: TextIO = sys.stdout
) -> None:
    """Write one machine-readable JSON record without buffering partial lines."""
    print(json.dumps(record, separators=(",", ":")), file=output, flush=True)


def _run_schedule(
    *,
    examples: Sequence[object],
    epochs: int,
    phase_steps: int,
    variance_lower_threshold: float = DEFAULT_VARIANCE_LOWER_THRESHOLD,
    variance_upper_threshold: float = DEFAULT_VARIANCE_UPPER_THRESHOLD,
    log_every: int = 50,
    eggroll_engine: TrainingEngine[StepResult],
    gradient_engine: TrainingEngine[StepResult],
    evaluator: PhaseEvaluator[EvaluationResult],
    save_boundary: _BoundarySaver,
    output: TextIO = sys.stdout,
    log_output: TextIO | None = None,
    resume: CheckpointSchedule | None = None,
) -> CheckpointSchedule | None:
    """Run injected alternating components without loading production dependencies."""
    validate_scheduler_config(
        phase_steps=phase_steps,
        epochs=epochs,
        log_every=log_every,
        variance_lower_threshold=variance_lower_threshold,
        variance_upper_threshold=variance_upper_threshold,
    )
    if not examples:
        return resume

    if resume is None:
        next_epoch = 1
        next_example_position = 0
        optimizer_call_offsets = {"gradient": 0, "eggroll": 0}
    else:
        next_epoch = resume.epoch
        next_example_position = resume.dataset_position + 1
        optimizer_call_offsets = {
            "gradient": resume.gradient_optimizer_calls,
            "eggroll": resume.eggroll_optimizer_calls,
        }
        if resume.dataset_position < 0:
            next_epoch += 1
        elif next_example_position == len(examples):
            next_epoch += 1
            next_example_position = 0

    scheduler = VarianceHysteresisScheduler(
        phase_steps=phase_steps,
        variance_lower_threshold=variance_lower_threshold,
        variance_upper_threshold=variance_upper_threshold,
        eggroll_engine=eggroll_engine,
        gradient_engine=gradient_engine,
        evaluator=evaluator,
    )
    if resume is not None:
        scheduler.restore(
            active_phase=resume.active_phase,
            completed_steps=resume.global_step,
            completed_phase_steps=resume.completed_phase_steps,
            phase_variance_sum=resume.phase_variance_sum,
        )
        scheduler._optimizer_call_counts.update(optimizer_call_offsets)

    latest_schedule = resume
    optimizer_call_counts = dict(optimizer_call_offsets)
    training_start = time.monotonic()
    for epoch in range(next_epoch, epochs + 1):
        first_position = next_example_position if epoch == next_epoch else 0
        epoch_start = time.monotonic()
        recent_losses: list[float] = []
        recent_vars: list[float] = []
        example_position = first_position
        while example_position < len(examples):
            evaluation_count = len(scheduler.evaluation_results)
            result = scheduler.train_records(
                examples[example_position:],
                epoch=epoch,
                example_position=example_position,
                records_until_logging_boundary=(
                    log_every - (scheduler.completed_steps % log_every)
                ),
            )
            optimizer_call_counts[result.position.update_method] = (
                result.position.optimizer_call_count
            )
            recent_losses.append(result.language_model_loss)
            recent_vars.append(result.shared_variance)
            if result.position.global_step > 0 and result.position.global_step % log_every == 0:
                _write_progress_record(_training_record(result), output)
                if log_output is not None:
                    elapsed = time.monotonic() - epoch_start
                    completed = result.position.example_position - first_position + 1
                    remaining = len(examples) - result.position.example_position - 1
                    per_example = elapsed / completed
                    _log(
                        f"  {result.position.update_method} epoch {epoch}/{epochs} "
                        f"[{example_position + 1}/{len(examples)}] "
                        f"step={result.position.global_step} "
                        f"loss={result.language_model_loss:.4f} "
                        f"avg_loss={sum(recent_losses) / len(recent_losses):.4f} "
                        f"var={result.shared_variance:.6f} "
                        f"avg_var={sum(recent_vars) / len(recent_vars):.6f} "
                        f"ETA {_format_duration(per_example * remaining)}",
                        log_output,
                    )

            phase_boundary = result.position.phase_step == phase_steps
            epoch_boundary = result.position.example_position == len(examples) - 1
            if epoch_boundary:
                scheduler.evaluate_epoch_boundary(result.position)
                if epoch == epochs:
                    scheduler.evaluate_final_partial_phase(result.position)

            boundary_evaluations = scheduler.evaluation_results[evaluation_count:]
            for record in boundary_evaluations:
                if "epoch" in record.boundaries:
                    _write_progress_record(_evaluation_record(record), output)
                    if log_output is not None:
                        _log(
                            f"  evaluation epoch={record.position.epoch} "
                            f"step={record.position.global_step} "
                            f"loss={record.result.language_model_loss:.4f} "
                            f"variance={record.result.shared_variance:.6f} "
                            f"exact_match={record.result.answer_exact_match:.3f}",
                            log_output,
                        )

            if phase_boundary or epoch_boundary:
                latest_schedule = _checkpoint_schedule(
                    result.position,
                    active_phase=scheduler.active_phase,
                    completed_phase_steps=scheduler.completed_phase_steps,
                    phase_variance_sum=scheduler.phase_variance_sum,
                    gradient_optimizer_calls=optimizer_call_counts["gradient"],
                    eggroll_optimizer_calls=optimizer_call_counts["eggroll"],
                )
                paths = save_boundary(
                    latest_schedule,
                    phase_boundary=phase_boundary,
                    epoch_boundary=epoch_boundary,
                    evaluation_record=boundary_evaluations[-1],
                )
                if epoch_boundary:
                    _write_progress_record(_checkpoint_record(paths), output)
                    if log_output is not None:
                        _log(f"saved: {', '.join(str(path) for path in paths)}", log_output)

            example_position += result.consumed_record_count

        if log_output is not None:
            _log(
                f"epoch {epoch}/{epochs} done "
                f"({_format_duration(time.monotonic() - epoch_start)})",
                log_output,
            )
            epochs_left = epochs - epoch
            if epochs_left:
                _log(
                    f"  ETA {_format_duration((time.monotonic() - training_start) / epoch * epochs_left)} "
                    f"for remaining {epochs_left} epochs",
                    log_output,
                )

    return latest_schedule


def _checkpoint_schedule(
    position: ExperimentPosition,
    *,
    active_phase: str,
    completed_phase_steps: int,
    phase_variance_sum: float,
    gradient_optimizer_calls: int = 0,
    eggroll_optimizer_calls: int = 0,
) -> CheckpointSchedule:
    return CheckpointSchedule(
        active_phase=active_phase,
        completed_phase_steps=completed_phase_steps,
        global_step=position.global_step,
        epoch=position.epoch,
        dataset_position=position.example_position,
        phase_variance_sum=phase_variance_sum,
        gradient_optimizer_calls=gradient_optimizer_calls,
        eggroll_optimizer_calls=eggroll_optimizer_calls,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one shared latent core on filtered Calc-ASDiv_A with alternating "
            "Eggroll and gradient phases through the frozen "
            "Qwen/Qwen2.5-0.5B-Instruct backbone."
        )
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--phase-steps", type=int, default=DEFAULT_PHASE_STEPS)
    parser.add_argument(
        "--variance-lower-threshold",
        type=float,
        default=DEFAULT_VARIANCE_LOWER_THRESHOLD,
    )
    parser.add_argument(
        "--variance-upper-threshold",
        type=float,
        default=DEFAULT_VARIANCE_UPPER_THRESHOLD,
    )
    parser.add_argument("--slot-count", type=int, default=DEFAULT_SLOT_COUNT)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--gradient-lr", type=float, default=DEFAULT_GRADIENT_LR)
    parser.add_argument("--eggroll-lr", type=float, default=DEFAULT_EGGROLL_LR)
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument(
        "--variance-weight", type=float, default=DEFAULT_VARIANCE_WEIGHT
    )
    parser.add_argument(
        "--prompt-alignment-weight",
        type=float,
        default=DEFAULT_PROMPT_ALIGNMENT_WEIGHT,
        help="Weight for preserving Qwen's question-conditioned decoder state.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
        help="Perturbed candidates evaluated together.",
    )
    parser.add_argument(
        "--fitness-batch-size",
        type=int,
        default=DEFAULT_FITNESS_BATCH_SIZE,
        help="Training records averaged into one Eggroll optimizer update.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        dest="use_amp",
        help="Use CUDA mixed precision for Eggroll candidate evaluation.",
    )
    parser.add_argument(
        "--eval-problem-count",
        type=int,
        default=DEFAULT_EVAL_PROBLEM_COUNT,
        help=(
            "Number of filtered Calc-ASDiv_A validation records used at each "
            "held-out evaluation. Default uses 128 validation records."
        ),
    )
    parser.add_argument(
        "--problem-count",
        type=int,
        default=None,
        help="Limit Calc-ASDiv_A training records. Default uses all 570 train records.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="checkpoints/alternating/")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume from. 'latest' uses the largest global step.",
    )
    parser.add_argument(
        "--stability-report",
        type=str,
        default=None,
        help="Passing stability-report artifact whose byte digest identifies this run.",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    validate_scheduler_config(
        phase_steps=args.phase_steps,
        epochs=args.epochs,
        log_every=args.log_every,
        variance_lower_threshold=args.variance_lower_threshold,
        variance_upper_threshold=args.variance_upper_threshold,
    )
    validate_prompt_alignment_weight(args.prompt_alignment_weight)
    validate_eggroll_config(
        args.pop_size,
        args.sigma,
        args.eggroll_lr,
        args.rank,
        args.eval_batch_size,
        args.fitness_batch_size,
    )
    return args


def _run_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_selection": {
            "dataset": ASDIV_DATASET,
            "revision": ASDIV_REVISION,
            "split": "train",
            "seed": 0,
            "count": args.problem_count,
        },
        "epochs": args.epochs,
        "phase_steps": args.phase_steps,
        "model_shape": {"slot_count": args.slot_count, "num_steps": args.num_steps},
        "gradient_optimizer": {
            "type": "Adam",
            "momentum": 0.0,
            "learning_rate": args.gradient_lr,
            "parameter_paths": list(GRADIENT_PARAMETER_PATHS),
        },
        "eggroll_optimizer": {
            "type": "SGD",
            "momentum": 0.0,
            "learning_rate": args.eggroll_lr,
            "parameter_paths": list(EGGROLL_PARAMETER_PATHS),
        },
        "eggroll_population": {
            "size": args.pop_size,
            "sigma": args.sigma,
            "rank": args.rank,
            "variance_weight": args.variance_weight,
            "prompt_alignment_weight": args.prompt_alignment_weight,
            "variance_lower_threshold": args.variance_lower_threshold,
            "variance_upper_threshold": args.variance_upper_threshold,
            "eval_batch_size": args.eval_batch_size,
            "fitness_batch_size": args.fitness_batch_size,
            "use_amp": args.use_amp,
        },
        "held_out_selection": {
            "dataset": ASDIV_DATASET,
            "revision": ASDIV_REVISION,
            "split": "validation",
            "seed": 0,
            "count": args.eval_problem_count,
        },
        "stability_report_identity": stability_report_identity(
            args.stability_report
        ),
        "logging_frequency": args.log_every,
    }


def _restore_model_state(state: TrainingState, values: object) -> None:
    model_state = values  # retain the named registry boundary
    with torch.no_grad():
        for name, parameter in state.trainable_params.items():
            parameter.copy_(model_state[f"model.{name}"])  # type: ignore[index]


def _optimizer_state_from_checkpoint(
    checkpoint: AlternatingCheckpoint,
    method: str,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    manifest = next(
        item
        for item in checkpoint.metadata["optimizer_manifests"]
        if item["method"] == method
    )
    current = optimizer.state_dict()
    parameter_ids = current["param_groups"][0]["params"]
    parameters = optimizer.param_groups[0]["params"]
    names = list(manifest["parameter_names"])
    if len(parameter_ids) != len(names) or len(parameters) != len(names):
        raise ValueError(
            f"$.optimizer_manifests.{method}.parameter_names: "
            "does not match the constructed optimizer"
        )
    restored_state: dict[int, dict[str, object]] = {}
    for parameter_id, parameter, name in zip(parameter_ids, parameters, names):
        values = dict(manifest["scalar_state"][name])
        for state_name, tensor_name in manifest["tensor_references"][name].items():
            tensor = checkpoint.tensors[tensor_name]
            if tensor.shape != parameter.shape:
                raise ValueError(
                    f"$.optimizer_manifests.{method}.tensor_references.{name}.{state_name}: "
                    "tensor shape does not match the constructed parameter"
                )
            values[state_name] = tensor
        if values:
            restored_state[parameter_id] = values

    group = dict(current["param_groups"][0])
    scalars = dict(manifest["parameter_groups"][0]["scalars"])
    if "beta1" in scalars or "beta2" in scalars:
        group["betas"] = (scalars.pop("beta1"), scalars.pop("beta2"))
    group.update(scalars)
    group["params"] = parameter_ids
    return {"state": restored_state, "param_groups": [group]}


def _restore_checkpoint_state(
    checkpoint: AlternatingCheckpoint,
    state: TrainingState,
    gradient_optimizer: torch.optim.Optimizer,
    eggroll_optimizer: torch.optim.Optimizer,
) -> None:
    current_names = set(state.trainable_params)
    expected_names = set(stage0_parameter_paths())
    if current_names != expected_names:
        raise ValueError(
            "$.tensor_manifest: constructed model parameter names do not match the checkpoint contract"
        )
    incompatibilities: list[str] = []
    for name, parameter in state.trainable_params.items():
        saved = checkpoint.tensors[f"model.{name}"]
        if saved.shape != parameter.shape:
            incompatibilities.append(f"$.tensor_manifest.model.{name}.shape")
        if saved.dtype != parameter.dtype:
            incompatibilities.append(f"$.tensor_manifest.model.{name}.dtype")
    if incompatibilities:
        raise ValueError(
            "incompatible resume configuration: " + ", ".join(incompatibilities)
        )
    gradient_state = _optimizer_state_from_checkpoint(
        checkpoint, "gradient", gradient_optimizer
    )
    eggroll_state = _optimizer_state_from_checkpoint(
        checkpoint, "eggroll", eggroll_optimizer
    )
    rng = checkpoint.metadata["rng"]
    python_rng = rng["python"]
    numpy_rng = rng["numpy"]
    python_state = (
        int(python_rng["version"]),
        tuple(python_rng["state"]),
        python_rng["gaussian_cache"],
    )
    numpy_state = (
        str(numpy_rng["bit_generator"]),
        checkpoint.tensors[numpy_rng["state_tensor"]].numpy(),
        int(numpy_rng["position"]),
        int(numpy_rng["has_gaussian"]),
        float(numpy_rng["gaussian_cache"]),
    )
    cuda_states = [
        checkpoint.tensors[item["state_tensor"]] for item in rng["cuda"]
    ]

    _restore_model_state(state, checkpoint.tensors)
    gradient_optimizer.load_state_dict(gradient_state)
    eggroll_optimizer.load_state_dict(eggroll_state)
    random.setstate(python_state)
    import numpy as np

    np.random.set_state(numpy_state)
    torch.set_rng_state(checkpoint.tensors[rng["pytorch_cpu"]["state_tensor"]])
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)


def main(argv: Sequence[str] | None = None) -> None:
    """Run a variance-controlled alternating experiment, optionally from a checkpoint."""
    configure_deterministic_runtime()
    args = _parse_args(argv)
    save_dir = Path(args.save_dir)
    run_config = _run_config(args)

    _log(f"device={args.device}")
    _log(
        f"config: epochs={args.epochs} variance_window={args.phase_steps} "
        f"variance_hysteresis={args.variance_lower_threshold}:"
        f"{args.variance_upper_threshold} "
        f"slots={args.slot_count} steps={args.num_steps} "
        f"pop={args.pop_size} sigma={args.sigma} "
        f"gradient_lr={args.gradient_lr} eggroll_optimizer=SGD "
        f"eggroll_lr={args.eggroll_lr} "
        f"rank={args.rank} var_weight={args.variance_weight} "
        f"alignment_weight={args.prompt_alignment_weight} "
        f"eval_batch={args.eval_batch_size} amp={args.use_amp}"
        f" fitness_batch={args.fitness_batch_size}"
    )

    load_start = time.monotonic()
    resumed: AlternatingCheckpoint | None = None
    if args.resume is not None:
        resume_path = latest_checkpoint(save_dir) if args.resume == "latest" else Path(args.resume)
        if resume_path is None:
            raise FileNotFoundError(f"no alternating checkpoint found in {save_dir}")
        resumed = load_checkpoint(resume_path)

    stage0_dataset = load_stage0_dataset()
    examples = training_examples(
        stage0_dataset,
        mode="alternating",
        epoch=1,
        problem_count=args.problem_count,
    )
    train_selection = _selection_metadata(
        stage0_dataset.train_selection, args.problem_count
    )
    held_out_selection = _selection_metadata(
        stage0_dataset.held_out_selection, args.eval_problem_count
    )
    selections = {"train": train_selection, "held_out": held_out_selection}
    checkpoint_identity = training_identity(
        runtime=_runtime_identity(),
        held_out_item_ids=held_out_selection["ordered_item_ids"],
    )
    if resumed is not None:
        _validate_resume_metadata(
            resumed,
            identity=checkpoint_identity,
            selections=selections,
            run_config=run_config,
        )
    _log(f"loaded {len(examples)} training problems ({_format_duration(time.monotonic() - load_start)})")
    held_out = load_held_out_problems(
        stage0_dataset.held_out_records(), args.eval_problem_count
    )

    _log(
        "building shared trainer (loading frozen "
        "Qwen/Qwen2.5-0.5B-Instruct + frozen MiniLM)..."
    )
    build_start = time.monotonic()
    state = TrainingState(args.slot_count, args.num_steps, args.device)
    gradient = LatentCoreTrainer(
        lr=args.gradient_lr,
        variance_weight=args.variance_weight,
        prompt_alignment_weight=args.prompt_alignment_weight,
        device=args.device,
        state=state,
    )
    eggroll = EggrollTrainer(
        pop_size=args.pop_size,
        sigma=args.sigma,
        lr=args.eggroll_lr,
        rank=args.rank,
        variance_weight=args.variance_weight,
        prompt_alignment_weight=args.prompt_alignment_weight,
        eval_batch_size=args.eval_batch_size,
        fitness_batch_size=args.fitness_batch_size,
        use_amp=args.use_amp,
        device=args.device,
        state=state,
    )
    parameters = getattr(state, "parameters", None)
    trainable_params = (
        f"{sum(parameter.numel() for parameter in parameters()):,}"
        if callable(parameters)
        else "unknown"
    )
    _log(
        f"trainers ready ({_format_duration(time.monotonic() - build_start)}), "
        f"trainable_params={trainable_params}"
    )
    _log("=" * 60)
    _log(f"  alternating training: {args.epochs} epochs x {len(examples)} examples")
    _log(
        f"  variance controller={args.phase_steps}-step averages "
        f"({args.variance_lower_threshold:.6f}, {args.variance_upper_threshold:.6f})"
    )
    _log("=" * 60)
    evaluator = _Evaluator(gradient, held_out)

    if resumed is not None:
        _restore_checkpoint_state(
            resumed, state, gradient.optimizer, eggroll.optimizer
        )

    def save_boundary(
        schedule: CheckpointSchedule,
        *,
        phase_boundary: bool,
        epoch_boundary: bool,
        evaluation_record: EvaluationRecord[EvaluationResult],
    ) -> Sequence[Path]:
        checkpoint = build_alternating_checkpoint(
            identity=checkpoint_identity,
            selections=selections,
            model_state={
                name: parameter.detach().cpu().clone()
                for name, parameter in state.trainable_params.items()
            },
            eggroll_optimizer=eggroll.optimizer,
            gradient_optimizer=gradient.optimizer,
            schedule=schedule,
            phase_steps=args.phase_steps,
            next_dataset_position=0 if epoch_boundary else schedule.dataset_position + 1,
            metrics={
                **_evaluation_record(evaluation_record),
                "phase_variance_sum": schedule.phase_variance_sum,
            },
            run_config=run_config,
        )
        return save_boundary_checkpoints(
            save_dir,
            checkpoint,
            phase_boundary=phase_boundary,
            epoch_boundary=epoch_boundary,
        )

    _run_schedule(
        examples=examples,
        epochs=args.epochs,
        phase_steps=args.phase_steps,
        variance_lower_threshold=args.variance_lower_threshold,
        variance_upper_threshold=args.variance_upper_threshold,
        log_every=args.log_every,
        eggroll_engine=_TrainerEngine(eggroll),
        gradient_engine=_TrainerEngine(gradient),
        evaluator=evaluator,
        save_boundary=save_boundary,
        log_output=sys.stdout,
        resume=resumed.schedule if resumed is not None else None,
    )


if __name__ == "__main__":
    main()
