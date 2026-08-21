"""Alternating training on filtered Calc-MAWPS with frozen Qwen and MiniLM."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, TextIO

import torch

from eval.stage0_identity import CALC_MAWPS_DATASET, CALC_MAWPS_REVISION
from train.alternating_config import validate_scheduler_config
from train.alternating_checkpoint import (
    AlternatingCheckpoint,
    CheckpointSchedule,
    capture_checkpoint,
    latest_checkpoint,
    load_checkpoint,
    save_boundary_checkpoints,
    validate_resume_config,
)
from train.alternating_evaluation import (
    DEFAULT_EVAL_PROBLEM_COUNT,
    HeldOutProblem,
    evaluate_unperturbed,
    load_held_out_problems,
)
from train.alternating_scheduler import (
    EvaluationRecord,
    FixedBudgetScheduler,
    PhaseEvaluator,
    TrainingEngine,
)
from train.eggroll_trainer import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_LR as DEFAULT_EGGROLL_LR,
    DEFAULT_NUM_STEPS,
    DEFAULT_POP_SIZE,
    DEFAULT_RANK,
    DEFAULT_SIGMA,
    DEFAULT_VARIANCE_WEIGHT,
    EggrollTrainer,
)
from train.trainer import DEFAULT_LR as DEFAULT_GRADIENT_LR, LatentCoreTrainer
from train.training_state import TrainingState
from train.training_results import EvaluationResult, ExperimentPosition, StepResult
from train.stage0_data import load_stage0_dataset, training_examples
from workspace.concept_slots import DEFAULT_SLOT_COUNT


DEFAULT_EPOCHS = 5
DEFAULT_PHASE_STEPS = 500


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

    def train_step(self, example: object, position: ExperimentPosition) -> StepResult:
        question, answer = example  # type: ignore[misc]
        return self._trainer.train_step(question, answer, position)  # type: ignore[attr-defined,no-any-return]


class _Evaluator:
    def __init__(self, model: object, problems: Sequence[HeldOutProblem]) -> None:
        self._model = model
        self._problems = problems
        self.latest: EvaluationResult | None = None

    def evaluate(self, position: ExperimentPosition) -> EvaluationResult:
        self.latest = evaluate_unperturbed(self._model, self._problems, position)  # type: ignore[arg-type]
        return self.latest


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
    log_every: int = 50,
    eggroll_engine: TrainingEngine[StepResult],
    gradient_engine: TrainingEngine[StepResult],
    evaluator: PhaseEvaluator[EvaluationResult],
    save_boundary: Callable[..., Sequence[str | Path]],
    output: TextIO = sys.stdout,
    log_output: TextIO | None = None,
    resume: CheckpointSchedule | None = None,
) -> CheckpointSchedule | None:
    """Run injected alternating components without loading production dependencies."""
    validate_scheduler_config(
        phase_steps=phase_steps, epochs=epochs, log_every=log_every
    )
    if not examples:
        return resume

    scheduler = FixedBudgetScheduler(
        phase_steps=phase_steps,
        eggroll_engine=eggroll_engine,
        gradient_engine=gradient_engine,
        evaluator=evaluator,
    )
    if resume is None:
        next_epoch = 1
        next_example_position = 0
    else:
        scheduler._completed_steps = resume.global_step
        next_epoch = resume.epoch
        next_example_position = resume.dataset_position + 1
        if next_example_position == len(examples):
            next_epoch += 1
            next_example_position = 0

    latest_schedule = resume
    training_start = time.monotonic()
    for epoch in range(next_epoch, epochs + 1):
        first_position = next_example_position if epoch == next_epoch else 0
        epoch_start = time.monotonic()
        recent_losses: list[float] = []
        recent_vars: list[float] = []
        for example_position in range(first_position, len(examples)):
            evaluation_count = len(scheduler.evaluation_results)
            result = scheduler.train_step(
                examples[example_position],
                epoch=epoch,
                example_position=example_position,
            )
            recent_losses.append(result.language_model_loss)
            recent_vars.append(result.shared_variance)
            if result.position.global_step > 0 and result.position.global_step % log_every == 0:
                _write_progress_record(_training_record(result), output)
                if log_output is not None:
                    elapsed = time.monotonic() - epoch_start
                    completed = example_position - first_position + 1
                    remaining = len(examples) - example_position - 1
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
            epoch_boundary = example_position == len(examples) - 1
            if epoch_boundary:
                scheduler.evaluate_epoch_boundary(result.position)
                if epoch == epochs:
                    scheduler.evaluate_final_partial_phase(result.position)

            for record in scheduler.evaluation_results[evaluation_count:]:
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
                latest_schedule = _checkpoint_schedule(result.position, phase_steps)
                paths = save_boundary(
                    latest_schedule,
                    phase_boundary=phase_boundary,
                    epoch_boundary=epoch_boundary,
                )
                if epoch_boundary:
                    _write_progress_record(_checkpoint_record(paths), output)
                    if log_output is not None:
                        _log(f"saved: {', '.join(str(path) for path in paths)}", log_output)

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
    position: ExperimentPosition, phase_steps: int
) -> CheckpointSchedule:
    next_phase_index = position.global_step // phase_steps
    return CheckpointSchedule(
        active_phase="eggroll" if next_phase_index % 2 == 0 else "gradient",
        completed_phase_steps=position.global_step % phase_steps,
        global_step=position.global_step,
        epoch=position.epoch,
        dataset_position=position.example_position,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one shared latent core on filtered Calc-MAWPS with alternating "
            "Eggroll and gradient phases through the frozen "
            "Qwen/Qwen2.5-0.5B-Instruct backbone."
        )
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--phase-steps", type=int, default=DEFAULT_PHASE_STEPS)
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
        "--eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
        help="Perturbed candidates evaluated together.",
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
            "Number of filtered Calc-MAWPS validation records used at each "
            "held-out evaluation. Default uses 128 validation records."
        ),
    )
    parser.add_argument(
        "--problem-count",
        type=int,
        default=None,
        help="Limit filtered Calc-MAWPS training records. Default uses all 1,089 train records.",
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
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    validate_scheduler_config(
        phase_steps=args.phase_steps, epochs=args.epochs, log_every=args.log_every
    )
    return args


def _run_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_selection": {
            "dataset": CALC_MAWPS_DATASET,
            "revision": CALC_MAWPS_REVISION,
            "split": "train",
            "seed": 0,
            "count": args.problem_count,
        },
        "epochs": args.epochs,
        "phase_steps": args.phase_steps,
        "model_shape": {"slot_count": args.slot_count, "num_steps": args.num_steps},
        "gradient_optimizer": {"learning_rate": args.gradient_lr},
        "eggroll_optimizer": {"learning_rate": args.eggroll_lr},
        "eggroll_population": {
            "size": args.pop_size,
            "sigma": args.sigma,
            "rank": args.rank,
            "variance_weight": args.variance_weight,
            "eval_batch_size": args.eval_batch_size,
            "use_amp": args.use_amp,
        },
        "held_out_selection": {
            "dataset": CALC_MAWPS_DATASET,
            "revision": CALC_MAWPS_REVISION,
            "split": "validation",
            "seed": 0,
            "count": args.eval_problem_count,
        },
        "logging_frequency": args.log_every,
    }


def _restore_model_state(state: TrainingState, values: object) -> None:
    model_state = values  # retain the named registry boundary
    with torch.no_grad():
        for name, parameter in state.trainable_params.items():
            parameter.copy_(model_state[name])  # type: ignore[index]


def main(argv: Sequence[str] | None = None) -> None:
    """Run a fixed-budget alternating experiment, optionally from a checkpoint."""
    args = _parse_args(argv)
    run_config = _run_config(args)
    save_dir = Path(args.save_dir)

    _log(f"device={args.device}")
    _log(
        f"config: epochs={args.epochs} phase_steps={args.phase_steps} "
        f"slots={args.slot_count} steps={args.num_steps} "
        f"pop={args.pop_size} sigma={args.sigma} "
        f"gradient_lr={args.gradient_lr} eggroll_lr={args.eggroll_lr} "
        f"rank={args.rank} var_weight={args.variance_weight} "
        f"eval_batch={args.eval_batch_size} amp={args.use_amp}"
    )

    load_start = time.monotonic()
    resumed: AlternatingCheckpoint | None = None
    if args.resume is not None:
        resume_path = latest_checkpoint(save_dir) if args.resume == "latest" else Path(args.resume)
        if resume_path is None:
            raise FileNotFoundError(f"no alternating checkpoint found in {save_dir}")
        resumed = load_checkpoint(resume_path)
        validate_resume_config(
            checkpoint_config=resumed.run_config,
            resume_config=run_config,
            completed_epochs=resumed.schedule.epoch,
        )

    stage0_dataset = load_stage0_dataset()
    examples = training_examples(
        stage0_dataset,
        mode="alternating",
        epoch=1,
        problem_count=args.problem_count,
    )
    _log(f"loaded {len(examples)} training problems ({_format_duration(time.monotonic() - load_start)})")
    held_out = (
        resumed.held_out_selection
        if resumed is not None
        else load_held_out_problems(
            stage0_dataset.held_out_records(), args.eval_problem_count
        )
    )

    _log(
        "building shared trainer (loading frozen "
        "Qwen/Qwen2.5-0.5B-Instruct + frozen MiniLM)..."
    )
    build_start = time.monotonic()
    state = TrainingState(args.slot_count, args.num_steps, args.device)
    gradient = LatentCoreTrainer(
        lr=args.gradient_lr, device=args.device, state=state
    )
    eggroll = EggrollTrainer(
        pop_size=args.pop_size,
        sigma=args.sigma,
        lr=args.eggroll_lr,
        rank=args.rank,
        variance_weight=args.variance_weight,
        eval_batch_size=args.eval_batch_size,
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
    _log(f"  phase budget={args.phase_steps} steps (eggroll / gradient)")
    _log("=" * 60)
    evaluator = _Evaluator(gradient, held_out)

    if resumed is not None:
        _restore_model_state(state, resumed.model_state)
        eggroll.optimizer.load_state_dict(resumed.eggroll_optimizer_state)
        gradient.optimizer.load_state_dict(resumed.gradient_optimizer_state)
        random.setstate(resumed.python_random_state)
        torch.set_rng_state(resumed.torch_random_state)

    def save_boundary(
        schedule: CheckpointSchedule, *, phase_boundary: bool, epoch_boundary: bool
    ) -> Sequence[Path]:
        checkpoint = capture_checkpoint(
            model_state={
                name: parameter.detach().cpu().clone()
                for name, parameter in state.trainable_params.items()
            },
            eggroll_optimizer=eggroll.optimizer,
            gradient_optimizer=gradient.optimizer,
            schedule=schedule,
            run_config=run_config,
            held_out_selection=held_out,
            metrics=asdict(evaluator.latest) if evaluator.latest is not None else {},
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
