from __future__ import annotations

from pathlib import Path

from eval.stage0_identity import training_identity
from train.eggroll_stability import (
    FailedThreshold,
    ParameterRms,
    StabilityCheckpoint,
    StabilityConfiguration,
    StabilityMetrics,
    StabilityReport,
    canonical_implementation_identity,
)
from train.standalone_checkpoint import runtime_identity
from train.training_state import EGGROLL_PARAMETER_PATHS


def write_stability_report(
    path: Path,
    *,
    status: str = "passed",
    population: int = 128,
    sigma: float = 0.001,
    rank: int = 4,
    fitness_batch_size: int = 8,
    evaluation_batch_size: int = 8,
    variance_weight: float = 1.0,
    prompt_alignment_weight: float = 0.0,
    learning_rate: float = 0.1,
) -> StabilityReport:
    root = Path(__file__).resolve().parents[1]
    parameter_rms = tuple(ParameterRms(name, 1.0) for name in EGGROLL_PARAMETER_PATHS)
    baseline = StabilityMetrics(
        problem_count=64,
        parameter_rms=parameter_rms,
        language_model_loss=2.0,
        exact_accuracy=0.0,
        first_token_accuracy=0.0,
        valid_answer_rate=0.5,
        output_diversity=0.5,
        output_dominance=0.5,
        shared_slot_variance=0.1,
        student_teacher_mse=1.0,
        student_cross_problem_cosine=0.5,
        teacher_cross_problem_cosine=0.25,
        separation_retention=1.0,
    )
    healthy = StabilityMetrics(
        problem_count=64,
        parameter_rms=parameter_rms,
        language_model_loss=1.5,
        exact_accuracy=0.125,
        first_token_accuracy=0.25,
        valid_answer_rate=0.75,
        output_diversity=0.75,
        output_dominance=0.25,
        shared_slot_variance=0.1,
        student_teacher_mse=0.8,
        student_cross_problem_cosine=0.4,
        teacher_cross_problem_cosine=0.25,
        separation_retention=1.0,
    )
    failure = FailedThreshold(
        metric="language_model_loss",
        observed=2.2,
        operator="<=",
        threshold=2.1,
        baseline=2.0,
    )
    failed = StabilityMetrics(
        **{
            **healthy.__dict__,
            "language_model_loss": 2.2,
            "exact_accuracy": 0.0,
            "first_token_accuracy": 0.0,
        }
    )
    configuration = StabilityConfiguration(
        asset_identity=tuple(
            sorted(
                training_identity(
                    runtime=runtime_identity(),
                    held_out_item_ids=[f"held-out-{index}" for index in range(64)],
                ).items()
            )
        ),
        implementation=canonical_implementation_identity(root),
        optimizer_contract="torch.optim.SGD(momentum=0.0)",
        parameter_scope=EGGROLL_PARAMETER_PATHS,
        population=population,
        sigma=sigma,
        rank=rank,
        fitness_batch_size=fitness_batch_size,
        evaluation_batch_size=evaluation_batch_size,
        variance_weight=variance_weight,
        prompt_alignment_weight=prompt_alignment_weight,
        learning_rate=learning_rate,
        initialization_seed=0,
    )
    if status == "passed":
        checkpoints = tuple(
            StabilityCheckpoint(
                consumed_examples=examples,
                optimizer_call_count=0 if examples == 0 else examples // fitness_batch_size,
                baseline=baseline,
                current=baseline if examples == 0 else healthy,
                deltas=(),
                max_update_relative_matrix_rms=0.0 if examples == 0 else 0.005,
                elapsed_seconds=float(examples),
                eta_seconds=None,
            )
            for examples in (0, 8, 32, 256)
        )
        report = StabilityReport(1, configuration, "passed", 0, checkpoints, ())
    elif status == "failed":
        checkpoints = (
            StabilityCheckpoint(0, 0, baseline, baseline, (), 0.0, 0.0, None),
            StabilityCheckpoint(8, 1, baseline, failed, (), 0.005, 8.0, None, (failure,)),
        )
        report = StabilityReport(1, configuration, "failed", 1, checkpoints, (failure,))
    else:
        raise ValueError(f"unknown fixture status {status!r}")
    path.write_text(report.canonical_json() + "\n", encoding="utf-8")
    return report
