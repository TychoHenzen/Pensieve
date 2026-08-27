"""Adaptive search for method-specific prompt-alignment weights."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
import torch.nn.functional as F

from codecs_module.decoder import DEFAULT_MAX_TOKENS, SlotDecoder
from eval.gate.answer_scoring import extract_predicted_number, score_numerical_answer
from eval.stage0_identity import (
    STAGE0_IDENTITY,
    load_frozen_qwen_backbone,
    load_minilm_model,
)
from train.answer_objective import (
    SUBJECT_LATENT_RUNS_PER_ANSWER,
    _decoder_hidden_and_logits,
    decoder_aligned_inputs_and_labels,
    prepare_training_example,
    prompt_teacher_state,
)
from train.eggroll_stability import StabilityMetrics, ValidatedStabilityReport
from train.eggroll_stability_guard import load_guarded_stability_report
from train.eggroll_trainer import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_FITNESS_BATCH_SIZE,
    DEFAULT_RANK,
    DEFAULT_SIGMA,
    DEFAULT_VARIANCE_WEIGHT,
    EggrollTrainer,
)
from train.eggroll_trainer import (
    DEFAULT_LR as DEFAULT_EGGROLL_LR,
)
from train.stage0_data import HELD_OUT_COUNT, load_stage0_dataset, training_examples
from train.standalone_checkpoint import configure_deterministic_runtime
from train.trainer import DEFAULT_LR as DEFAULT_GRADIENT_LR
from train.trainer import LatentCoreTrainer
from train.training_state import TrainingState
from train.vicreg import post_loop_slot_variance
from workspace.concept_slots import DEFAULT_SLOT_COUNT

Method = Literal["gradient", "eggroll"]
SearchStatus = Literal[
    "bracketed",
    "insufficient_effect",
    "no_positive_improvement",
    "unbracketed_lower",
    "unbracketed_upper",
    "method_unhealthy",
]
TrialRunner = Callable[[Method, float], "TrialMetrics"]
TrialObserver = Callable[["TrialResult"], None]

DEFAULT_LOWER_WEIGHT = 0.001
DEFAULT_UPPER_WEIGHT = 2.0
DEFAULT_CANDIDATES_PER_ROUND = 10
DEFAULT_ROUNDS = 8
DEFAULT_MAX_EXPANSIONS = 8
DEFAULT_MIN_SCORE_IMPROVEMENT = 0.01
MIN_SEPARATION_RETENTION_RATIO = 0.95
MAX_LANGUAGE_MODEL_LOSS_REGRESSION = 0.05
DEFAULT_TRAIN_PROBLEMS = 256
DEFAULT_VALIDATION_PROBLEMS = 64
DEFAULT_GRADIENT_UPDATES = 256
DEFAULT_EGGROLL_UPDATES = 256
DEFAULT_SEARCH_POP_SIZE = 32

SCORE_WEIGHTS = {
    "exact_accuracy_delta": 4.0,
    "first_token_accuracy_delta": 2.0,
    "language_model_loss_relative_improvement": 2.0,
    "prompt_alignment_mse_relative_improvement": 1.0,
    "separation_retention_delta": 1.0,
}


@dataclass(frozen=True)
class TrialMetrics:
    problem_count: int
    exact_accuracy: float
    first_token_accuracy: float
    valid_answer_rate: float
    diversity_ratio: float
    dominance_ratio: float
    separation_retention: float
    language_model_loss: float
    shared_variance: float
    student_cross_problem_cosine: float
    teacher_cross_problem_cosine: float
    student_teacher_mse: float
    decoded_answers: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrialResult:
    method: Method
    weight: float
    score: float
    score_components: Mapping[str, float]
    eligible: bool
    rejection_reasons: tuple[str, ...]
    metrics: TrialMetrics
    duration_seconds: float


@dataclass(frozen=True)
class MethodSearchResult:
    method: Method
    status: SearchStatus
    recommended_weight: float | None
    best_observed: TrialResult
    zero_control: TrialResult
    trials: tuple[TrialResult, ...]
    evaluation_waves: tuple[tuple[float, ...], ...]
    stability_report_sha256: str | None = None
    stability_evidence: Mapping[str, Any] | None = None


class TrainableModel(Protocol):
    workspace: Any
    encoder: Any
    latent_loop: Any
    tokenizer: Any


@dataclass(frozen=True)
class SearchAssets:
    backbone: Any
    sentence_model: Any
    train_examples: Mapping[Method, list[tuple[str, str]]]
    held_out: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    slot_count: int
    num_steps: int
    gradient_updates: int
    eggroll_updates: int
    gradient_lr: float
    eggroll_lr: float
    variance_weight: float
    pop_size: int
    sigma: float
    rank: int
    eval_batch_size: int
    fitness_batch_size: int
    use_amp: bool
    max_decode_tokens: int


def logarithmic_weights(lower: float, upper: float, count: int) -> list[float]:
    """Return an inclusive positive grid that is uniform in log space."""
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("weight bounds must be finite")
    if lower <= 0.0 or upper <= lower:
        raise ValueError("weight bounds must satisfy 0 < lower < upper")
    if isinstance(count, bool) or not isinstance(count, int) or count < 3:
        raise ValueError("candidate count must be an integer of at least 3")
    log_lower = math.log(lower)
    step = (math.log(upper) - log_lower) / (count - 1)
    weights = [math.exp(log_lower + step * index) for index in range(count)]
    weights[0] = lower
    weights[-1] = upper
    return weights


def alignment_effectiveness_components(
    metrics: TrialMetrics,
    baseline: TrialMetrics,
) -> dict[str, float]:
    """Return interpretable deltas from the same method's zero-weight control."""
    values = (
        metrics.exact_accuracy,
        metrics.first_token_accuracy,
        metrics.separation_retention,
        baseline.exact_accuracy,
        baseline.first_token_accuracy,
        baseline.separation_retention,
    )
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("scored trial ratios must be finite values from 0 through 1")
    losses = (
        metrics.language_model_loss,
        metrics.student_teacher_mse,
        baseline.language_model_loss,
        baseline.student_teacher_mse,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in losses):
        raise ValueError("scored trial losses must be finite and non-negative")
    lm_denominator = max(baseline.language_model_loss, 1e-8)
    alignment_denominator = max(baseline.student_teacher_mse, 1e-8)
    lm_improvement = (
        baseline.language_model_loss - metrics.language_model_loss
    ) / lm_denominator
    alignment_improvement = (
        baseline.student_teacher_mse - metrics.student_teacher_mse
    ) / alignment_denominator
    return {
        "exact_accuracy_delta": metrics.exact_accuracy - baseline.exact_accuracy,
        "first_token_accuracy_delta": (
            metrics.first_token_accuracy - baseline.first_token_accuracy
        ),
        "language_model_loss_relative_improvement": lm_improvement,
        "prompt_alignment_mse_relative_improvement": alignment_improvement,
        "separation_retention_delta": (
            metrics.separation_retention - baseline.separation_retention
        ),
    }


def alignment_effectiveness_score(
    metrics: TrialMetrics,
    baseline: TrialMetrics,
) -> float:
    """Score improvement over the same method's zero-alignment control."""
    components = alignment_effectiveness_components(metrics, baseline)
    return sum(SCORE_WEIGHTS[name] * value for name, value in components.items())


def alignment_candidate_rejection_reasons(
    metrics: TrialMetrics,
    baseline: TrialMetrics,
) -> tuple[str, ...]:
    """Reject proxy gains that worsen output conditioning or answer loss."""
    alignment_effectiveness_components(metrics, baseline)
    reasons: list[str] = []
    minimum_separation = (
        baseline.separation_retention * MIN_SEPARATION_RETENTION_RATIO
    )
    if metrics.separation_retention < minimum_separation:
        reasons.append("question_separation_regressed")
    maximum_lm_loss = baseline.language_model_loss * (
        1.0 + MAX_LANGUAGE_MODEL_LOSS_REGRESSION
    )
    if metrics.language_model_loss > maximum_lm_loss:
        reasons.append("language_model_loss_regressed")
    return tuple(reasons)


def _weight_key(weight: float) -> str:
    return format(weight, ".12g")


def _trial_rank(result: TrialResult) -> tuple[float, ...]:
    metrics = result.metrics
    return (
        float(result.eligible),
        result.score,
        metrics.exact_accuracy,
        metrics.first_token_accuracy,
        -metrics.language_model_loss,
        -metrics.student_teacher_mse,
        metrics.separation_retention,
        -result.weight,
    )


def search_method_weights(
    method: Method,
    *,
    lower: float,
    upper: float,
    candidates_per_round: int,
    rounds: int,
    max_expansions: int,
    run_trial: TrialRunner,
    on_trial: TrialObserver | None = None,
    min_score_improvement: float = DEFAULT_MIN_SCORE_IMPROVEMENT,
    zero_control_metrics: TrialMetrics | None = None,
) -> MethodSearchResult:
    """Bracket and refine one method against its zero-alignment control."""
    if method not in {"gradient", "eggroll"}:
        raise ValueError(f"unknown training method {method!r}")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be an integer of at least 1")
    if (
        isinstance(max_expansions, bool)
        or not isinstance(max_expansions, int)
        or max_expansions < 0
    ):
        raise ValueError("max_expansions must be a non-negative integer")
    logarithmic_weights(lower, upper, candidates_per_round)
    if not math.isfinite(min_score_improvement) or min_score_improvement < 0.0:
        raise ValueError("min_score_improvement must be finite and non-negative")

    cache: dict[str, TrialResult] = {}
    evaluation_waves: list[tuple[float, ...]] = []

    def evaluate(weight: float, baseline: TrialMetrics | None) -> TrialResult:
        key = _weight_key(weight)
        cached = cache.get(key)
        if cached is not None:
            return cached
        started = time.perf_counter()
        metrics = run_trial(method, weight)
        score_components = (
            dict.fromkeys(SCORE_WEIGHTS, 0.0)
            if baseline is None
            else alignment_effectiveness_components(metrics, baseline)
        )
        score = sum(
            SCORE_WEIGHTS[name] * value
            for name, value in score_components.items()
        )
        rejection_reasons = (
            ()
            if baseline is None
            else alignment_candidate_rejection_reasons(metrics, baseline)
        )
        result = TrialResult(
            method=method,
            weight=weight,
            score=score,
            score_components=score_components,
            eligible=not rejection_reasons,
            rejection_reasons=rejection_reasons,
            metrics=metrics,
            duration_seconds=time.perf_counter() - started,
        )
        cache[key] = result
        if on_trial is not None:
            on_trial(result)
        return result

    if zero_control_metrics is None:
        zero_control = evaluate(0.0, None)
    else:
        zero_control = TrialResult(
            method=method,
            weight=0.0,
            score=0.0,
            score_components=dict.fromkeys(SCORE_WEIGHTS, 0.0),
            eligible=True,
            rejection_reasons=(),
            metrics=zero_control_metrics,
            duration_seconds=0.0,
        )
        cache[_weight_key(0.0)] = zero_control
    initial_candidates = logarithmic_weights(lower, upper, candidates_per_round)
    evaluation_waves.append((0.0, *initial_candidates))
    for weight in initial_candidates:
        evaluate(weight, zero_control.metrics)

    log_step = (upper / lower) ** (1.0 / (candidates_per_round - 1))
    for _ in range(max_expansions):
        ranked = max(cache.values(), key=_trial_rank)
        positive = sorted(result.weight for result in cache.values() if result.weight > 0.0)
        if ranked.weight == 0.0 or ranked.weight == positive[0]:
            next_weight = positive[0] / log_step
        elif ranked.weight == positive[-1]:
            next_weight = positive[-1] * log_step
        else:
            break
        if not math.isfinite(next_weight) or next_weight <= 0.0:
            break
        evaluation_waves.append((next_weight,))
        evaluate(next_weight, zero_control.metrics)

    for _ in range(rounds):
        ranked = max(cache.values(), key=_trial_rank)
        positive = sorted(result.weight for result in cache.values() if result.weight > 0.0)
        if ranked.weight == 0.0:
            break
        index = positive.index(ranked.weight)
        if index == 0 or index == len(positive) - 1:
            break
        candidates = (
            math.sqrt(positive[index - 1] * ranked.weight),
            math.sqrt(ranked.weight * positive[index + 1]),
        )
        evaluation_waves.append(candidates)
        for weight in candidates:
            evaluate(weight, zero_control.metrics)

    trials = tuple(sorted(cache.values(), key=lambda result: result.weight))
    best_observed = max(trials, key=_trial_rank)
    positive = [result.weight for result in trials if result.weight > 0.0]
    if best_observed.weight == 0.0:
        status: SearchStatus = "no_positive_improvement"
        recommended_weight = None
    elif best_observed.score < min_score_improvement:
        status = "insufficient_effect"
        recommended_weight = None
    elif best_observed.weight == positive[0]:
        status = "unbracketed_lower"
        recommended_weight = None
    elif best_observed.weight == positive[-1]:
        status = "unbracketed_upper"
        recommended_weight = None
    else:
        status = "bracketed"
        recommended_weight = best_observed.weight
    return MethodSearchResult(
        method=method,
        status=status,
        recommended_weight=recommended_weight,
        best_observed=best_observed,
        zero_control=zero_control,
        trials=trials,
        evaluation_waves=tuple(evaluation_waves),
    )


def _trial_metrics_from_stability(metrics: StabilityMetrics) -> TrialMetrics:
    return TrialMetrics(
        problem_count=metrics.problem_count,
        exact_accuracy=metrics.exact_accuracy,
        first_token_accuracy=metrics.first_token_accuracy,
        valid_answer_rate=metrics.valid_answer_rate,
        diversity_ratio=metrics.output_diversity,
        dominance_ratio=metrics.output_dominance,
        separation_retention=metrics.separation_retention,
        language_model_loss=metrics.language_model_loss,
        shared_variance=metrics.shared_slot_variance,
        student_cross_problem_cosine=metrics.student_cross_problem_cosine,
        teacher_cross_problem_cosine=metrics.teacher_cross_problem_cosine,
        student_teacher_mse=metrics.student_teacher_mse,
    )


def unhealthy_eggroll_search_result(
    stability: ValidatedStabilityReport,
) -> MethodSearchResult:
    """Represent a failed absolute control without running positive trials."""
    report = stability.report
    final_metrics = report.checkpoints[-1].current
    failure_paths = tuple(failure.metric for failure in report.failed_thresholds)
    zero_control = TrialResult(
        method="eggroll",
        weight=0.0,
        score=0.0,
        score_components=dict.fromkeys(SCORE_WEIGHTS, 0.0),
        eligible=False,
        rejection_reasons=failure_paths,
        metrics=_trial_metrics_from_stability(final_metrics),
        duration_seconds=0.0,
    )
    return MethodSearchResult(
        method="eggroll",
        status="method_unhealthy",
        recommended_weight=None,
        best_observed=zero_control,
        zero_control=zero_control,
        trials=(zero_control,),
        evaluation_waves=((0.0,),),
        stability_report_sha256=stability.sha256,
        stability_evidence=report.to_dict(),
    )


def _cross_problem_cosine(states: Sequence[torch.Tensor]) -> float:
    normalized = F.normalize(torch.stack(list(states)), dim=-1)
    similarities = normalized @ normalized.T
    mask = ~torch.eye(len(states), dtype=torch.bool)
    return float(similarities[mask].mean().item())


def evaluate_trial(
    model: TrainableModel,
    problems: Sequence[tuple[str, str]],
    *,
    device: str,
    max_decode_tokens: int,
) -> TrialMetrics:
    """Measure task quality and cross-problem collapse without updating state."""
    if len(problems) < 2:
        raise ValueError("alignment search requires at least two validation problems")
    workspace = model.workspace
    encoder = model.encoder
    latent_loop = model.latent_loop
    tokenizer = model.tokenizer
    language_model = latent_loop.model
    decoder = SlotDecoder(
        language_model,
        tokenizer,
        max_tokens=max_decode_tokens,
        device=device,
    )
    modules = [encoder, latent_loop, language_model]
    previous_modes = [module.training for module in modules]
    workspace_state = workspace.snapshot()

    decoded_answers: list[str] = []
    student_states: list[torch.Tensor] = []
    teacher_states: list[torch.Tensor] = []
    first_token_correct = 0
    exact = 0
    valid = 0
    total_loss = 0.0
    total_variance = 0.0
    try:
        for module in modules:
            module.eval()
        with torch.no_grad():
            for question, answer in problems:
                prepared = prepare_training_example(tokenizer, question, answer)
                context_ids = prepared.context_input_ids.to(device)
                context_embeds = latent_loop.embed_tokens(context_ids)
                workspace.write_slots(encoder.encode(question))
                for _ in range(SUBJECT_LATENT_RUNS_PER_ANSWER):
                    latent_loop.run(workspace, context_embeds=context_embeds)
                slots = workspace.read_slots()
                input_embeds, labels = decoder_aligned_inputs_and_labels(
                    language_model,
                    slots,
                    prepared.answer_ids.to(device),
                    getattr(tokenizer, "eos_token_id", None),
                )
                sequence_length = labels.shape[1]
                attention_mask = torch.ones_like(labels, dtype=torch.long)
                position_ids = torch.arange(
                    sequence_length,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                hidden, logits = _decoder_hidden_and_logits(
                    language_model,
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                supervised = labels.ne(-100)
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).reshape_as(labels)
                total_loss += float(
                    ((losses * supervised).sum() / supervised.sum()).item()
                )
                slot_count = slots.shape[-2]
                student_state = hidden[0, slot_count - 1]
                teacher_state = prompt_teacher_state(language_model, context_ids)[0]
                prediction = int(logits[0, slot_count - 1].argmax().item())
                target = int(prepared.answer_ids[0].item())
                generated = decoder.decode(workspace)

                student_states.append(student_state.detach().cpu())
                teacher_states.append(teacher_state.detach().cpu())
                decoded_answers.append(generated)
                first_token_correct += int(prediction == target)
                exact += int(score_numerical_answer(generated, answer))
                valid += int(extract_predicted_number(generated) is not None)
                total_variance += float(post_loop_slot_variance(slots).item())
    finally:
        workspace.restore(workspace_state)
        for module, previous_mode in zip(modules, previous_modes, strict=True):
            module.train(previous_mode)

    count = len(problems)
    answer_counts = Counter(decoded_answers)
    student_cross = _cross_problem_cosine(student_states)
    teacher_cross = _cross_problem_cosine(teacher_states)
    teacher_separation = max(1.0 - teacher_cross, 1e-8)
    student_separation = max(1.0 - student_cross, 0.0)
    separation_retention = min(student_separation / teacher_separation, 1.0)
    return TrialMetrics(
        problem_count=count,
        exact_accuracy=exact / count,
        first_token_accuracy=first_token_correct / count,
        valid_answer_rate=valid / count,
        diversity_ratio=len(answer_counts) / count,
        dominance_ratio=answer_counts.most_common(1)[0][1] / count,
        separation_retention=separation_retention,
        language_model_loss=total_loss / count,
        shared_variance=total_variance / count,
        student_cross_problem_cosine=student_cross,
        teacher_cross_problem_cosine=teacher_cross,
        student_teacher_mse=float(
            F.mse_loss(
                torch.stack(student_states),
                torch.stack(teacher_states),
            ).item()
        ),
        decoded_answers=tuple(decoded_answers),
    )


def _trainer_for_trial(
    method: Method,
    weight: float,
    assets: SearchAssets,
    config: RuntimeConfig,
) -> LatentCoreTrainer | EggrollTrainer:
    configure_deterministic_runtime()
    state = TrainingState(
        slot_count=config.slot_count,
        num_steps=config.num_steps,
        device=config.device,
        backbone=assets.backbone,
        sentence_model=assets.sentence_model,
    )
    if method == "gradient":
        return LatentCoreTrainer(
            lr=config.gradient_lr,
            variance_weight=config.variance_weight,
            prompt_alignment_weight=weight,
            device=config.device,
            state=state,
        )
    return EggrollTrainer(
        pop_size=config.pop_size,
        sigma=config.sigma,
        lr=config.eggroll_lr,
        rank=config.rank,
        variance_weight=config.variance_weight,
        prompt_alignment_weight=weight,
        eval_batch_size=config.eval_batch_size,
        fitness_batch_size=config.fitness_batch_size,
        use_amp=config.use_amp,
        device=config.device,
        state=state,
    )


def _run_real_trial(
    method: Method,
    weight: float,
    assets: SearchAssets,
    config: RuntimeConfig,
) -> TrialMetrics:
    trainer = _trainer_for_trial(method, weight, assets, config)
    examples = assets.train_examples[method]
    updates = (
        config.gradient_updates if method == "gradient" else config.eggroll_updates
    )
    progress_interval = max(1, updates // 4)
    try:
        for index in range(updates):
            question, answer = examples[index % len(examples)]
            trainer.train_step(question, answer)
            if (index + 1) % progress_interval == 0 or index + 1 == updates:
                _log(
                    f"  {method} weight={weight:.8g} "
                    f"updates={index + 1}/{updates}"
                )
        return evaluate_trial(
            trainer,
            assets.held_out,
            device=config.device,
            max_decode_tokens=config.max_decode_tokens,
        )
    finally:
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_assets(args: argparse.Namespace) -> SearchAssets:
    configure_deterministic_runtime()
    dataset = load_stage0_dataset()
    train_examples = {
        method: training_examples(
            dataset,
            mode=method,
            epoch=1,
            problem_count=args.train_problems,
        )
        for method in ("gradient", "eggroll")
    }
    backbone = load_frozen_qwen_backbone(device=args.device)
    sentence_model = load_minilm_model()
    sentence_model.to(args.device)
    held_out = tuple(
        (record.question, record.target)
        for record in dataset.held_out_records()[: args.validation_problems]
    )
    return SearchAssets(
        backbone=backbone,
        sentence_model=sentence_model,
        train_examples=train_examples,
        held_out=held_out,
    )


def _ts() -> str:
    current = time.localtime()
    return f"[{current.tm_hour:02d}:{current.tm_min:02d}:{current.tm_sec:02d}]"


def _log(message: str) -> None:
    print(f"{_ts()} {message}", file=sys.stderr, flush=True)


def _trial_record(result: TrialResult) -> dict[str, Any]:
    return {
        "record_type": "alignment_weight_trial",
        **asdict(result),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _default_output_path() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("gate_results") / f"alignment-weight-search-{timestamp}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptively bracket and refine prompt-alignment weights independently "
            "for gradient and Eggroll on fresh deterministic Stage 0 states."
        )
    )
    parser.add_argument("--method", choices=("gradient", "eggroll", "both"), default="both")
    parser.add_argument("--lower-weight", type=float, default=DEFAULT_LOWER_WEIGHT)
    parser.add_argument("--upper-weight", type=float, default=DEFAULT_UPPER_WEIGHT)
    parser.add_argument(
        "--candidates-per-round", type=int, default=DEFAULT_CANDIDATES_PER_ROUND
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--max-expansions", type=int, default=DEFAULT_MAX_EXPANSIONS)
    parser.add_argument(
        "--min-score-improvement",
        type=float,
        default=DEFAULT_MIN_SCORE_IMPROVEMENT,
    )
    parser.add_argument("--train-problems", type=int, default=DEFAULT_TRAIN_PROBLEMS)
    parser.add_argument(
        "--validation-problems", type=int, default=DEFAULT_VALIDATION_PROBLEMS
    )
    parser.add_argument(
        "--gradient-updates", type=int, default=DEFAULT_GRADIENT_UPDATES
    )
    parser.add_argument("--eggroll-updates", type=int, default=DEFAULT_EGGROLL_UPDATES)
    parser.add_argument("--slot-count", type=int, default=DEFAULT_SLOT_COUNT)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--gradient-lr", type=float, default=DEFAULT_GRADIENT_LR)
    parser.add_argument("--eggroll-lr", type=float, default=DEFAULT_EGGROLL_LR)
    parser.add_argument("--variance-weight", type=float, default=DEFAULT_VARIANCE_WEIGHT)
    parser.add_argument("--pop-size", type=int, default=DEFAULT_SEARCH_POP_SIZE)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument(
        "--fitness-batch-size", type=int, default=DEFAULT_FITNESS_BATCH_SIZE
    )
    parser.add_argument(
        "--stability-report",
        type=Path,
        default=None,
        help="Absolute zero-alignment EGGROLL stability report.",
    )
    parser.add_argument("--amp", action="store_true", dest="use_amp")
    parser.add_argument(
        "--max-decode-tokens", type=int, default=DEFAULT_MAX_TOKENS
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    logarithmic_weights(
        args.lower_weight,
        args.upper_weight,
        args.candidates_per_round,
    )
    for name in (
        "rounds",
        "train_problems",
        "gradient_updates",
        "eggroll_updates",
        "slot_count",
        "num_steps",
        "pop_size",
        "rank",
        "eval_batch_size",
        "fitness_batch_size",
        "max_decode_tokens",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.max_expansions < 0:
        raise ValueError("--max-expansions must be non-negative")
    if (
        not math.isfinite(args.min_score_improvement)
        or args.min_score_improvement < 0.0
    ):
        raise ValueError("--min-score-improvement must be finite and non-negative")
    if not 2 <= args.validation_problems <= HELD_OUT_COUNT:
        raise ValueError(
            f"--validation-problems must be from 2 through {HELD_OUT_COUNT}"
        )
    if args.pop_size % 2 != 0:
        raise ValueError("--pop-size must be even for antithetic sampling")
    for name in ("gradient_lr", "eggroll_lr", "sigma"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.variance_weight) or args.variance_weight < 0.0:
        raise ValueError("--variance-weight must be finite and non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"requested CUDA device {args.device!r} is unavailable")


def _runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        device=args.device,
        slot_count=args.slot_count,
        num_steps=args.num_steps,
        gradient_updates=args.gradient_updates,
        eggroll_updates=args.eggroll_updates,
        gradient_lr=args.gradient_lr,
        eggroll_lr=args.eggroll_lr,
        variance_weight=args.variance_weight,
        pop_size=args.pop_size,
        sigma=args.sigma,
        rank=args.rank,
        eval_batch_size=args.eval_batch_size,
        fitness_batch_size=args.fitness_batch_size,
        use_amp=args.use_amp,
        max_decode_tokens=args.max_decode_tokens,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    configure_deterministic_runtime()
    methods: tuple[Method, ...] = (
        ("gradient", "eggroll") if args.method == "both" else (args.method,)
    )
    eggroll_stability: ValidatedStabilityReport | None = None
    if "eggroll" in methods:
        if args.stability_report is None:
            raise ValueError(
                "--stability-report is required for EGGROLL alignment search"
            )
        eggroll_stability = load_guarded_stability_report(
            args.stability_report,
            population=args.pop_size,
            sigma=args.sigma,
            rank=args.rank,
            fitness_batch_size=args.fitness_batch_size,
            evaluation_batch_size=args.eval_batch_size,
            variance_weight=args.variance_weight,
            prompt_alignment_weight=0.0,
            learning_rate=args.eggroll_lr,
            require_passing=False,
        )
    output = args.output or _default_output_path()
    progress_output = output.with_suffix(".jsonl")
    output_identity = os.path.normcase(str(output.resolve()))
    progress_identity = os.path.normcase(str(progress_output.resolve()))
    if output_identity == progress_identity:
        raise ValueError(
            "--output must not resolve to the derived JSONL progress path"
        )
    if output.exists() or progress_output.exists():
        raise FileExistsError(
            f"refusing to overwrite search output: {output} or {progress_output}"
        )
    progress_output.parent.mkdir(parents=True, exist_ok=True)

    runnable_methods = tuple(
        method
        for method in methods
        if method != "eggroll"
        or eggroll_stability is None
        or eggroll_stability.report.status == "passed"
    )
    assets: SearchAssets | None = None
    if runnable_methods:
        _log(
            f"loading pinned assets; methods={','.join(runnable_methods)} "
            f"train={args.train_problems} validation={args.validation_problems}"
        )
        assets = _load_assets(args)
    config = _runtime_config(args)
    completed_trials = 0
    maximum_trials = len(methods) * (
        1 + args.candidates_per_round + args.max_expansions + 2 * args.rounds
    )
    started = time.perf_counter()
    search_results: dict[str, MethodSearchResult] = {}

    def observe(result: TrialResult) -> None:
        nonlocal completed_trials
        completed_trials += 1
        record = _trial_record(result)
        with progress_output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        elapsed = time.perf_counter() - started
        eta = (
            elapsed / completed_trials * max(maximum_trials - completed_trials, 0)
        )
        _log(
            f"trial {completed_trials} (max {maximum_trials}) method={result.method} "
            f"weight={result.weight:.8g} score={result.score:.6f} "
            f"eligible={result.eligible} "
            f"exact={result.metrics.exact_accuracy:.3f} "
            f"first={result.metrics.first_token_accuracy:.3f} "
            f"lm={result.metrics.language_model_loss:.4f} "
            f"align_mse={result.metrics.student_teacher_mse:.4f} "
            f"separation={result.metrics.separation_retention:.3f} "
            f"ETA<={eta / 60:.1f}min"
        )

    for method in methods:
        if method == "eggroll" and eggroll_stability is not None:
            if eggroll_stability.report.status != "passed":
                search_results[method] = unhealthy_eggroll_search_result(
                    eggroll_stability
                )
                _log("EGGROLL zero control is unhealthy; skipping positive trials")
                continue
        _log(f"starting independent {method} search")
        if assets is None:
            raise RuntimeError("search assets were not loaded for a runnable method")
        result = search_method_weights(
            method,
            lower=args.lower_weight,
            upper=args.upper_weight,
            candidates_per_round=args.candidates_per_round,
            rounds=args.rounds,
            max_expansions=args.max_expansions,
            min_score_improvement=args.min_score_improvement,
            run_trial=lambda selected_method, weight: _run_real_trial(
                selected_method,
                weight,
                assets,
                config,
            ),
            on_trial=observe,
            zero_control_metrics=(
                _trial_metrics_from_stability(
                    eggroll_stability.report.checkpoints[-1].current
                )
                if method == "eggroll" and eggroll_stability is not None
                else None
            ),
        )
        if method == "eggroll" and eggroll_stability is not None:
            result = replace(
                result,
                stability_report_sha256=eggroll_stability.sha256,
                stability_evidence=eggroll_stability.report.to_dict(),
            )
        search_results[method] = result

    payload = {
        "schema_version": 2,
        "assets": dict(STAGE0_IDENTITY),
        "config": {
            **vars(args),
            "output": str(output),
            "progress_output": str(progress_output),
            "stability_report": (
                str(args.stability_report)
                if args.stability_report is not None
                else None
            ),
            "score_weights": SCORE_WEIGHTS,
            "score_reference": "same_method_zero_weight_control",
            "eligibility_guards": {
                "minimum_separation_retention_ratio": (
                    MIN_SEPARATION_RETENTION_RATIO
                ),
                "maximum_language_model_loss_regression": (
                    MAX_LANGUAGE_MODEL_LOSS_REGRESSION
                ),
            },
            "initialization_seed": 0,
            "fresh_state_per_trial": True,
        },
        "methods": {
            method: asdict(result) for method, result in search_results.items()
        },
        "duration_seconds": time.perf_counter() - started,
    }
    _write_json(output, payload)
    summary: dict[str, Any] = {}
    for method, result in search_results.items():
        summary[method] = {
            "status": result.status,
            "recommended_weight": result.recommended_weight,
            "best_observed_weight": result.best_observed.weight,
            "best_observed_score": result.best_observed.score,
        }
        _log(
            f"{method} status={result.status} "
            f"recommended_weight={result.recommended_weight} "
            f"best_observed={result.best_observed.weight:.8g} "
            f"score={result.best_observed.score:.6f}"
        )
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    _log(f"saved {output} and {progress_output}")


if __name__ == "__main__":
    main()
