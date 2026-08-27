from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests.eggroll_stability_fixtures import write_stability_report
from train.eggroll_stability_guard import load_guarded_stability_report
from train.search_alignment_weight import (
    TrialMetrics,
    alignment_candidate_rejection_reasons,
    alignment_effectiveness_score,
    logarithmic_weights,
    main,
    search_method_weights,
    unhealthy_eggroll_search_result,
)
from train.standalone_checkpoint import configure_deterministic_runtime


def _metrics(
    *,
    exact: float,
    first_token: float,
    valid: float,
    diversity: float,
    dominance: float,
    separation: float,
    language_model_loss: float = 4.0,
    alignment_mse: float = 20.0,
) -> TrialMetrics:
    return TrialMetrics(
        problem_count=8,
        exact_accuracy=exact,
        first_token_accuracy=first_token,
        valid_answer_rate=valid,
        diversity_ratio=diversity,
        dominance_ratio=dominance,
        separation_retention=separation,
        language_model_loss=language_model_loss,
        shared_variance=0.01,
        student_cross_problem_cosine=0.8,
        teacher_cross_problem_cosine=0.6,
        student_teacher_mse=alignment_mse,
    )


def test_effectiveness_score_does_not_reward_varied_nonsense() -> None:
    baseline = _metrics(
        exact=0.0,
        first_token=0.0,
        valid=0.5,
        diversity=0.5,
        dominance=0.5,
        separation=0.4,
        language_model_loss=6.0,
        alignment_mse=100.0,
    )
    varied_nonsense = _metrics(
        exact=0.0,
        first_token=0.0,
        valid=0.0,
        diversity=1.0,
        dominance=0.125,
        separation=1.0,
        language_model_loss=7.0,
        alignment_mse=100.0,
    )
    useful_outputs = _metrics(
        exact=0.25,
        first_token=0.5,
        valid=1.0,
        diversity=0.5,
        dominance=0.5,
        separation=0.5,
        language_model_loss=5.0,
        alignment_mse=80.0,
    )

    assert alignment_effectiveness_score(useful_outputs, baseline) > (
        alignment_effectiveness_score(varied_nonsense, baseline)
    )


def test_logarithmic_weights_include_exact_bounds() -> None:
    weights = logarithmic_weights(0.001, 1.0, 5)

    assert weights[0] == pytest.approx(0.001)
    assert weights[-1] == pytest.approx(1.0)
    assert weights == sorted(weights)
    assert len(set(weights)) == 5


@pytest.mark.parametrize(
    ("lower", "upper", "count"),
    [(0.0, 1.0, 5), (-0.1, 1.0, 5), (1.0, 1.0, 5), (1.0, 0.1, 5), (0.1, 1.0, 2)],
)
def test_logarithmic_weights_reject_invalid_search_space(
    lower: float,
    upper: float,
    count: int,
) -> None:
    with pytest.raises(ValueError):
        logarithmic_weights(lower, upper, count)


def test_methods_refine_independently_and_keep_zero_control() -> None:
    peaks = {"gradient": 0.03, "eggroll": 0.3}

    def trial(method: str, weight: float) -> TrialMetrics:
        distance = abs(
            math.log10(max(weight, 1e-6)) - math.log10(peaks[method])
        )
        quality = max(0.0, 1.0 - distance)
        return _metrics(
            exact=quality,
            first_token=quality,
            valid=quality,
            diversity=quality,
            dominance=1.0 - 0.875 * quality,
            separation=quality,
            language_model_loss=5.0 - quality,
            alignment_mse=100.0 - 50.0 * quality,
        )

    gradient = search_method_weights(
        "gradient",
        lower=0.001,
        upper=1.0,
        candidates_per_round=5,
        rounds=3,
        max_expansions=4,
        run_trial=trial,
    )
    eggroll = search_method_weights(
        "eggroll",
        lower=0.001,
        upper=1.0,
        candidates_per_round=5,
        rounds=3,
        max_expansions=4,
        run_trial=trial,
    )

    assert any(result.weight == 0.0 for result in gradient.trials)
    assert any(result.weight == 0.0 for result in eggroll.trials)
    assert gradient.status == "bracketed"
    assert eggroll.status == "bracketed"
    assert gradient.recommended_weight == pytest.approx(0.03, rel=0.5)
    assert eggroll.recommended_weight == pytest.approx(0.3, rel=0.5)
    assert gradient.recommended_weight != eggroll.recommended_weight


def test_search_expands_below_initial_range_before_recommending() -> None:
    target = 0.0001

    def trial(_method: str, weight: float) -> TrialMetrics:
        distance = abs(math.log10(max(weight, 1e-8)) - math.log10(target))
        quality = max(0.0, 1.0 - 0.5 * distance)
        return _metrics(
            exact=0.0,
            first_token=quality,
            valid=1.0,
            diversity=1.0,
            dominance=0.125,
            separation=quality,
            language_model_loss=6.0 - quality,
            alignment_mse=100.0 - 50.0 * quality,
        )

    result = search_method_weights(
        "gradient",
        lower=0.001,
        upper=1.0,
        candidates_per_round=5,
        rounds=2,
        max_expansions=4,
        run_trial=trial,
    )

    assert min(trial.weight for trial in result.trials if trial.weight > 0.0) < 0.001
    assert result.status == "bracketed"
    assert result.recommended_weight == pytest.approx(target, rel=0.75)


def test_search_expands_above_initial_range_before_recommending() -> None:
    target = 10.0

    def trial(_method: str, weight: float) -> TrialMetrics:
        distance = abs(math.log10(max(weight, 1e-8)) - math.log10(target))
        quality = max(0.0, 1.0 - 0.5 * distance)
        return _metrics(
            exact=0.0,
            first_token=quality,
            valid=1.0,
            diversity=1.0,
            dominance=0.125,
            separation=quality,
            language_model_loss=6.0 - quality,
            alignment_mse=100.0 - 50.0 * quality,
        )

    result = search_method_weights(
        "eggroll",
        lower=0.001,
        upper=1.0,
        candidates_per_round=5,
        rounds=2,
        max_expansions=4,
        run_trial=trial,
    )

    assert max(trial.weight for trial in result.trials) > 1.0
    assert result.status == "bracketed"
    assert result.recommended_weight == pytest.approx(target, rel=0.75)


def test_zero_control_win_is_reported_as_no_positive_improvement() -> None:
    def trial(_method: str, weight: float) -> TrialMetrics:
        return _metrics(
            exact=0.0,
            first_token=0.0,
            valid=1.0,
            diversity=1.0,
            dominance=0.125,
            separation=max(0.0, 0.5 - weight),
            language_model_loss=5.0 + weight,
            alignment_mse=100.0 + weight,
        )

    result = search_method_weights(
        "gradient",
        lower=0.001,
        upper=1.0,
        candidates_per_round=5,
        rounds=2,
        max_expansions=2,
        run_trial=trial,
    )

    assert result.status == "no_positive_improvement"
    assert result.recommended_weight is None
    assert result.best_observed.weight == 0.0


def test_tiny_positive_score_is_reported_as_insufficient_effect() -> None:
    def trial(_method: str, weight: float) -> TrialMetrics:
        improvement = min(weight, 0.001)
        return _metrics(
            exact=0.0,
            first_token=0.0,
            valid=1.0,
            diversity=1.0,
            dominance=0.125,
            separation=0.5 + improvement,
            language_model_loss=5.0 - improvement,
            alignment_mse=100.0 - improvement,
        )

    result = search_method_weights(
        "gradient",
        lower=0.001,
        upper=1.0,
        candidates_per_round=5,
        rounds=1,
        max_expansions=1,
        run_trial=trial,
    )

    assert result.status == "insufficient_effect"
    assert result.recommended_weight is None


def test_observed_loss_gain_with_question_separation_collapse_is_rejected() -> None:
    baseline = _metrics(
        exact=0.03125,
        first_token=0.0625,
        valid=0.65625,
        diversity=0.265625,
        dominance=0.328125,
        separation=0.2698090348498622,
        language_model_loss=2.436549883335829,
        alignment_mse=104.90262603759766,
    )
    collapsed = _metrics(
        exact=0.0,
        first_token=0.234375,
        valid=1.0,
        diversity=0.15625,
        dominance=0.296875,
        separation=0.1264760971877472,
        language_model_loss=1.9271381851285696,
        alignment_mse=100.95265197753906,
    )

    reasons = alignment_candidate_rejection_reasons(collapsed, baseline)

    assert "question_separation_regressed" in reasons


# covers: train/eggroll-stability-gate :: Alignment search rejects an unhealthy zero-weight control :: Eggroll zero control is unhealthy
def test_unhealthy_zero_control_blocks_every_positive_eggroll_trial_and_preserves_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_deterministic_runtime()
    report_path = tmp_path / "failed-stability.json"
    expected_report = write_stability_report(
        report_path,
        status="failed",
        population=32,
    )
    validated = load_guarded_stability_report(
        report_path,
        population=32,
        sigma=0.001,
        rank=4,
        fitness_batch_size=8,
        evaluation_batch_size=8,
        variance_weight=1.0,
        prompt_alignment_weight=0.0,
        learning_rate=0.1,
        require_passing=False,
    )

    result = unhealthy_eggroll_search_result(validated)

    assert result.status == "method_unhealthy"
    assert result.recommended_weight is None
    assert [trial.weight for trial in result.trials] == [0.0]
    assert result.evaluation_waves == ((0.0,),)
    assert result.stability_evidence == expected_report.to_dict()
    assert result.zero_control.rejection_reasons == ("language_model_loss",)

    output_path = tmp_path / "eggroll-only.json"
    runtime_accesses: list[str] = []
    monkeypatch.setattr(
        "train.search_alignment_weight._load_assets",
        lambda _args: runtime_accesses.append("assets"),
    )
    monkeypatch.setattr(
        "train.search_alignment_weight._run_real_trial",
        lambda *_args: runtime_accesses.append("trial"),
    )
    main(
        [
            "--method", "eggroll",
            "--stability-report", str(report_path),
            "--output", str(output_path),
            "--device", "cpu",
        ]
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert runtime_accesses == []
    assert payload["methods"]["eggroll"]["status"] == "method_unhealthy"
    assert payload["methods"]["eggroll"][
        "stability_evidence"
    ] == expected_report.to_dict()


# covers: train/eggroll-stability-gate :: Alignment search rejects an unhealthy zero-weight control :: Gradient search remains independent
def test_public_combined_search_keeps_gradient_independent_when_eggroll_is_unhealthy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_deterministic_runtime()
    report_path = tmp_path / "failed-stability.json"
    write_stability_report(report_path, status="failed", population=32)
    output_path = tmp_path / "search.json"
    calls: list[tuple[str, float]] = []
    loads: list[str] = []

    def fake_assets(_args):
        loads.append("assets")
        return object()

    def fake_trial(method: str, weight: float, _assets, _config) -> TrialMetrics:
        calls.append((method, weight))
        quality = min(weight, 0.1)
        return _metrics(
            exact=quality,
            first_token=quality,
            valid=1.0,
            diversity=1.0,
            dominance=0.125,
            separation=0.8,
            language_model_loss=2.0 - quality,
            alignment_mse=1.0 - quality,
        )

    monkeypatch.setattr(
        "train.search_alignment_weight._load_assets", fake_assets
    )
    monkeypatch.setattr(
        "train.search_alignment_weight._run_real_trial", fake_trial
    )

    main(
        [
            "--method", "both",
            "--stability-report", str(report_path),
            "--candidates-per-round", "3",
            "--rounds", "1",
            "--max-expansions", "0",
            "--output", str(output_path),
            "--device", "cpu",
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert loads == ["assets"]
    assert calls
    assert all(method == "gradient" for method, _weight in calls)
    assert any(weight > 0.0 for _method, weight in calls)
    assert payload["methods"]["eggroll"]["status"] == "method_unhealthy"
    assert payload["methods"]["eggroll"]["recommended_weight"] is None
    assert payload["methods"]["gradient"]["status"] != "method_unhealthy"
    assert payload["methods"]["gradient"]["trials"]


def test_public_search_rejects_output_progress_collision_before_runtime_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "nested" / "search.jsonl"
    runtime_accesses: list[str] = []
    monkeypatch.setattr(
        "train.search_alignment_weight._load_assets",
        lambda _args: runtime_accesses.append("assets"),
    )
    monkeypatch.setattr(
        "train.search_alignment_weight._run_real_trial",
        lambda *_args: runtime_accesses.append("trial"),
    )

    with pytest.raises(ValueError, match="derived JSONL progress path"):
        main(
            [
                "--method", "gradient",
                "--output", str(output_path),
                "--device", "cpu",
            ]
        )

    assert runtime_accesses == []
    assert not output_path.exists()
    assert not output_path.parent.exists()
