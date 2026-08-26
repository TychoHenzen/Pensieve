"""Tests for Stage 0 trainability report types, validation, and serialization."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eval.stage0_identity import training_identity
from train.eggroll_stability import ImplementationIdentity, StabilityMetrics
from train.stage0_trainability import (
    ArmCheckpoint,
    ArmResult,
    CausalProbeResult,
    OverfitAttempt,
    OverfitProbeResult,
    TrainabilityAssetIdentity,
    TrainabilityConfiguration,
    TrainabilityReport,
    TrainabilityReportValidationError,
    build_trainability_record_selections,
    canonical_trainability_implementation_identity,
)


@pytest.fixture
def sample_metrics() -> StabilityMetrics:
    """Create a sample StabilityMetrics instance for testing."""
    return StabilityMetrics(
        problem_count=64,
        parameter_rms=tuple(),
        language_model_loss=1.5,
        exact_accuracy=0.5,
        first_token_accuracy=0.75,
        valid_answer_rate=0.9,
        output_diversity=0.8,
        output_dominance=0.1,
        shared_slot_variance=0.5,
        student_teacher_mse=0.3,
        student_cross_problem_cosine=0.6,
        teacher_cross_problem_cosine=0.7,
        separation_retention=0.95,
    )


@pytest.fixture
def sample_implementation() -> ImplementationIdentity:
    """Create a sample ImplementationIdentity."""
    sha = "a" * 64
    return ImplementationIdentity(sha, (("train/stage0_trainability.py", "b" * 64),))


@pytest.fixture
def sample_asset_identity() -> TrainabilityAssetIdentity:
    """Create a sample TrainabilityAssetIdentity."""
    return TrainabilityAssetIdentity(
        stability_report_digest="c" * 64,
        held_out_record_identifiers=(
            ("test_001", "id_001"),
            ("test_002", "id_002"),
        ),
        training_record_identifiers_overfit=(
            ("train_001", "id_001"),
        ),
        training_record_identifiers_32=(
            ("train_001", "id_001"),
            ("train_002", "id_002"),
        ),
    )


class TestTrainabilityAssetIdentity:
    """Test TrainabilityAssetIdentity validation and serialization."""

    def test_invalid_digest(self, sample_asset_identity: TrainabilityAssetIdentity) -> None:
        """Test that invalid digest raises ValueError."""
        with pytest.raises(ValueError, match="SHA-256"):
            TrainabilityAssetIdentity(
                stability_report_digest="invalid",
                held_out_record_identifiers=sample_asset_identity.held_out_record_identifiers,
                training_record_identifiers_overfit=(
                    sample_asset_identity.training_record_identifiers_overfit
                ),
                training_record_identifiers_32=sample_asset_identity.training_record_identifiers_32,
            )

    def test_empty_held_out_records(self, sample_asset_identity: TrainabilityAssetIdentity) -> None:
        """Test that empty held-out records raises ValueError."""
        with pytest.raises(ValueError, match="held-out"):
            TrainabilityAssetIdentity(
                stability_report_digest=sample_asset_identity.stability_report_digest,
                held_out_record_identifiers=(),
                training_record_identifiers_overfit=sample_asset_identity.training_record_identifiers_overfit,
                training_record_identifiers_32=sample_asset_identity.training_record_identifiers_32,
            )

    def test_to_dict(self, sample_asset_identity: TrainabilityAssetIdentity) -> None:
        """Test to_dict serialization."""
        result = sample_asset_identity.to_dict()
        assert result["stability_report_digest"] == "c" * 64
        assert result["held_out_record_count"] == 2
        assert result["training_record_count_overfit"] == 1
        assert result["training_record_count_32"] == 2


class TestTrainabilityConfiguration:
    """Test TrainabilityConfiguration validation and serialization."""

    def test_valid_configuration(
        self,
        sample_asset_identity: TrainabilityAssetIdentity,
        sample_implementation: ImplementationIdentity,
    ) -> None:
        """Test creating a valid configuration."""
        config = TrainabilityConfiguration(
            asset_identity=sample_asset_identity,
            implementation=sample_implementation,
            stability_configuration={"test": "value"},
        )
        assert config.overfit_learning_rates == (0.0001, 0.001, 0.01)
        assert config.min_separation_ratio == 0.95

    def test_to_dict(
        self,
        sample_asset_identity: TrainabilityAssetIdentity,
        sample_implementation: ImplementationIdentity,
    ) -> None:
        """Test to_dict serialization."""
        config = TrainabilityConfiguration(
            asset_identity=sample_asset_identity,
            implementation=sample_implementation,
            stability_configuration={"test": "value"},
        )
        result = config.to_dict()
        assert result["min_separation_ratio"] == 0.95
        assert result["max_update_relative_matrix_rms"] == 0.01
        assert len(result["overfit_learning_rates"]) == 3


class TestOverfitAttempt:
    """Test OverfitAttempt validation and serialization."""

    def test_passed_attempt(self, sample_metrics: StabilityMetrics) -> None:
        """Test creating a passed overfit attempt."""
        attempt = OverfitAttempt(
            learning_rate=0.001,
            status="passed",
            baseline_metrics=sample_metrics,
            checkpoints=((64, sample_metrics),),
        )
        assert attempt.status == "passed"

    def test_failed_attempt_missing_baseline(self, sample_metrics: StabilityMetrics) -> None:
        """Test that passed status requires baseline."""
        with pytest.raises(ValueError, match="baseline"):
            OverfitAttempt(
                learning_rate=0.001,
                status="passed",
                baseline_metrics=None,
                checkpoints=((64, sample_metrics),),
            )

    def test_non_finite_attempt(self) -> None:
        """Test creating a non-finite overfit attempt."""
        attempt = OverfitAttempt(
            learning_rate=0.001,
            status="non_finite",
            baseline_metrics=None,
            failed_reason="NaN in gradient",
        )
        assert attempt.failed_reason == "NaN in gradient"

    def test_invalid_learning_rate(self, sample_metrics: StabilityMetrics) -> None:
        """Test that invalid learning rate raises ValueError."""
        with pytest.raises(ValueError, match="positive finite"):
            OverfitAttempt(
                learning_rate=-0.001,
                status="passed",
                baseline_metrics=sample_metrics,
                checkpoints=((64, sample_metrics),),
            )

    def test_to_dict(self, sample_metrics: StabilityMetrics) -> None:
        """Test to_dict serialization."""
        attempt = OverfitAttempt(
            learning_rate=0.001,
            status="passed",
            baseline_metrics=sample_metrics,
            checkpoints=((0, sample_metrics), (64, sample_metrics)),
        )
        result = attempt.to_dict()
        assert result["learning_rate"] == 0.001
        assert result["status"] == "passed"
        assert len(result["checkpoints"]) == 2


class TestOverfitProbeResult:
    """Test OverfitProbeResult validation and serialization."""

    def test_passed_probe(self, sample_metrics: StabilityMetrics) -> None:
        """Test creating a passed probe."""
        attempt = OverfitAttempt(
            learning_rate=0.001,
            status="passed",
            baseline_metrics=sample_metrics,
            checkpoints=((64, sample_metrics),),
        )
        probe = OverfitProbeResult(
            status="passed",
            attempts=(attempt,),
        )
        assert probe.status == "passed"

    def test_passed_probe_without_passing_attempt(
        self, sample_metrics: StabilityMetrics
    ) -> None:
        """Test that passed probe requires a passing attempt."""
        attempt = OverfitAttempt(
            learning_rate=0.001,
            status="failed",
            baseline_metrics=None,
        )
        with pytest.raises(ValueError, match="passing attempt"):
            OverfitProbeResult(
                status="passed",
                attempts=(attempt,),
            )

    def test_inconclusive_probe(self, sample_metrics: StabilityMetrics) -> None:
        """Test creating an inconclusive probe."""
        attempt = OverfitAttempt(
            learning_rate=0.001,
            status="failed",
            baseline_metrics=None,
        )
        probe = OverfitProbeResult(
            status="inconclusive",
            attempts=(attempt,),
            failed_conditions=("timeout",),
        )
        assert probe.status == "inconclusive"

    def test_to_dict(self, sample_metrics: StabilityMetrics) -> None:
        """Test to_dict serialization."""
        attempt = OverfitAttempt(
            learning_rate=0.001,
            status="passed",
            baseline_metrics=sample_metrics,
            checkpoints=((64, sample_metrics),),
        )
        probe = OverfitProbeResult(status="passed", attempts=(attempt,))
        result = probe.to_dict()
        assert result["status"] == "passed"
        assert result["attempt_count"] == 1


class TestCausalProbeResult:
    """Test CausalProbeResult validation and serialization."""

    def test_passed_causal_probe(self) -> None:
        """Test creating a passed causal probe."""
        probe = CausalProbeResult(
            method="gradient",
            status="passed",
            baseline_objective=1.5,
            predicted_objective_delta=-0.1,
            observed_objective_delta=-0.15,
            max_update_relative_matrix_rms=0.005,
            baseline_separation=0.95,
            post_update_separation=0.94,
        )
        assert probe.status == "passed"

    def test_passed_probe_missing_metrics(self) -> None:
        """Test that passed probe requires all metrics."""
        with pytest.raises(ValueError, match="all metrics"):
            CausalProbeResult(
                method="gradient",
                status="passed",
                baseline_objective=None,
                predicted_objective_delta=-0.1,
                observed_objective_delta=-0.15,
                max_update_relative_matrix_rms=0.005,
                baseline_separation=0.95,
                post_update_separation=0.94,
            )

    def test_direction_mismatch(self) -> None:
        """Test creating a direction_mismatch probe."""
        probe = CausalProbeResult(
            method="eggroll",
            status="direction_mismatch",
            baseline_objective=1.5,
            predicted_objective_delta=-0.1,
            observed_objective_delta=0.05,
            max_update_relative_matrix_rms=0.002,
            baseline_separation=0.95,
            post_update_separation=0.95,
        )
        assert probe.status == "direction_mismatch"

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        probe = CausalProbeResult(
            method="gradient",
            status="passed",
            baseline_objective=1.5,
            predicted_objective_delta=-0.1,
            observed_objective_delta=-0.15,
            max_update_relative_matrix_rms=0.005,
            baseline_separation=0.95,
            post_update_separation=0.94,
        )
        result = probe.to_dict()
        assert result["method"] == "gradient"
        assert result["status"] == "passed"
        assert result["predicted_objective_delta"] == -0.1


class TestArmCheckpoint:
    """Test ArmCheckpoint validation and serialization."""

    def test_valid_checkpoint(self, sample_metrics: StabilityMetrics) -> None:
        """Test creating a valid checkpoint."""
        cp = ArmCheckpoint(
            consumed_examples=32,
            optimizer_call_count=32,
            metrics=sample_metrics,
            max_update_relative_matrix_rms=0.005,
        )
        assert cp.consumed_examples == 32

    def test_to_dict(self, sample_metrics: StabilityMetrics) -> None:
        """Test to_dict serialization."""
        cp = ArmCheckpoint(
            consumed_examples=32,
            optimizer_call_count=32,
            metrics=sample_metrics,
        )
        result = cp.to_dict()
        assert result["consumed_examples"] == 32
        assert result["optimizer_call_count"] == 32


class TestArmResult:
    """Test ArmResult validation and serialization."""

    def test_no_update_arm(self, sample_metrics: StabilityMetrics) -> None:
        """Test creating a no-update control arm."""
        baseline = ArmCheckpoint(0, 0, sample_metrics)
        cp = ArmCheckpoint(32, 0, sample_metrics)
        arm = ArmResult(
            arm="no_update",
            method=None,
            status="passed",
            baseline_checkpoint=baseline,
            checkpoints=(cp,),
            recalibration_eligible=False,
        )
        assert arm.arm == "no_update"

    def test_gradient_arm(self, sample_metrics: StabilityMetrics) -> None:
        """Test creating a gradient-only arm."""
        baseline = ArmCheckpoint(0, 0, sample_metrics)
        cp = ArmCheckpoint(32, 32, sample_metrics)
        arm = ArmResult(
            arm="gradient_only",
            method="gradient",
            status="viable",
            baseline_checkpoint=baseline,
            checkpoints=(cp,),
            recalibration_eligible=True,
        )
        assert arm.method == "gradient"
        assert arm.recalibration_eligible

    def test_invalid_no_update_with_method(self, sample_metrics: StabilityMetrics) -> None:
        """Test that no-update arm cannot specify a method."""
        baseline = ArmCheckpoint(0, 0, sample_metrics)
        with pytest.raises(ValueError, match="must not specify"):
            ArmResult(
                arm="no_update",
                method="gradient",
                status="passed",
                baseline_checkpoint=baseline,
                checkpoints=(),
                recalibration_eligible=False,
            )

    def test_to_dict(self, sample_metrics: StabilityMetrics) -> None:
        """Test to_dict serialization."""
        baseline = ArmCheckpoint(0, 0, sample_metrics)
        cp = ArmCheckpoint(32, 32, sample_metrics)
        arm = ArmResult(
            arm="gradient_only",
            method="gradient",
            status="viable",
            baseline_checkpoint=baseline,
            checkpoints=(cp,),
            recalibration_eligible=True,
        )
        result = arm.to_dict()
        assert result["arm"] == "gradient_only"
        assert result["method"] == "gradient"
        assert result["checkpoint_count"] == 1


class TestTrainabilityReport:
    """Test TrainabilityReport validation and serialization."""

    def test_valid_report(
        self,
        sample_metrics: StabilityMetrics,
        sample_asset_identity: TrainabilityAssetIdentity,
        sample_implementation: ImplementationIdentity,
    ) -> None:
        """Test creating a valid trainability report."""
        config = TrainabilityConfiguration(
            asset_identity=sample_asset_identity,
            implementation=sample_implementation,
            stability_configuration={"test": "value"},
        )
        baseline = ArmCheckpoint(0, 0, sample_metrics)
        cp = ArmCheckpoint(32, 32, sample_metrics)
        arm = ArmResult(
            arm="gradient_only",
            method="gradient",
            status="viable",
            baseline_checkpoint=baseline,
            checkpoints=(cp,),
            recalibration_eligible=True,
        )
        attempt = OverfitAttempt(
            learning_rate=0.001,
            status="passed",
            baseline_metrics=sample_metrics,
            checkpoints=((64, sample_metrics),),
        )
        probe = OverfitProbeResult(status="passed", attempts=(attempt,))

        report = TrainabilityReport(
            schema_version=1,
            configuration=config,
            asset_identity_digest="d" * 64,
            initial_state_digest="e" * 64,
            overall_status="bounded_trainability_observed",
            overfit_probe=probe,
            causal_probes=(),
            arms=(arm,),
            elapsed_seconds=10.5,
        )
        assert report.overall_status == "bounded_trainability_observed"

    def test_to_dict(
        self,
        sample_metrics: StabilityMetrics,
        sample_asset_identity: TrainabilityAssetIdentity,
        sample_implementation: ImplementationIdentity,
    ) -> None:
        """Test to_dict serialization."""
        config = TrainabilityConfiguration(
            asset_identity=sample_asset_identity,
            implementation=sample_implementation,
            stability_configuration={"test": "value"},
        )
        baseline = ArmCheckpoint(0, 0, sample_metrics)
        arm = ArmResult(
            arm="no_update",
            method=None,
            status="passed",
            baseline_checkpoint=baseline,
            checkpoints=(),
            recalibration_eligible=False,
        )
        attempt = OverfitAttempt(
            learning_rate=0.001,
            status="passed",
            baseline_metrics=sample_metrics,
            checkpoints=((64, sample_metrics),),
        )
        probe = OverfitProbeResult(status="passed", attempts=(attempt,))

        report = TrainabilityReport(
            schema_version=1,
            configuration=config,
            asset_identity_digest="d" * 64,
            initial_state_digest="e" * 64,
            overall_status="bounded_trainability_observed",
            overfit_probe=probe,
            causal_probes=(),
            arms=(arm,),
            elapsed_seconds=10.5,
        )
        result = report.to_dict()
        assert result["schema_version"] == 1
        assert result["overall_status"] == "bounded_trainability_observed"
        assert result["elapsed_seconds"] == 10.5

    def test_canonical_json(
        self,
        sample_metrics: StabilityMetrics,
        sample_asset_identity: TrainabilityAssetIdentity,
        sample_implementation: ImplementationIdentity,
    ) -> None:
        """Test canonical JSON serialization."""
        config = TrainabilityConfiguration(
            asset_identity=sample_asset_identity,
            implementation=sample_implementation,
            stability_configuration={"test": "value"},
        )
        baseline = ArmCheckpoint(0, 0, sample_metrics)
        arm = ArmResult(
            arm="no_update",
            method=None,
            status="passed",
            baseline_checkpoint=baseline,
            checkpoints=(),
            recalibration_eligible=False,
        )
        attempt = OverfitAttempt(
            learning_rate=0.001,
            status="passed",
            baseline_metrics=sample_metrics,
            checkpoints=((64, sample_metrics),),
        )
        probe = OverfitProbeResult(status="passed", attempts=(attempt,))

        report = TrainabilityReport(
            schema_version=1,
            configuration=config,
            asset_identity_digest="d" * 64,
            initial_state_digest="e" * 64,
            overall_status="bounded_trainability_observed",
            overfit_probe=probe,
            causal_probes=(),
            arms=(arm,),
            elapsed_seconds=10.5,
        )
        json_str = report.canonical_json()
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == 1
        assert "overall_status" in parsed
        assert all(math.isfinite(v) if isinstance(v, float) else True
                   for v in parsed.values()
                   if not isinstance(v, (dict, list)))


class TestCanonicalImplementationIdentity:
    """Test canonical trainability implementation identity."""

    def test_implementation_identity(self) -> None:
        """Test building implementation identity."""
        repo_root = Path(__file__).resolve().parents[1]
        identity = canonical_trainability_implementation_identity(repo_root)
        assert identity.sha256 is not None
        assert len(identity.sources) > 0
        assert all(digest and len(digest) == 64 for _, digest in identity.sources)


class TestRecordSelections:
    """Test trainability record selection builders."""

    def test_record_selections_with_real_dataset(self) -> None:
        """Test building record selections from real dataset."""
        from train.stage0_data import load_stage0_dataset
        from train.stage0_trainability import build_trainability_record_selections

        dataset = load_stage0_dataset()
        overfit_ids, training_32_ids, held_out_64_ids = (
            build_trainability_record_selections(dataset)
        )

        assert len(overfit_ids) == 1
        assert len(training_32_ids) == 32
        assert len(held_out_64_ids) == 64

        for record_id, content_hash in overfit_ids:
            assert isinstance(record_id, str)
            assert len(content_hash) == 64
            assert all(c in "0123456789abcdef" for c in content_hash)

        for record_id, content_hash in training_32_ids:
            assert isinstance(record_id, str)
            assert len(content_hash) == 64

        for record_id, content_hash in held_out_64_ids:
            assert isinstance(record_id, str)
            assert len(content_hash) == 64

    def test_record_selections_determinism(self) -> None:
        """Test that record selections are deterministic."""
        from train.stage0_data import load_stage0_dataset
        from train.stage0_trainability import build_trainability_record_selections

        dataset = load_stage0_dataset()
        overfit_ids_1, training_32_ids_1, held_out_64_ids_1 = (
            build_trainability_record_selections(dataset)
        )
        overfit_ids_2, training_32_ids_2, held_out_64_ids_2 = (
            build_trainability_record_selections(dataset)
        )

        assert overfit_ids_1 == overfit_ids_2
        assert training_32_ids_1 == training_32_ids_2
        assert held_out_64_ids_1 == held_out_64_ids_2

    def test_record_selections_no_overlap(self) -> None:
        """Test that overfit and training selections don't overlap record IDs."""
        from train.stage0_data import load_stage0_dataset
        from train.stage0_trainability import build_trainability_record_selections

        dataset = load_stage0_dataset()
        overfit_ids, training_32_ids, held_out_64_ids = (
            build_trainability_record_selections(dataset)
        )

        overfit_record_ids = {rid for rid, _ in overfit_ids}
        training_record_ids = {rid for rid, _ in training_32_ids}
        held_out_record_ids = {rid for rid, _ in held_out_64_ids}

        assert overfit_record_ids.issubset(training_record_ids)


class TestOverfitProbeClassification:
    """Test overfit probe classification logic."""

    def test_classify_passed_probe(self, sample_metrics: StabilityMetrics) -> None:
        """Test classification when an attempt passes all criteria."""
        from train.stage0_trainability import classify_overfit_probe

        passing_metrics = StabilityMetrics(
            problem_count=64,
            parameter_rms=tuple(),
            language_model_loss=0.3,
            exact_accuracy=1.0,
            first_token_accuracy=1.0,
            valid_answer_rate=1.0,
            output_diversity=0.8,
            output_dominance=0.0,
            shared_slot_variance=0.5,
            student_teacher_mse=0.0,
            student_cross_problem_cosine=0.9,
            teacher_cross_problem_cosine=0.95,
            separation_retention=0.98,
        )
        baseline_metrics = StabilityMetrics(
            problem_count=64,
            parameter_rms=tuple(),
            language_model_loss=1.0,
            exact_accuracy=0.0,
            first_token_accuracy=0.0,
            valid_answer_rate=0.5,
            output_diversity=0.5,
            output_dominance=0.5,
            shared_slot_variance=0.5,
            student_teacher_mse=1.0,
            student_cross_problem_cosine=0.1,
            teacher_cross_problem_cosine=0.2,
            separation_retention=0.5,
        )

        attempt = OverfitAttempt(
            learning_rate=0.001,
            status="passed",
            baseline_metrics=baseline_metrics,
            checkpoints=((64, passing_metrics),),
        )
        result = classify_overfit_probe((attempt,))

        assert result.status == "passed"
        assert len(result.attempts) == 1

    def test_classify_failed_probe_no_passing_attempts(
        self, sample_metrics: StabilityMetrics
    ) -> None:
        """Test classification when no attempt passes."""
        from train.stage0_trainability import classify_overfit_probe

        failed_attempt = OverfitAttempt(
            learning_rate=0.001,
            status="failed",
            baseline_metrics=sample_metrics,
            checkpoints=((64, sample_metrics),),
        )
        result = classify_overfit_probe((failed_attempt,))

        assert result.status == "objective_untrainable"
        assert len(result.attempts) == 1

    def test_classify_preserves_all_attempts(
        self, sample_metrics: StabilityMetrics
    ) -> None:
        """Test that classification preserves all completed attempts."""
        from train.stage0_trainability import classify_overfit_probe

        attempt1 = OverfitAttempt(
            learning_rate=0.0001,
            status="failed",
            baseline_metrics=sample_metrics,
        )
        attempt2 = OverfitAttempt(
            learning_rate=0.001,
            status="failed",
            baseline_metrics=sample_metrics,
        )
        attempt3 = OverfitAttempt(
            learning_rate=0.01,
            status="non_finite",
            baseline_metrics=None,
            failed_reason="NaN in gradient",
        )

        result = classify_overfit_probe((attempt1, attempt2, attempt3))

        assert result.status == "objective_untrainable"
        assert len(result.attempts) == 3
        assert result.attempts[0].learning_rate == 0.0001
        assert result.attempts[1].learning_rate == 0.001
        assert result.attempts[2].learning_rate == 0.01

    def test_classify_requires_exact_accuracy_1_0(
        self, sample_metrics: StabilityMetrics
    ) -> None:
        """Test that classification requires exact_accuracy = 1.0."""
        from train.stage0_trainability import classify_overfit_probe

        almost_passing_metrics = StabilityMetrics(
            problem_count=64,
            parameter_rms=tuple(),
            language_model_loss=0.3,
            exact_accuracy=0.99,
            first_token_accuracy=1.0,
            valid_answer_rate=1.0,
            output_diversity=0.8,
            output_dominance=0.0,
            shared_slot_variance=0.5,
            student_teacher_mse=0.0,
            student_cross_problem_cosine=0.9,
            teacher_cross_problem_cosine=0.95,
            separation_retention=0.98,
        )
        baseline_metrics = StabilityMetrics(
            problem_count=64,
            parameter_rms=tuple(),
            language_model_loss=1.0,
            exact_accuracy=0.0,
            first_token_accuracy=0.0,
            valid_answer_rate=0.5,
            output_diversity=0.5,
            output_dominance=0.5,
            shared_slot_variance=0.5,
            student_teacher_mse=1.0,
            student_cross_problem_cosine=0.1,
            teacher_cross_problem_cosine=0.2,
            separation_retention=0.5,
        )

        attempt = OverfitAttempt(
            learning_rate=0.001,
            status="passed",
            baseline_metrics=baseline_metrics,
            checkpoints=((64, almost_passing_metrics),),
        )
        result = classify_overfit_probe((attempt,))

        assert result.status == "objective_untrainable"


class TestOverfitAttemptDeterminism:
    """Test determinism properties and non-emission of recommendations."""

    def test_overfit_learning_rates_are_diagnostic_only(self) -> None:
        """Test that diagnostic learning rates are never recommended as production."""
        from train.stage0_trainability import OVERFIT_LEARNING_RATES

        for lr in OVERFIT_LEARNING_RATES:
            assert isinstance(lr, float), f"learning rate {lr} must be float"
            assert lr > 0, f"learning rate {lr} must be positive"
            assert lr in (0.0001, 0.001, 0.01), (
                f"learning rate {lr} must be diagnostic value, not production"
            )

    def test_overfit_checkpoint_structure_deterministic(self) -> None:
        """Test that checkpoint structure is deterministically defined."""
        from train.stage0_trainability import OVERFIT_CHECKPOINTS

        assert isinstance(OVERFIT_CHECKPOINTS, tuple), "checkpoints must be tuple"
        assert len(OVERFIT_CHECKPOINTS) > 0, "must have at least one checkpoint"
        assert OVERFIT_CHECKPOINTS[0] == 0, "first checkpoint must be 0 (baseline)"
        assert OVERFIT_CHECKPOINTS[-1] == 64, "final checkpoint must be 64"

        for i in range(len(OVERFIT_CHECKPOINTS) - 1):
            assert OVERFIT_CHECKPOINTS[i] < OVERFIT_CHECKPOINTS[i + 1], (
                "checkpoints must be strictly increasing"
            )

    def test_overfit_attempt_result_keys_documented(self) -> None:
        """Test that overfit attempt results have consistent documented keys."""
        expected_keys_by_status = {
            "passed": {"status", "learning_rate", "baseline_loss", "checkpoints"},
            "failed": {"status", "learning_rate", "baseline_loss", "checkpoints"},
            "non_finite": {"status", "learning_rate", "failed_reason"},
        }

        for status, expected in expected_keys_by_status.items():
            assert isinstance(expected, set), (
                f"expected keys for {status} must be set"
            )
            for key in expected:
                assert isinstance(key, str), f"key {key} must be string"

    def test_overfit_attempt_no_recommendation_fields(self) -> None:
        """Test that function signature excludes recommendation-emission capabilities."""
        from train.stage0_trainability import run_overfit_attempt
        import inspect

        sig = inspect.signature(run_overfit_attempt)
        params = list(sig.parameters.keys())

        forbidden_params = ["recommend", "recommendation", "production_lr"]
        for forbidden in forbidden_params:
            for param in params:
                assert forbidden not in param.lower(), (
                    f"parameter {param} looks like a recommendation parameter"
                )

    def test_overfit_attempt_accepts_required_parameters(self) -> None:
        """Test that function requires canonical parameters, no optional recommendation."""
        from train.stage0_trainability import run_overfit_attempt
        import inspect

        sig = inspect.signature(run_overfit_attempt)
        param_names = set(sig.parameters.keys())

        required_params = {"question", "answer", "learning_rate"}
        for required in required_params:
            assert required in param_names, (
                f"function must accept {required} parameter"
            )

    def test_overfit_attempt_documentation_excludes_recommendations(self) -> None:
        """Test that function documentation doesn't mention production recommendations."""
        from train.stage0_trainability import run_overfit_attempt

        doc = run_overfit_attempt.__doc__ or ""
        lower_doc = doc.lower()

        forbidden_phrases = ["production recommendation", "recommend", "default learning rate"]
        for forbidden in forbidden_phrases:
            assert forbidden not in lower_doc, (
                f"documentation must not mention '{forbidden}' as it's diagnostic-only"
            )

        assert "overfit" in lower_doc or "memorization" in lower_doc, (
            "documentation should describe the overfit/memorization purpose"
        )


class TestUpdateDirectionAndPrediction:
    """Test update direction capture and first-order prediction."""

    def test_update_direction_requires_finite_baseline(self) -> None:
        """Test that update direction requires finite baseline objective."""
        from train.stage0_trainability import UpdateDirection

        with pytest.raises(ValueError, match="finite"):
            UpdateDirection(
                method="gradient",
                baseline_objective=float("inf"),
                predicted_delta=-0.1,
            )

    def test_update_direction_requires_positive_baseline(self) -> None:
        """Test that baseline objective must be positive."""
        from train.stage0_trainability import UpdateDirection

        with pytest.raises(ValueError, match="positive"):
            UpdateDirection(
                method="eggroll",
                baseline_objective=-1.0,
                predicted_delta=-0.1,
            )

    def test_update_direction_requires_finite_delta(self) -> None:
        """Test that predicted delta must be finite."""
        from train.stage0_trainability import UpdateDirection

        with pytest.raises(ValueError, match="finite"):
            UpdateDirection(
                method="gradient",
                baseline_objective=1.0,
                predicted_delta=float("nan"),
            )

    def test_update_direction_valid(self) -> None:
        """Test creating valid update direction."""
        from train.stage0_trainability import UpdateDirection

        direction = UpdateDirection(
            method="gradient",
            baseline_objective=1.5,
            predicted_delta=-0.1,
        )
        assert direction.method == "gradient"
        assert direction.baseline_objective == 1.5
        assert direction.predicted_delta == -0.1

    def test_update_direction_to_dict(self) -> None:
        """Test update direction serialization."""
        from train.stage0_trainability import UpdateDirection

        direction = UpdateDirection(
            method="eggroll",
            baseline_objective=2.0,
            predicted_delta=-0.2,
        )
        result = direction.to_dict()
        assert result["method"] == "eggroll"
        assert result["baseline_objective"] == 2.0
        assert result["predicted_delta"] == -0.2

    def test_update_prediction_requires_valid_objectives(self) -> None:
        """Test that update prediction requires consistent objectives."""
        from train.stage0_trainability import UpdatePrediction

        with pytest.raises(ValueError, match="inconsistent"):
            UpdatePrediction(
                method="gradient",
                pre_update_objective=1.0,
                predicted_post_update_objective=0.5,
                predicted_delta=-0.6,
            )

    def test_update_prediction_consistent_objectives(self) -> None:
        """Test creating valid update prediction."""
        from train.stage0_trainability import UpdatePrediction

        prediction = UpdatePrediction(
            method="gradient",
            pre_update_objective=1.0,
            predicted_post_update_objective=0.9,
            predicted_delta=-0.1,
        )
        assert prediction.predicted_delta == -0.1

    def test_update_prediction_to_dict(self) -> None:
        """Test update prediction serialization."""
        from train.stage0_trainability import UpdatePrediction

        prediction = UpdatePrediction(
            method="eggroll",
            pre_update_objective=2.0,
            predicted_post_update_objective=1.8,
            predicted_delta=-0.2,
        )
        result = prediction.to_dict()
        assert result["method"] == "eggroll"
        assert result["pre_update_objective"] == 2.0

    def test_first_order_prediction_computation(self) -> None:
        """Test first-order objective change computation."""
        from train.stage0_trainability import compute_first_order_prediction

        predicted_delta = compute_first_order_prediction(
            baseline_objective=1.0,
            gradient_norm=0.5,
            step_size=0.1,
        )

        assert predicted_delta < 0, "prediction should indicate improvement"
        assert abs(predicted_delta - (-0.05)) < 1e-6, (
            "delta should be -step_size * gradient_norm"
        )

    def test_first_order_prediction_requires_positive_baseline(self) -> None:
        """Test that prediction requires positive baseline."""
        from train.stage0_trainability import compute_first_order_prediction

        with pytest.raises(ValueError, match="positive"):
            compute_first_order_prediction(
                baseline_objective=-1.0,
                gradient_norm=0.5,
                step_size=0.1,
            )

    def test_first_order_prediction_requires_non_negative_gradient(self) -> None:
        """Test that gradient norm must be non-negative."""
        from train.stage0_trainability import compute_first_order_prediction

        with pytest.raises(ValueError, match="non-negative"):
            compute_first_order_prediction(
                baseline_objective=1.0,
                gradient_norm=-0.1,
                step_size=0.1,
            )

    def test_first_order_prediction_requires_positive_step(self) -> None:
        """Test that step size must be positive."""
        from train.stage0_trainability import compute_first_order_prediction

        with pytest.raises(ValueError, match="positive"):
            compute_first_order_prediction(
                baseline_objective=1.0,
                gradient_norm=0.5,
                step_size=-0.1,
            )


class TestCausalResultClassification:
    """Test classification of causal probe results."""

    def test_classify_causal_passed_both_improve(self) -> None:
        """Test passed classification when both predicted and observed improve."""
        from train.stage0_trainability import classify_causal_result

        status, conditions = classify_causal_result(
            predicted_delta=-0.1,
            observed_delta=-0.15,
            baseline_objective=1.0,
            post_update_objective=0.85,
        )

        assert status == "passed"
        assert len(conditions) == 0

    def test_classify_causal_direction_mismatch(self) -> None:
        """Test direction_mismatch when predicted improves but observed worsens."""
        from train.stage0_trainability import classify_causal_result

        status, conditions = classify_causal_result(
            predicted_delta=-0.1,
            observed_delta=0.05,
            baseline_objective=1.0,
            post_update_objective=1.05,
        )

        assert status == "direction_mismatch"
        assert "observed_not_improvement" in conditions

    def test_classify_causal_predicted_not_improvement(self) -> None:
        """Test classification when prediction is not improvement."""
        from train.stage0_trainability import classify_causal_result

        status, conditions = classify_causal_result(
            predicted_delta=0.1,
            observed_delta=-0.05,
            baseline_objective=1.0,
            post_update_objective=0.95,
        )

        assert status == "non_finite"
        assert "predicted_not_improvement" in conditions

    def test_classify_causal_non_finite_predicted(self) -> None:
        """Test classification when predicted delta is non-finite."""
        from train.stage0_trainability import classify_causal_result

        status, conditions = classify_causal_result(
            predicted_delta=float("nan"),
            observed_delta=-0.1,
            baseline_objective=1.0,
            post_update_objective=0.9,
        )

        assert status == "non_finite"
        assert "predicted_delta_non_finite" in conditions

    def test_classify_causal_non_finite_observed(self) -> None:
        """Test classification when observed delta is non-finite."""
        from train.stage0_trainability import classify_causal_result

        status, conditions = classify_causal_result(
            predicted_delta=-0.1,
            observed_delta=float("inf"),
            baseline_objective=1.0,
            post_update_objective=0.9,
        )

        assert status == "non_finite"
        assert "observed_delta_non_finite" in conditions

    def test_classify_causal_requires_positive_baseline(self) -> None:
        """Test that classification requires positive baseline objective."""
        from train.stage0_trainability import classify_causal_result

        with pytest.raises(ValueError, match="positive"):
            classify_causal_result(
                predicted_delta=-0.1,
                observed_delta=-0.1,
                baseline_objective=-1.0,
                post_update_objective=0.9,
            )

    def test_classify_causal_requires_finite_post(self) -> None:
        """Test that post-update objective must be finite."""
        from train.stage0_trainability import classify_causal_result

        with pytest.raises(ValueError, match="finite"):
            classify_causal_result(
                predicted_delta=-0.1,
                observed_delta=-0.1,
                baseline_objective=1.0,
                post_update_objective=float("nan"),
            )


class TestCausalSafetyChecks:
    """Test safety checks for causal probe results."""

    def test_safety_checks_passed(self) -> None:
        """Test safety checks when all pass."""
        from train.stage0_trainability import validate_causal_safety_checks

        status, conditions = validate_causal_safety_checks(
            baseline_separation=0.8,
            post_update_separation=0.76,
            max_update_relative_matrix_rms=0.005,
        )

        assert status == "passed"
        assert len(conditions) == 0

    def test_safety_checks_separation_below_floor(self) -> None:
        """Test detection of separation retention below 0.95 floor."""
        from train.stage0_trainability import validate_causal_safety_checks

        status, conditions = validate_causal_safety_checks(
            baseline_separation=0.8,
            post_update_separation=0.74,
            max_update_relative_matrix_rms=0.005,
        )

        assert status == "unsafe_update"
        assert any("separation" in c for c in conditions)

    def test_safety_checks_rms_above_ceiling(self) -> None:
        """Test detection of relative matrix RMS above 0.01 ceiling."""
        from train.stage0_trainability import validate_causal_safety_checks

        status, conditions = validate_causal_safety_checks(
            baseline_separation=0.8,
            post_update_separation=0.76,
            max_update_relative_matrix_rms=0.015,
        )

        assert status == "unsafe_update"
        assert any("relative_matrix_rms" in c for c in conditions)

    def test_safety_checks_both_violations(self) -> None:
        """Test detection of multiple safety violations."""
        from train.stage0_trainability import validate_causal_safety_checks

        status, conditions = validate_causal_safety_checks(
            baseline_separation=0.8,
            post_update_separation=0.70,
            max_update_relative_matrix_rms=0.02,
        )

        assert status == "unsafe_update"
        assert len(conditions) >= 2

    def test_safety_checks_non_finite_separation(self) -> None:
        """Test detection of non-finite separation."""
        from train.stage0_trainability import validate_causal_safety_checks

        status, conditions = validate_causal_safety_checks(
            baseline_separation=0.8,
            post_update_separation=float("nan"),
            max_update_relative_matrix_rms=0.005,
        )

        assert status == "non_finite"
        assert "non_finite_separation" in conditions

    def test_safety_checks_non_finite_rms(self) -> None:
        """Test detection of non-finite RMS."""
        from train.stage0_trainability import validate_causal_safety_checks

        status, conditions = validate_causal_safety_checks(
            baseline_separation=0.8,
            post_update_separation=0.76,
            max_update_relative_matrix_rms=float("inf"),
        )

        assert status == "non_finite"
        assert "non_finite_rms" in conditions

    def test_safety_checks_requires_valid_baseline(self) -> None:
        """Test that baseline separation must be valid."""
        from train.stage0_trainability import validate_causal_safety_checks

        with pytest.raises(ValueError, match="baseline separation"):
            validate_causal_safety_checks(
                baseline_separation=float("nan"),
                post_update_separation=0.76,
                max_update_relative_matrix_rms=0.005,
            )

    def test_safety_checks_requires_positive_baseline_for_ratio(self) -> None:
        """Test that baseline must be positive for ratio calculation."""
        from train.stage0_trainability import validate_causal_safety_checks

        with pytest.raises(ValueError, match="positive"):
            validate_causal_safety_checks(
                baseline_separation=0.0,
                post_update_separation=0.0,
                max_update_relative_matrix_rms=0.005,
            )


class TestFocusedCausalProbes:
    """Focused causal probe tests with known update scenarios."""

    def test_causal_probe_aligned_update_gradient(self) -> None:
        """Test gradient method with aligned prediction and observation."""
        from train.stage0_trainability import classify_causal_result, validate_causal_safety_checks

        predicted_delta = -0.1
        observed_delta = -0.12
        status, _ = classify_causal_result(
            predicted_delta=predicted_delta,
            observed_delta=observed_delta,
            baseline_objective=1.0,
            post_update_objective=0.88,
        )
        assert status == "passed"

    def test_causal_probe_aligned_update_eggroll(self) -> None:
        """Test EGGROLL method with aligned prediction and observation."""
        from train.stage0_trainability import classify_causal_result

        status, _ = classify_causal_result(
            predicted_delta=-0.15,
            observed_delta=-0.18,
            baseline_objective=2.0,
            post_update_objective=1.82,
        )
        assert status == "passed"

    def test_causal_probe_reversed_direction_gradient(self) -> None:
        """Test gradient method with reversed prediction vs observation."""
        from train.stage0_trainability import classify_causal_result

        status, conditions = classify_causal_result(
            predicted_delta=-0.1,
            observed_delta=0.05,
            baseline_objective=1.0,
            post_update_objective=1.05,
        )
        assert status == "direction_mismatch"

    def test_causal_probe_reversed_direction_eggroll(self) -> None:
        """Test EGGROLL method with reversed direction."""
        from train.stage0_trainability import classify_causal_result

        status, _ = classify_causal_result(
            predicted_delta=-0.2,
            observed_delta=0.1,
            baseline_objective=2.0,
            post_update_objective=2.1,
        )
        assert status == "direction_mismatch"

    def test_causal_probe_oversized_rms_change(self) -> None:
        """Test detection of RMS change exceeding 0.01 ceiling."""
        from train.stage0_trainability import validate_causal_safety_checks

        status, conditions = validate_causal_safety_checks(
            baseline_separation=0.8,
            post_update_separation=0.76,
            max_update_relative_matrix_rms=0.02,
        )
        assert status == "unsafe_update"
        assert any("relative_matrix_rms" in c for c in conditions)

    def test_causal_probe_non_finite_gradient(self) -> None:
        """Test non-finite value detection in gradient probe."""
        from train.stage0_trainability import classify_causal_result

        status, _ = classify_causal_result(
            predicted_delta=float("nan"),
            observed_delta=-0.1,
            baseline_objective=1.0,
            post_update_objective=0.9,
        )
        assert status == "non_finite"

    def test_causal_probe_non_finite_eggroll(self) -> None:
        """Test non-finite value detection in EGGROLL probe."""
        from train.stage0_trainability import classify_causal_result

        status, _ = classify_causal_result(
            predicted_delta=-0.1,
            observed_delta=float("inf"),
            baseline_objective=1.0,
            post_update_objective=0.9,
        )
        assert status == "non_finite"

    def test_causal_probe_safety_with_aligned_update(self) -> None:
        """Test safety checks pass with aligned update."""
        from train.stage0_trainability import validate_causal_safety_checks

        status, conditions = validate_causal_safety_checks(
            baseline_separation=0.8,
            post_update_separation=0.77,
            max_update_relative_matrix_rms=0.005,
        )
        assert status == "passed"
        assert len(conditions) == 0


class TestCausalProbeEvaluation:
    """Test complete-objective evaluation for causal probes."""

    def test_causal_probe_evaluation_requires_8_records(
        self, sample_metrics: StabilityMetrics
    ) -> None:
        """Test that causal probe evaluation requires exactly 8 records."""
        from train.stage0_trainability import CausalProbeEvaluation

        valid_records = tuple(
            (f"Q{i}", f"A{i}") for i in range(8)
        )
        eval_result = CausalProbeEvaluation(
            training_records=valid_records,
            pre_update_metrics=sample_metrics,
        )
        assert len(eval_result.training_records) == 8

    def test_causal_probe_evaluation_rejects_wrong_count(
        self, sample_metrics: StabilityMetrics
    ) -> None:
        """Test that causal probe evaluation rejects wrong record count."""
        from train.stage0_trainability import CausalProbeEvaluation

        too_few = tuple((f"Q{i}", f"A{i}") for i in range(4))
        with pytest.raises(ValueError, match="exactly 8"):
            CausalProbeEvaluation(
                training_records=too_few,
                pre_update_metrics=sample_metrics,
            )

    def test_causal_probe_evaluation_requires_pre_update(self) -> None:
        """Test that pre-update metrics are required."""
        from train.stage0_trainability import CausalProbeEvaluation

        records = tuple((f"Q{i}", f"A{i}") for i in range(8))
        with pytest.raises(ValueError, match="pre-update"):
            CausalProbeEvaluation(
                training_records=records,
                pre_update_metrics=None,
            )

    def test_causal_probe_evaluation_optional_post_update(
        self, sample_metrics: StabilityMetrics
    ) -> None:
        """Test that post-update metrics are optional."""
        from train.stage0_trainability import CausalProbeEvaluation

        records = tuple((f"Q{i}", f"A{i}") for i in range(8))
        eval_result = CausalProbeEvaluation(
            training_records=records,
            pre_update_metrics=sample_metrics,
            post_update_metrics=None,
        )
        assert eval_result.post_update_metrics is None

    def test_causal_probe_evaluation_to_dict(
        self, sample_metrics: StabilityMetrics
    ) -> None:
        """Test causal probe evaluation serialization."""
        from train.stage0_trainability import CausalProbeEvaluation

        records = tuple((f"Q{i}", f"A{i}") for i in range(8))
        eval_result = CausalProbeEvaluation(
            training_records=records,
            pre_update_metrics=sample_metrics,
            post_update_metrics=sample_metrics,
        )
        result_dict = eval_result.to_dict()

        assert result_dict["training_record_count"] == 8
        assert "pre_update_metrics" in result_dict
        assert "post_update_metrics" in result_dict
        assert result_dict["post_update_metrics"] is not None


class TestOverfitProbeFixtures:
    """Integration tests for one-record probe with lightweight fixtures."""

    def test_overfit_probe_runs_with_simple_question(self) -> None:
        """Test that one-record probe runs successfully with simple fixture."""
        from train.stage0_trainability import run_overfit_attempt

        simple_question = "What is 1+1?"
        simple_answer = "2"

        result = run_overfit_attempt(
            simple_question,
            simple_answer,
            learning_rate=0.001,
            checkpoint_steps=(1,),
        )

        assert isinstance(result, dict)
        assert result["status"] in ("passed", "failed", "non_finite")
        assert result["learning_rate"] == 0.001

    def test_overfit_probe_fixture_returns_consistent_structure(self) -> None:
        """Test that probe results have consistent structure across different fixtures."""
        from train.stage0_trainability import run_overfit_attempt

        fixtures = [
            ("What is 2+2?", "4"),
            ("What is 5+3?", "8"),
            ("What is 10-5?", "5"),
        ]

        for question, answer in fixtures:
            result = run_overfit_attempt(
                question, answer, learning_rate=0.001, checkpoint_steps=(1,)
            )

            assert isinstance(result, dict), f"Result for {question} must be dict"
            assert "status" in result, f"Result for {question} must have status"
            assert "learning_rate" in result, f"Result for {question} must have learning_rate"
            assert result["learning_rate"] == 0.001, (
                f"Result for {question} must preserve learning_rate"
            )

    def test_overfit_probe_result_validity(self) -> None:
        """Test that probe results are well-formed and valid."""
        from train.stage0_trainability import run_overfit_attempt
        import math

        result = run_overfit_attempt(
            "What is 3+3?", "6", learning_rate=0.001, checkpoint_steps=(1, 4)
        )

        if result["status"] != "non_finite":
            if "baseline_loss" in result and result["baseline_loss"] is not None:
                assert isinstance(result["baseline_loss"], float), (
                    "baseline_loss must be float"
                )
                assert math.isfinite(result["baseline_loss"]), (
                    "baseline_loss must be finite"
                )
                assert result["baseline_loss"] >= 0, (
                    "baseline_loss must be non-negative"
                )

        if result["status"] == "passed":
            assert "checkpoints" in result, "passed result must have checkpoints"
            assert isinstance(result["checkpoints"], (list, tuple)), (
                "checkpoints must be sequence"
            )

    def test_overfit_probe_independently_runnable(self) -> None:
        """Test that probe is independently runnable without external state."""
        from train.stage0_trainability import run_overfit_attempt

        questions = ["What is 7+1?", "What is 9-4?"]
        results = []

        for question in questions:
            result = run_overfit_attempt(
                question, "8" if "7" in question else "5",
                learning_rate=0.001,
                checkpoint_steps=(1,)
            )
            results.append(result)

        assert len(results) == len(questions), "probe must run for each question"

        for i, result in enumerate(results):
            assert result["status"] in ("passed", "failed", "non_finite"), (
                f"probe {i} returned invalid status"
            )
            assert result["learning_rate"] == 0.001, (
                f"probe {i} did not preserve learning_rate"
            )


class TestMethodArms:
    """Test the three method arms for equal-budget comparison."""

    def test_fresh_state_manifest_requires_records(self) -> None:
        """Test that FreshStateManifest requires training records."""
        from train.stage0_trainability import FreshStateManifest

        with pytest.raises(ValueError):
            FreshStateManifest(training_records=())

    def test_fresh_state_manifest_requires_32_records(self) -> None:
        """Test that FreshStateManifest requires at least 32 records."""
        from train.stage0_trainability import FreshStateManifest

        records = tuple(
            (f"Question {i}", f"Answer {i}") for i in range(16)
        )
        with pytest.raises(ValueError):
            FreshStateManifest(training_records=records)

    def test_fresh_state_manifest_accepts_32_records(self) -> None:
        """Test that FreshStateManifest accepts exactly 32 records."""
        from train.stage0_trainability import FreshStateManifest

        records = tuple(
            (f"What is {i}+{i}?", str(i*2)) for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)
        assert len(manifest.training_records) == 32
        assert manifest.checkpoint_example_counts == (0, 8, 32)

    def test_fresh_state_manifest_requires_sorted_checkpoints(self) -> None:
        """Test that checkpoint counts must be sorted."""
        from train.stage0_trainability import FreshStateManifest

        records = tuple(
            (f"What is {i}+{i}?", str(i*2)) for i in range(32)
        )
        with pytest.raises(ValueError):
            FreshStateManifest(
                training_records=records,
                checkpoint_example_counts=(32, 8, 0)
            )

    def test_fresh_state_manifest_to_dict(self) -> None:
        """Test FreshStateManifest serialization."""
        from train.stage0_trainability import FreshStateManifest

        records = tuple(
            (f"What is {i}+{i}?", str(i*2)) for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)
        manifest_dict = manifest.to_dict()

        assert manifest_dict["training_record_count"] == 32
        assert manifest_dict["checkpoint_example_counts"] == [0, 8, 32]

    def test_arm_evaluation_requires_non_negative_counts(self) -> None:
        """Test that ArmEvaluation validates counts."""
        from train.stage0_trainability import ArmEvaluation
        from train.eggroll_stability import StabilityMetrics

        metrics = StabilityMetrics(
            problem_count=64,
            parameter_rms=tuple(),
            language_model_loss=0.5,
            exact_accuracy=0.0,
            first_token_accuracy=0.0,
            valid_answer_rate=1.0,
            output_diversity=0.5,
            output_dominance=0.5,
            shared_slot_variance=0.5,
            student_teacher_mse=0.5,
            student_cross_problem_cosine=0.5,
            teacher_cross_problem_cosine=0.5,
            separation_retention=0.95,
        )

        with pytest.raises(ValueError):
            ArmEvaluation(example_count=-1, metrics=metrics)

        with pytest.raises(ValueError):
            ArmEvaluation(example_count=0, metrics=metrics, examples_consumed=-1)

    def test_arm_evaluation_to_dict(self) -> None:
        """Test ArmEvaluation serialization."""
        from train.stage0_trainability import ArmEvaluation
        from train.eggroll_stability import StabilityMetrics

        metrics = StabilityMetrics(
            problem_count=64,
            parameter_rms=tuple(),
            language_model_loss=0.5,
            exact_accuracy=0.0,
            first_token_accuracy=0.0,
            valid_answer_rate=1.0,
            output_diversity=0.5,
            output_dominance=0.5,
            shared_slot_variance=0.5,
            student_teacher_mse=0.5,
            student_cross_problem_cosine=0.5,
            teacher_cross_problem_cosine=0.5,
            separation_retention=0.95,
        )

        evaluation = ArmEvaluation(
            example_count=8,
            metrics=metrics,
            examples_consumed=8,
            status="active",
        )
        eval_dict = evaluation.to_dict()

        assert eval_dict["example_count"] == 8
        assert eval_dict["examples_consumed"] == 8
        assert eval_dict["status"] == "active"
        assert "metrics" in eval_dict

    def test_method_arm_requires_training_records(self) -> None:
        """Test that MethodArm requires training records."""
        from train.stage0_trainability import MethodArm

        with pytest.raises(ValueError):
            MethodArm(
                arm_kind="no_update",
                training_records=(),
            )

    def test_method_arm_requires_32_records(self) -> None:
        """Test that MethodArm requires at least 32 records."""
        from train.stage0_trainability import MethodArm

        records = tuple(
            (f"Question {i}", f"Answer {i}") for i in range(16)
        )
        with pytest.raises(ValueError):
            MethodArm(
                arm_kind="gradient_only",
                training_records=records,
            )

    def test_method_arm_accepts_32_records(self) -> None:
        """Test that MethodArm accepts exactly 32 records."""
        from train.stage0_trainability import MethodArm

        records = tuple(
            (f"What is {i}+{i}?", str(i*2)) for i in range(32)
        )
        arm = MethodArm(
            arm_kind="eggroll_only",
            training_records=records,
        )
        assert len(arm.training_records) == 32
        assert arm.arm_kind == "eggroll_only"
        assert arm.final_status == "active"

    def test_method_arm_to_dict(self) -> None:
        """Test MethodArm serialization."""
        from train.stage0_trainability import MethodArm

        records = tuple(
            (f"What is {i}+{i}?", str(i*2)) for i in range(32)
        )
        arm = MethodArm(
            arm_kind="gradient_only",
            training_records=records,
            evaluations=(),
        )
        arm_dict = arm.to_dict()

        assert arm_dict["arm_kind"] == "gradient_only"
        assert arm_dict["training_record_count"] == 32
        assert arm_dict["evaluation_count"] == 0
        assert arm_dict["final_status"] == "active"
        assert isinstance(arm_dict["evaluations"], list)

    def test_consistent_example_counting_across_checkpoints(self) -> None:
        """Test that example consumption is consistent across checkpoints."""
        from train.stage0_trainability import ArmEvaluation, MethodArm
        from train.eggroll_stability import StabilityMetrics

        metrics = StabilityMetrics(
            problem_count=64,
            parameter_rms=tuple(),
            language_model_loss=0.5,
            exact_accuracy=0.0,
            first_token_accuracy=0.0,
            valid_answer_rate=1.0,
            output_diversity=0.5,
            output_dominance=0.5,
            shared_slot_variance=0.5,
            student_teacher_mse=0.5,
            student_cross_problem_cosine=0.5,
            teacher_cross_problem_cosine=0.5,
            separation_retention=0.95,
        )

        records = tuple(
            (f"What is {i}+{i}?", str(i*2)) for i in range(32)
        )

        eval0 = ArmEvaluation(example_count=0, metrics=metrics, examples_consumed=0)
        eval8 = ArmEvaluation(example_count=8, metrics=metrics, examples_consumed=8)
        eval32 = ArmEvaluation(
            example_count=32, metrics=metrics, examples_consumed=32
        )

        arm = MethodArm(
            arm_kind="gradient_only",
            training_records=records,
            evaluations=(eval0, eval8, eval32),
        )

        assert len(arm.evaluations) == 3
        assert arm.evaluations[0].examples_consumed == 0
        assert arm.evaluations[1].examples_consumed == 8
        assert arm.evaluations[2].examples_consumed == 32
        assert (
            arm.evaluations[1].examples_consumed
            > arm.evaluations[0].examples_consumed
        )
        assert (
            arm.evaluations[2].examples_consumed
            > arm.evaluations[1].examples_consumed
        )

    def test_evaluations_at_checkpoint_boundaries(self) -> None:
        """Test that evaluations are emitted at exactly 0, 8, 32 checkpoints."""
        from train.stage0_trainability import FreshStateManifest

        records = tuple(
            (f"What is {i}+{i}?", str(i*2)) for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)

        assert manifest.checkpoint_example_counts == (0, 8, 32)
        expected_evals = 3
        assert len(manifest.checkpoint_example_counts) == expected_evals

    def test_no_update_arm_consistent_counting(self) -> None:
        """Test that no-update arm counts examples consistently."""
        from train.stage0_trainability import FreshStateManifest, run_no_update_arm

        records = tuple(
            (f"What is {i}+{i}?", str(i*2)) for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)

        arm = run_no_update_arm(manifest)

        assert arm.arm_kind == "no_update"
        if len(arm.evaluations) >= 1:
            assert arm.evaluations[0].examples_consumed == 0
        if len(arm.evaluations) >= 2:
            assert arm.evaluations[1].example_count == 8
            assert arm.evaluations[1].examples_consumed >= 0
        if len(arm.evaluations) >= 3:
            assert arm.evaluations[2].example_count == 32
            assert arm.evaluations[2].examples_consumed >= 8

    def test_gradient_arm_consistent_counting(self) -> None:
        """Test that gradient arm counts examples consistently."""
        from train.stage0_trainability import FreshStateManifest, run_gradient_only_arm

        records = tuple(
            (f"What is {i}+{i}?", str(i*2)) for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)

        arm = run_gradient_only_arm(manifest)

        assert arm.arm_kind == "gradient_only"
        if len(arm.evaluations) >= 1:
            assert arm.evaluations[0].examples_consumed == 0
        if len(arm.evaluations) >= 2:
            assert arm.evaluations[1].example_count == 8
        if len(arm.evaluations) >= 3:
            assert arm.evaluations[2].example_count == 32

    def test_eggroll_arm_consistent_counting(self) -> None:
        """Test that EGGROLL arm counts examples consistently."""
        from train.stage0_trainability import FreshStateManifest, run_eggroll_only_arm

        records = tuple(
            (f"What is {i}+{i}?", str(i*2)) for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)

        arm = run_eggroll_only_arm(manifest)

        assert arm.arm_kind == "eggroll_only"
        if len(arm.evaluations) >= 1:
            assert arm.evaluations[0].examples_consumed == 0
        if len(arm.evaluations) >= 2:
            assert arm.evaluations[1].example_count == 8
        if len(arm.evaluations) >= 3:
            assert arm.evaluations[2].example_count == 32


class TestNoUpdateControl:
    """Test the no-update control arm for state drift detection."""

    def test_no_update_control_requires_drift_data(self) -> None:
        """Test that NoUpdateControl requires metrics drift."""
        from train.stage0_trainability import NoUpdateControl
        from train.eggroll_stability import StabilityMetrics

        metrics = StabilityMetrics(
            problem_count=64,
            parameter_rms=tuple(),
            language_model_loss=0.5,
            exact_accuracy=0.0,
            first_token_accuracy=0.0,
            valid_answer_rate=1.0,
            output_diversity=0.5,
            output_dominance=0.5,
            shared_slot_variance=0.5,
            student_teacher_mse=0.5,
            student_cross_problem_cosine=0.5,
            teacher_cross_problem_cosine=0.5,
            separation_retention=0.95,
        )

        with pytest.raises(ValueError):
            NoUpdateControl(
                baseline_metrics=metrics,
                final_metrics=metrics,
                metrics_drift={},
            )

    def test_no_update_control_accepts_drift_data(self) -> None:
        """Test that NoUpdateControl accepts valid drift data."""
        from train.stage0_trainability import NoUpdateControl
        from train.eggroll_stability import StabilityMetrics

        metrics = StabilityMetrics(
            problem_count=64,
            parameter_rms=tuple(),
            language_model_loss=0.5,
            exact_accuracy=0.0,
            first_token_accuracy=0.0,
            valid_answer_rate=1.0,
            output_diversity=0.5,
            output_dominance=0.5,
            shared_slot_variance=0.5,
            student_teacher_mse=0.5,
            student_cross_problem_cosine=0.5,
            teacher_cross_problem_cosine=0.5,
            separation_retention=0.95,
        )

        control = NoUpdateControl(
            baseline_metrics=metrics,
            final_metrics=metrics,
            metrics_drift={"lm_loss_ratio": 1.0},
        )
        assert control.metrics_drift["lm_loss_ratio"] == 1.0

    def test_no_update_control_to_dict(self) -> None:
        """Test NoUpdateControl serialization."""
        from train.stage0_trainability import NoUpdateControl
        from train.eggroll_stability import StabilityMetrics

        metrics = StabilityMetrics(
            problem_count=64,
            parameter_rms=tuple(),
            language_model_loss=0.5,
            exact_accuracy=0.0,
            first_token_accuracy=0.0,
            valid_answer_rate=1.0,
            output_diversity=0.5,
            output_dominance=0.5,
            shared_slot_variance=0.5,
            student_teacher_mse=0.5,
            student_cross_problem_cosine=0.5,
            teacher_cross_problem_cosine=0.5,
            separation_retention=0.95,
        )

        control = NoUpdateControl(
            baseline_metrics=metrics,
            final_metrics=metrics,
            metrics_drift={
                "lm_loss_ratio": 1.0,
                "accuracy_delta": 0.0,
            },
        )
        control_dict = control.to_dict()

        assert "baseline_metrics" in control_dict
        assert "final_metrics" in control_dict
        assert "metrics_drift" in control_dict
        assert control_dict["metrics_drift"]["lm_loss_ratio"] == 1.0

    def test_compute_metrics_drift(self) -> None:
        """Test metrics drift computation."""
        from train.stage0_trainability import _compute_metrics_drift
        from train.eggroll_stability import StabilityMetrics

        baseline = StabilityMetrics(
            problem_count=64,
            parameter_rms=tuple(),
            language_model_loss=1.0,
            exact_accuracy=0.0,
            first_token_accuracy=0.0,
            valid_answer_rate=1.0,
            output_diversity=0.5,
            output_dominance=0.5,
            shared_slot_variance=0.5,
            student_teacher_mse=1.0,
            student_cross_problem_cosine=0.5,
            teacher_cross_problem_cosine=0.5,
            separation_retention=1.0,
        )

        final = StabilityMetrics(
            problem_count=64,
            parameter_rms=tuple(),
            language_model_loss=1.0,
            exact_accuracy=0.0,
            first_token_accuracy=0.0,
            valid_answer_rate=1.0,
            output_diversity=0.5,
            output_dominance=0.5,
            shared_slot_variance=0.5,
            student_teacher_mse=1.0,
            student_cross_problem_cosine=0.5,
            teacher_cross_problem_cosine=0.5,
            separation_retention=1.0,
        )

        drift = _compute_metrics_drift(baseline, final)

        assert "lm_loss_ratio" in drift
        assert drift["lm_loss_ratio"] == 1.0
        assert "exact_accuracy_delta" in drift
        assert drift["exact_accuracy_delta"] == 0.0
        assert "separation_retention_ratio" in drift
        assert drift["separation_retention_ratio"] == 1.0


class TestArmSafetyChecks:
    """Test independent safety checks for arm evaluations."""

    def test_arm_safety_check_passes_valid_metrics(self) -> None:
        """Test that safety check passes for valid metrics."""
        from train.stage0_trainability import check_arm_evaluation_safety
        from train.eggroll_stability import StabilityMetrics

        metrics = StabilityMetrics(
            problem_count=64,
            parameter_rms=tuple(),
            language_model_loss=0.5,
            exact_accuracy=0.8,
            first_token_accuracy=0.9,
            valid_answer_rate=1.0,
            output_diversity=0.5,
            output_dominance=0.5,
            shared_slot_variance=0.5,
            student_teacher_mse=0.5,
            student_cross_problem_cosine=0.5,
            teacher_cross_problem_cosine=0.5,
            separation_retention=0.95,
        )

        check = check_arm_evaluation_safety(metrics)

        assert check.status == "passed"
        assert not check.non_finite_detected
        assert not check.rms_ceiling_violated
        assert not check.separation_floor_violated

    def test_arm_safety_check_stops_separation_violation(self) -> None:
        """Test that safety check stops on separation floor violation."""
        from train.stage0_trainability import check_arm_evaluation_safety
        from train.eggroll_stability import StabilityMetrics

        metrics = StabilityMetrics(
            problem_count=64,
            parameter_rms=tuple(),
            language_model_loss=0.5,
            exact_accuracy=0.8,
            first_token_accuracy=0.9,
            valid_answer_rate=1.0,
            output_diversity=0.5,
            output_dominance=0.5,
            shared_slot_variance=0.5,
            student_teacher_mse=0.5,
            student_cross_problem_cosine=0.5,
            teacher_cross_problem_cosine=0.5,
            separation_retention=0.9,
        )

        check = check_arm_evaluation_safety(metrics, baseline_separation=1.0)

        assert check.status == "stopped"
        assert check.separation_floor_violated
        assert "Separation retention" in check.stop_reason

    def test_arm_safety_check_stops_rms_ceiling_violation(self) -> None:
        """Test that safety check stops on RMS ceiling violation."""
        from train.stage0_trainability import check_arm_evaluation_safety
        from train.eggroll_stability import StabilityMetrics, ParameterRms

        metrics = StabilityMetrics(
            problem_count=64,
            parameter_rms=(
                ParameterRms(path="layer1", rms=0.001),
                ParameterRms(path="layer2", rms=0.02),
            ),
            language_model_loss=0.5,
            exact_accuracy=0.8,
            first_token_accuracy=0.9,
            valid_answer_rate=1.0,
            output_diversity=0.5,
            output_dominance=0.5,
            shared_slot_variance=0.5,
            student_teacher_mse=0.5,
            student_cross_problem_cosine=0.5,
            teacher_cross_problem_cosine=0.5,
            separation_retention=0.95,
        )

        check = check_arm_evaluation_safety(metrics)

        assert check.status == "stopped"
        assert check.rms_ceiling_violated
        assert "RMS" in check.stop_reason

    def test_arm_safety_check_to_dict(self) -> None:
        """Test ArmSafetyCheck serialization."""
        from train.stage0_trainability import ArmSafetyCheck

        check = ArmSafetyCheck(
            status="stopped",
            stop_reason="Test reason",
            non_finite_detected=True,
        )
        check_dict = check.to_dict()

        assert check_dict["status"] == "stopped"
        assert check_dict["stop_reason"] == "Test reason"
        assert check_dict["non_finite_detected"] is True
        assert check_dict["rms_ceiling_violated"] is False


class TestArmIntegration:
    """Integration tests for equal-budget arm execution and isolation."""

    def test_identical_record_exposure_across_arms(self) -> None:
        """Test that all arms see the same records in the same order."""
        from train.stage0_trainability import FreshStateManifest

        records = tuple(
            (f"What is {i}+{i}?", str(i*2)) for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)

        assert len(manifest.training_records) == 32
        for i, (question, answer) in enumerate(manifest.training_records):
            assert question == f"What is {i}+{i}?"
            assert answer == str(i*2)

    def test_checkpoint_boundary_at_8_and_32(self) -> None:
        """Test that checkpoints are exactly at 8 and 32 records."""
        from train.stage0_trainability import FreshStateManifest

        records = tuple(
            (f"Question {i}", f"Answer {i}") for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)

        assert 0 in manifest.checkpoint_example_counts
        assert 8 in manifest.checkpoint_example_counts
        assert 32 in manifest.checkpoint_example_counts
        assert len(manifest.checkpoint_example_counts) == 3

    def test_arm_isolation_independent_state(self) -> None:
        """Test that arm state doesn't leak between arms."""
        from train.stage0_trainability import (
            FreshStateManifest,
            run_no_update_arm,
            run_gradient_only_arm,
            run_eggroll_only_arm,
        )

        records = tuple(
            (f"What is {i}+{i}?", str(i*2)) for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)

        no_update_arm = run_no_update_arm(manifest)
        gradient_arm = run_gradient_only_arm(manifest)
        eggroll_arm = run_eggroll_only_arm(manifest)

        assert no_update_arm.arm_kind == "no_update"
        assert gradient_arm.arm_kind == "gradient_only"
        assert eggroll_arm.arm_kind == "eggroll_only"

        assert len(no_update_arm.training_records) == 32
        assert len(gradient_arm.training_records) == 32
        assert len(eggroll_arm.training_records) == 32

        assert (
            no_update_arm.training_records == gradient_arm.training_records
        )
        assert (
            gradient_arm.training_records == eggroll_arm.training_records
        )

    def test_consistent_example_counting_across_arms(self) -> None:
        """Test that all arms count examples identically."""
        from train.stage0_trainability import (
            FreshStateManifest,
            run_no_update_arm,
            run_gradient_only_arm,
        )

        records = tuple(
            (f"What is {i}+{i}?", str(i*2)) for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)

        no_update_arm = run_no_update_arm(manifest)
        gradient_arm = run_gradient_only_arm(manifest)

        if len(no_update_arm.evaluations) > 0:
            assert no_update_arm.evaluations[0].examples_consumed == 0
        if len(gradient_arm.evaluations) > 0:
            assert gradient_arm.evaluations[0].examples_consumed == 0

        if len(no_update_arm.evaluations) > 1:
            assert no_update_arm.evaluations[1].example_count == 8
        if len(gradient_arm.evaluations) > 1:
            assert gradient_arm.evaluations[1].example_count == 8

    def test_arm_independent_completion_status(self) -> None:
        """Test that arms complete independently."""
        from train.stage0_trainability import (
            FreshStateManifest,
            run_no_update_arm,
            run_gradient_only_arm,
            run_eggroll_only_arm,
        )

        records = tuple(
            (f"Question {i}", f"Answer {i}") for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)

        no_update_arm = run_no_update_arm(manifest)
        gradient_arm = run_gradient_only_arm(manifest)
        eggroll_arm = run_eggroll_only_arm(manifest)

        assert no_update_arm.final_status in ("active", "stopped")
        assert gradient_arm.final_status in ("active", "stopped")
        assert eggroll_arm.final_status in ("active", "stopped")

    def test_identical_record_order_validation(self) -> None:
        """Test that record order is canonical and deterministic."""
        from train.stage0_trainability import FreshStateManifest

        records1 = tuple(
            (f"Q{i}", f"A{i}") for i in range(32)
        )
        records2 = tuple(
            (f"Q{i}", f"A{i}") for i in range(32)
        )

        manifest1 = FreshStateManifest(training_records=records1)
        manifest2 = FreshStateManifest(training_records=records2)

        for i in range(32):
            assert (
                manifest1.training_records[i]
                == manifest2.training_records[i]
            )

    def test_arm_evaluation_counts_preserve_order(self) -> None:
        """Test that arm evaluations maintain checkpoint order."""
        from train.stage0_trainability import (
            FreshStateManifest,
            run_no_update_arm,
        )

        records = tuple(
            (f"Q{i}", f"A{i}") for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)

        arm = run_no_update_arm(manifest)

        if len(arm.evaluations) >= 2:
            assert (
                arm.evaluations[0].example_count
                <= arm.evaluations[1].example_count
            )
        if len(arm.evaluations) >= 3:
            assert (
                arm.evaluations[1].example_count
                <= arm.evaluations[2].example_count
            )


class TestArmFixtures:
    """Method-comparison fixtures for independent arm verification."""

    def test_arms_with_simple_math_questions(self) -> None:
        """Test all three arms with simple math question fixtures."""
        from train.stage0_trainability import (
            FreshStateManifest,
            run_no_update_arm,
            run_gradient_only_arm,
            run_eggroll_only_arm,
        )

        records = tuple(
            (f"What is {i}+{i}?", str(i*2)) for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)

        no_update_arm = run_no_update_arm(manifest)
        gradient_arm = run_gradient_only_arm(manifest)
        eggroll_arm = run_eggroll_only_arm(manifest)

        assert no_update_arm.arm_kind == "no_update"
        assert gradient_arm.arm_kind == "gradient_only"
        assert eggroll_arm.arm_kind == "eggroll_only"

        for arm in [no_update_arm, gradient_arm, eggroll_arm]:
            assert len(arm.evaluations) >= 0
            assert arm.final_status in ("active", "stopped")
            assert len(arm.training_records) == 32

    def test_arms_have_independent_results(self) -> None:
        """Test that arms produce independent result objects."""
        from train.stage0_trainability import (
            FreshStateManifest,
            run_no_update_arm,
            run_gradient_only_arm,
            run_eggroll_only_arm,
        )

        records = tuple(
            (f"Question {i}", f"Answer {i}") for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)

        no_update_result = run_no_update_arm(manifest)
        gradient_result = run_gradient_only_arm(manifest)
        eggroll_result = run_eggroll_only_arm(manifest)

        assert no_update_result is not gradient_result
        assert gradient_result is not eggroll_result
        assert no_update_result is not eggroll_result

    def test_all_arms_runnable_without_exceptions(self) -> None:
        """Test that all three arms complete without exceptions."""
        from train.stage0_trainability import (
            FreshStateManifest,
            run_no_update_arm,
            run_gradient_only_arm,
            run_eggroll_only_arm,
        )

        records = tuple(
            (f"Q{i}", f"A{i}") for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)

        try:
            no_update_arm = run_no_update_arm(manifest)
            assert no_update_arm is not None
        except Exception:
            pytest.fail("no-update arm raised exception")

        try:
            gradient_arm = run_gradient_only_arm(manifest)
            assert gradient_arm is not None
        except Exception:
            pytest.fail("gradient arm raised exception")

        try:
            eggroll_arm = run_eggroll_only_arm(manifest)
            assert eggroll_arm is not None
        except Exception:
            pytest.fail("EGGROLL arm raised exception")

    def test_arms_produce_valid_arm_objects(self) -> None:
        """Test that all arms produce valid MethodArm objects."""
        from train.stage0_trainability import (
            FreshStateManifest,
            run_no_update_arm,
            run_gradient_only_arm,
            run_eggroll_only_arm,
            MethodArm,
        )

        records = tuple(
            (f"Math{i}", f"{i*2}") for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)

        arms = [
            run_no_update_arm(manifest),
            run_gradient_only_arm(manifest),
            run_eggroll_only_arm(manifest),
        ]

        for arm in arms:
            assert isinstance(arm, MethodArm)
            assert arm.arm_kind in ("no_update", "gradient_only", "eggroll_only")
            assert len(arm.training_records) == 32
            assert arm.final_status in ("active", "stopped")

    def test_arms_with_diverse_fixtures(self) -> None:
        """Test arms with diverse question/answer pairs."""
        from train.stage0_trainability import (
            FreshStateManifest,
            run_no_update_arm,
            run_gradient_only_arm,
        )

        records = tuple([
            ("What is 2+2?", "4"),
            ("What is 5+3?", "8"),
            ("What is 10-5?", "5"),
            ("What is 3*3?", "9"),
            ("What is 12/3?", "4"),
        ] + [
            (f"Math {i}", f"Result {i}") for i in range(27)
        ])
        manifest = FreshStateManifest(training_records=records)

        no_update_arm = run_no_update_arm(manifest)
        gradient_arm = run_gradient_only_arm(manifest)

        assert len(no_update_arm.training_records) == 32
        assert len(gradient_arm.training_records) == 32

        for i in range(5):
            assert no_update_arm.training_records[i] == records[i]
            assert gradient_arm.training_records[i] == records[i]

    def test_arms_independently_runnable(self) -> None:
        """Test that each arm can run independently in sequence."""
        from train.stage0_trainability import (
            FreshStateManifest,
            run_no_update_arm,
            run_gradient_only_arm,
            run_eggroll_only_arm,
        )

        records = tuple(
            (f"Independent{i}", f"Test{i}") for i in range(32)
        )
        manifest = FreshStateManifest(training_records=records)

        arms_results = []
        for run_fn, expected_kind in [
            (run_no_update_arm, "no_update"),
            (run_gradient_only_arm, "gradient_only"),
            (run_eggroll_only_arm, "eggroll_only"),
        ]:
            arm = run_fn(manifest)
            assert arm.arm_kind == expected_kind
            arms_results.append(arm)

        assert len(arms_results) == 3
        assert all(arm.final_status in ("active", "stopped") for arm in arms_results)
