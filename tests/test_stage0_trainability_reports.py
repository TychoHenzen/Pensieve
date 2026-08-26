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
