"""Comprehensive identity and report parser tests for trainability investigation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from train.stage0_trainability import (
    TrainabilityReportValidationError,
    canonical_trainability_implementation_identity,
    load_and_validate_stability_report,
)


class TestTrainabilityImplementationIdentity:
    """Test canonical trainability implementation identity contract."""

    def test_implementation_identity_structure(self) -> None:
        """Verify implementation identity has correct structure."""
        repo_root = Path(__file__).resolve().parents[1]
        identity = canonical_trainability_implementation_identity(repo_root)

        assert identity.sha256 is not None
        assert len(identity.sha256) == 64
        assert all(c in "0123456789abcdef" for c in identity.sha256)
        assert len(identity.sources) > 0

    def test_implementation_identity_sources_ordered(self) -> None:
        """Verify implementation identity sources are in canonical order."""
        repo_root = Path(__file__).resolve().parents[1]
        identity = canonical_trainability_implementation_identity(repo_root)

        source_paths = [path for path, _ in identity.sources]
        assert source_paths == sorted(source_paths), "implementation sources must be in canonical sorted order"

    def test_implementation_identity_all_digests_valid(self) -> None:
        """Verify all source digests are valid SHA-256."""
        repo_root = Path(__file__).resolve().parents[1]
        identity = canonical_trainability_implementation_identity(repo_root)

        for path, digest in identity.sources:
            assert len(digest) == 64, f"digest for {path} must be 64 hex chars"
            assert all(c in "0123456789abcdef" for c in digest), f"digest for {path} must be valid hex"

    def test_implementation_identity_includes_trainability_module(self) -> None:
        """Verify implementation identity includes stage0_trainability.py."""
        repo_root = Path(__file__).resolve().parents[1]
        identity = canonical_trainability_implementation_identity(repo_root)

        source_paths = [path for path, _ in identity.sources]
        assert "train/stage0_trainability.py" in source_paths

    def test_implementation_identity_deterministic(self) -> None:
        """Verify implementation identity is deterministic."""
        repo_root = Path(__file__).resolve().parents[1]
        identity1 = canonical_trainability_implementation_identity(repo_root)
        identity2 = canonical_trainability_implementation_identity(repo_root)

        assert identity1.sha256 == identity2.sha256
        assert identity1.sources == identity2.sources


class TestStabilityReportLoader:
    """Test diagnostic stability report loader contract."""

    def test_loader_rejects_missing_report(self) -> None:
        """Verify loader rejects missing report file."""
        repo_root = Path(__file__).resolve().parents[1]
        implementation = canonical_trainability_implementation_identity(repo_root)
        missing_path = Path("/nonexistent/report.json")

        with pytest.raises(TrainabilityReportValidationError) as exc_info:
            load_and_validate_stability_report(
                missing_path,
                implementation,
                (),
            )

        assert "not found" in str(exc_info.value)

    def test_loader_rejects_malformed_json(self) -> None:
        """Verify loader rejects malformed JSON."""
        repo_root = Path(__file__).resolve().parents[1]
        implementation = canonical_trainability_implementation_identity(repo_root)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json")
            temp_path = Path(f.name)

        try:
            with pytest.raises(TrainabilityReportValidationError) as exc_info:
                load_and_validate_stability_report(
                    temp_path,
                    implementation,
                    (),
                )

            assert "malformed" in str(exc_info.value).lower()
        finally:
            temp_path.unlink()

    def test_loader_rejects_passing_report(self) -> None:
        """Verify loader rejects passing report (not failed)."""
        repo_root = Path(__file__).resolve().parents[1]
        implementation = canonical_trainability_implementation_identity(repo_root)

        report_data = {
            "status": "passed",
            "configuration": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report_data, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(TrainabilityReportValidationError) as exc_info:
                load_and_validate_stability_report(
                    temp_path,
                    implementation,
                    (),
                )

            error_msg = str(exc_info.value)
            assert "status=failed" in error_msg or "passed" in error_msg.lower()
        finally:
            temp_path.unlink()

    def test_loader_accepts_failed_report_with_matching_implementation(
        self,
    ) -> None:
        """Verify loader accepts failed report with matching implementation."""
        repo_root = Path(__file__).resolve().parents[1]
        implementation = canonical_trainability_implementation_identity(repo_root)

        report_data = {
            "status": "failed",
            "configuration": {
                "implementation": {
                    "sha256": implementation.sha256,
                },
                "asset_identity": {
                    "held_out_problem_count": 64,
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report_data, f)
            temp_path = Path(f.name)

        try:
            held_out_ids: tuple[tuple[str, str], ...] = (("test_001", "id_001"),) * 64
            digest, config = load_and_validate_stability_report(
                temp_path,
                implementation,
                held_out_ids,
            )

            assert len(digest) == 64
            assert all(c in "0123456789abcdef" for c in digest)
            assert config is not None
            assert config["implementation"]["sha256"] == implementation.sha256
        finally:
            temp_path.unlink()

    def test_loader_rejects_implementation_mismatch(self) -> None:
        """Verify loader rejects report with different implementation."""
        repo_root = Path(__file__).resolve().parents[1]
        implementation = canonical_trainability_implementation_identity(repo_root)

        different_sha = "a" * 64
        assert different_sha != implementation.sha256

        report_data = {
            "status": "failed",
            "configuration": {
                "implementation": {
                    "sha256": different_sha,
                },
                "asset_identity": {
                    "held_out_problem_count": 64,
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report_data, f)
            temp_path = Path(f.name)

        try:
            held_out_ids: tuple[tuple[str, str], ...] = (("test_001", "id_001"),) * 64

            with pytest.raises(TrainabilityReportValidationError) as exc_info:
                load_and_validate_stability_report(
                    temp_path,
                    implementation,
                    held_out_ids,
                )

            assert "mismatch" in str(exc_info.value).lower()
        finally:
            temp_path.unlink()

    def test_loader_rejects_record_count_mismatch(self) -> None:
        """Verify loader rejects report with different record count."""
        repo_root = Path(__file__).resolve().parents[1]
        implementation = canonical_trainability_implementation_identity(repo_root)

        report_data = {
            "status": "failed",
            "configuration": {
                "implementation": {
                    "sha256": implementation.sha256,
                },
                "asset_identity": {
                    "held_out_problem_count": 100,
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report_data, f)
            temp_path = Path(f.name)

        try:
            held_out_ids: tuple[tuple[str, str], ...] = (("test_001", "id_001"),) * 64

            with pytest.raises(TrainabilityReportValidationError) as exc_info:
                load_and_validate_stability_report(
                    temp_path,
                    implementation,
                    held_out_ids,
                )

            assert "count" in str(exc_info.value).lower()
        finally:
            temp_path.unlink()


class TestReportDigestConsistency:
    """Test report digest consistency across multiple loads."""

    def test_report_digest_deterministic(self) -> None:
        """Verify report digest is deterministic."""
        repo_root = Path(__file__).resolve().parents[1]
        implementation = canonical_trainability_implementation_identity(repo_root)

        report_data = {
            "status": "failed",
            "configuration": {
                "implementation": {
                    "sha256": implementation.sha256,
                },
                "asset_identity": {
                    "held_out_problem_count": 64,
                },
            },
        }

        held_out_ids: tuple[tuple[str, str], ...] = (("test_001", "id_001"),) * 64

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report_data, f)
            temp_path = Path(f.name)

        try:
            digest1, _ = load_and_validate_stability_report(
                temp_path,
                implementation,
                held_out_ids,
            )
            digest2, _ = load_and_validate_stability_report(
                temp_path,
                implementation,
                held_out_ids,
            )

            assert digest1 == digest2
        finally:
            temp_path.unlink()

    def test_report_digest_differs_on_content_change(self) -> None:
        """Verify report digest changes when content changes."""
        repo_root = Path(__file__).resolve().parents[1]
        implementation = canonical_trainability_implementation_identity(repo_root)

        report_data1 = {
            "status": "failed",
            "configuration": {
                "implementation": {
                    "sha256": implementation.sha256,
                },
                "asset_identity": {
                    "held_out_problem_count": 64,
                },
            },
            "extra_field_1": "value1",
        }

        report_data2 = {
            "status": "failed",
            "configuration": {
                "implementation": {
                    "sha256": implementation.sha256,
                },
                "asset_identity": {
                    "held_out_problem_count": 64,
                },
            },
            "extra_field_1": "value2",
        }

        held_out_ids: tuple[tuple[str, str], ...] = (("test_001", "id_001"),) * 64

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report_data1, f)
            temp_path1 = Path(f.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report_data2, f)
            temp_path2 = Path(f.name)

        try:
            digest1, _ = load_and_validate_stability_report(
                temp_path1,
                implementation,
                held_out_ids,
            )
            digest2, _ = load_and_validate_stability_report(
                temp_path2,
                implementation,
                held_out_ids,
            )

            assert digest1 != digest2
        finally:
            temp_path1.unlink()
            temp_path2.unlink()
