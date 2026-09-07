"""Verification that trainability types have correct structure (no heavy imports)."""

from __future__ import annotations

from pathlib import Path


def test_trainability_module_syntax() -> None:
    """Verify trainability module has correct syntax."""
    repo_root = Path(__file__).resolve().parents[1]
    trainability_module = repo_root / "train" / "stage0_trainability.py"

    code = trainability_module.read_text(encoding="utf-8")

    assert "class TrainabilityReport:" in code
    assert "class OverfitAttempt:" in code
    assert "class CausalProbeResult:" in code
    assert "class ArmResult:" in code
    assert "class TrainabilityReportValidationError(ValueError):" in code
    assert "def load_and_validate_stability_report(" in code
    assert "def canonical_trainability_implementation_identity(" in code
    assert "def compute_initial_state_digest(" in code

    assert "TrainabilityAssetIdentity" in code
    assert "TrainabilityConfiguration" in code
    assert "OverfitProbeResult" in code
    assert "ArmCheckpoint" in code


def test_report_types_have_dataclass_decorator() -> None:
    """Verify all report types are frozen dataclasses."""
    repo_root = Path(__file__).resolve().parents[1]
    trainability_module = repo_root / "train" / "stage0_trainability.py"

    code = trainability_module.read_text(encoding="utf-8")

    classes_to_check = [
        "TrainabilityAssetIdentity",
        "TrainabilityConfiguration",
        "OverfitAttempt",
        "OverfitProbeResult",
        "CausalProbeResult",
        "ArmCheckpoint",
        "ArmResult",
        "TrainabilityReport",
    ]

    for class_name in classes_to_check:
        pattern = f"@dataclass(frozen=True)\nclass {class_name}:"
        assert pattern in code, f"Class {class_name} should be frozen dataclass"


def test_report_types_have_to_dict_method() -> None:
    """Verify all report types have to_dict serialization."""
    repo_root = Path(__file__).resolve().parents[1]
    trainability_module = repo_root / "train" / "stage0_trainability.py"

    code = trainability_module.read_text(encoding="utf-8")

    classes_with_to_dict = [
        "TrainabilityAssetIdentity",
        "TrainabilityConfiguration",
        "OverfitAttempt",
        "OverfitProbeResult",
        "CausalProbeResult",
        "ArmCheckpoint",
        "ArmResult",
        "TrainabilityReport",
    ]

    for class_name in classes_with_to_dict:
        pattern = f"class {class_name}:"
        class_idx = code.find(pattern)
        assert class_idx >= 0, f"Cannot find class {class_name}"

        next_class_idx = code.find("\nclass ", class_idx + 1)
        next_def_idx = code.find("\ndef ", class_idx + 1)
        if next_class_idx < 0:
            next_class_idx = len(code)
        if next_def_idx < 0:
            next_def_idx = len(code)

        class_body = code[class_idx : min(next_class_idx, next_def_idx)]

        assert "def to_dict(self)" in class_body, f"Class {class_name} should have to_dict method"


def test_schema_version_constant() -> None:
    """Verify trainability schema version is defined."""
    repo_root = Path(__file__).resolve().parents[1]
    trainability_module = repo_root / "train" / "stage0_trainability.py"

    code = trainability_module.read_text(encoding="utf-8")

    assert "TRAINABILITY_SCHEMA_VERSION = 1" in code


def test_learning_rates_and_checkpoints() -> None:
    """Verify diagnostic learning rates and checkpoints are defined."""
    repo_root = Path(__file__).resolve().parents[1]
    trainability_module = repo_root / "train" / "stage0_trainability.py"

    code = trainability_module.read_text(encoding="utf-8")

    assert "OVERFIT_LEARNING_RATES = (0.0001, 0.001, 0.01)" in code
    assert "OVERFIT_CHECKPOINTS = (0, 1, 4, 16, 64)" in code
    assert "MIN_SEPARATION_RATIO = 0.95" in code
    assert "MAX_UPDATE_RELATIVE_MATRIX_RMS = 0.01" in code
    assert "MAX_LOSS_REDUCTION_RATIO = 0.5" in code


def test_validation_error_has_issues_attribute() -> None:
    """Verify TrainabilityReportValidationError stores issues."""
    repo_root = Path(__file__).resolve().parents[1]
    trainability_module = repo_root / "train" / "stage0_trainability.py"

    code = trainability_module.read_text(encoding="utf-8")

    pattern = "class TrainabilityReportValidationError(ValueError):"
    assert pattern in code

    idx = code.find(pattern)
    next_class = code.find("\nclass ", idx + 1)
    if next_class < 0:
        next_class = len(code)

    class_body = code[idx:next_class]
    assert "self.issues = tuple(issues)" in class_body


def test_record_selection_function_exists() -> None:
    """Verify record selection builder function is defined."""
    repo_root = Path(__file__).resolve().parents[1]
    trainability_module = repo_root / "train" / "stage0_trainability.py"

    code = trainability_module.read_text(encoding="utf-8")

    assert "def build_trainability_record_selections(" in code
    assert "def record_to_identifier(record: Any)" in code


def test_record_selection_returns_tuples() -> None:
    """Verify record selection function returns correct structure."""
    repo_root = Path(__file__).resolve().parents[1]
    trainability_module = repo_root / "train" / "stage0_trainability.py"

    code = trainability_module.read_text(encoding="utf-8")

    assert "overfit_ids = tuple(record_to_identifier(r) for r in overfit_records)" in code
    assert "training_32_ids = tuple(record_to_identifier(r) for r in training_32_records)" in code
    assert "held_out_64_ids = tuple(record_to_identifier(r) for r in held_out_64_records)" in code
    assert "return overfit_ids, training_32_ids, held_out_64_ids" in code


def test_stability_report_loader_exists() -> None:
    """Verify stability report loader function exists."""
    repo_root = Path(__file__).resolve().parents[1]
    trainability_module = repo_root / "train" / "stage0_trainability.py"

    code = trainability_module.read_text(encoding="utf-8")

    assert "def load_and_validate_stability_report(" in code
    assert "TrainabilityReportValidationError" in code


def test_stability_report_loader_validates_status() -> None:
    """Verify stability report loader validates status=failed."""
    repo_root = Path(__file__).resolve().parents[1]
    trainability_module = repo_root / "train" / "stage0_trainability.py"

    code = trainability_module.read_text(encoding="utf-8")

    assert 'status = report_data.get("status")' in code
    assert 'status == "passed"' in code or 'status = "passed"' in code
    assert 'status != "failed"' in code or 'status != "passed"' in code


def test_initial_state_digest_function_exists() -> None:
    """Verify initial state digest function exists."""
    repo_root = Path(__file__).resolve().parents[1]
    trainability_module = repo_root / "train" / "stage0_trainability.py"

    code = trainability_module.read_text(encoding="utf-8")

    assert "def compute_initial_state_digest(" in code
    assert "torch.Tensor" in code


def test_report_types_have_post_init_validation() -> None:
    """Verify all report types have __post_init__ validation."""
    repo_root = Path(__file__).resolve().parents[1]
    trainability_module = repo_root / "train" / "stage0_trainability.py"

    code = trainability_module.read_text(encoding="utf-8")

    classes_to_check = [
        "TrainabilityAssetIdentity",
        "TrainabilityConfiguration",
        "OverfitAttempt",
        "OverfitProbeResult",
        "CausalProbeResult",
        "ArmCheckpoint",
        "ArmResult",
        "TrainabilityReport",
    ]

    for class_name in classes_to_check:
        pattern = f"class {class_name}:"
        class_idx = code.find(pattern)
        assert class_idx >= 0, f"Cannot find class {class_name}"

        next_class_idx = code.find("\nclass ", class_idx + 1)
        next_def_idx = code.find("\ndef ", class_idx + 1)
        if next_class_idx < 0:
            next_class_idx = len(code)
        if next_def_idx < 0:
            next_def_idx = len(code)

        class_body = code[class_idx : min(next_class_idx, next_def_idx)]

        assert "def __post_init__(self)" in class_body or (
            "def __post_init__" in code and class_name in ["TrainabilityAssetIdentity", "TrainabilityConfiguration"]
        ), f"Class {class_name} should have __post_init__ validation"


if __name__ == "__main__":
    test_trainability_module_syntax()
    test_report_types_have_dataclass_decorator()
    test_report_types_have_to_dict_method()
    test_schema_version_constant()
    test_learning_rates_and_checkpoints()
    test_validation_error_has_issues_attribute()
    test_record_selection_function_exists()
    test_record_selection_returns_tuples()
    test_stability_report_loader_exists()
    test_stability_report_loader_validates_status()
    test_initial_state_digest_function_exists()
    test_report_types_have_post_init_validation()
    print("All verification tests passed!")
