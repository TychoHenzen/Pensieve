"""Pre-runtime validation for workflows guarded by an EGGROLL stability report."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval.stage0_identity import STAGE0_IDENTITY, training_identity
from train.eggroll_stability import (
    BASELINE_PROBLEM_COUNT,
    DEVELOPMENT_CHECKPOINTS,
    DEVELOPMENT_EXAMPLE_COUNT,
    StabilityReportValidationError,
    ValidatedStabilityReport,
    canonical_implementation_identity,
    load_compatible_stability_report,
    load_matching_stability_report,
    read_stability_report,
)
from train.standalone_checkpoint import runtime_identity
from train.training_state import EGGROLL_PARAMETER_PATHS


def load_guarded_stability_report(
    path: str | Path,
    *,
    population: int,
    sigma: float,
    rank: int,
    fitness_batch_size: int,
    evaluation_batch_size: int,
    variance_weight: float,
    prompt_alignment_weight: float,
    learning_rate: float,
    require_passing: bool,
    repository_root: str | Path | None = None,
) -> ValidatedStabilityReport:
    """Validate every guard field available before model or dataset access."""
    parsed = read_stability_report(path)
    actual = parsed.report.configuration.to_dict()
    asset_identity = actual.get("asset_identity")
    if not isinstance(asset_identity, Mapping):
        # The strict parser normally rejects this. Keep this guard explicit for callers.
        raise StabilityReportValidationError(
            ("$.configuration.asset_identity: expected object",)
        )
    held_out_item_ids = asset_identity.get("held_out_item_ids")
    if (
        not isinstance(held_out_item_ids, list)
        or len(held_out_item_ids) != BASELINE_PROBLEM_COUNT
        or any(not isinstance(item, str) or not item for item in held_out_item_ids)
        or len(set(held_out_item_ids)) != len(held_out_item_ids)
    ):
        raise StabilityReportValidationError(
            (
                "$.configuration.asset_identity.held_out_item_ids: expected "
                f"{BASELINE_PROBLEM_COUNT} unique non-empty strings"
            ),
        )
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    expected_configuration: dict[str, Any] = {
        "asset_identity": training_identity(
            runtime=runtime_identity(),
            held_out_item_ids=held_out_item_ids,
        ),
        "implementation": canonical_implementation_identity(root).to_dict(),
        "eggroll": {
            "optimizer_contract": "torch.optim.SGD(momentum=0.0)",
            "parameter_scope": list(EGGROLL_PARAMETER_PATHS),
            "population": population,
            "sigma": sigma,
            "rank": rank,
            "fitness_batch_size": fitness_batch_size,
            "evaluation_batch_size": evaluation_batch_size,
            "variance_weight": variance_weight,
            "prompt_alignment_weight": prompt_alignment_weight,
            "learning_rate": learning_rate,
        },
        "initialization_seed": 0,
        "held_out_problem_count": BASELINE_PROBLEM_COUNT,
        "development_example_count": DEVELOPMENT_EXAMPLE_COUNT,
        "checkpoints": list(DEVELOPMENT_CHECKPOINTS),
    }
    # Assert the pinned identity here so a report cannot introduce extra asset fields.
    expected_configuration["asset_identity"] = {
        **dict(STAGE0_IDENTITY),
        **expected_configuration["asset_identity"],
    }
    loader = (
        load_compatible_stability_report
        if require_passing
        else load_matching_stability_report
    )
    return loader(path, expected_configuration)


def standalone_consumed_example_limit(
    *,
    epochs: int,
    problem_count: int | None,
    default_problem_count: int,
) -> int:
    """Return the maximum examples a standalone invocation can consume."""
    selected_count = default_problem_count if problem_count is None else problem_count
    return epochs * selected_count
