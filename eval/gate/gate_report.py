"""Render the exact Stage 0 Calc-ASDiv_A/Qwen gate verdict."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from eval.gate import result_cache, token_cot_baseline
from eval.stream.generators.asdiv_a import AsdivRecord, load_asdiv_a_record_split
from train import standalone_checkpoint

DEFAULT_RESULTS_DIR = Path("gate_results/asdiv_a_qwen")
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "gate_report.json"
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
BASELINE_FLOOR = Fraction(26, 520)


def _fraction(result: dict[str, Any]) -> Fraction:
    total = result["total"]
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        return Fraction(0, 1)
    correct = result["correct"]
    if isinstance(correct, bool) or not isinstance(correct, int):
        return Fraction(0, 1)
    return Fraction(correct, total)


def _exact_default_seeds(latent_result: dict[str, Any] | None) -> bool:
    if latent_result is None:
        return False
    seeds = latent_result.get("seeds")
    runs = latent_result.get("runs")
    return (
        isinstance(seeds, list)
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
        and seeds == DEFAULT_SEEDS
        and isinstance(runs, list)
        and [run.get("seed") for run in runs if isinstance(run, dict)] == DEFAULT_SEEDS
        and all(run.get("total") == 520 for run in runs if isinstance(run, dict))
        and latent_result.get("identity", {}).get("development_only") is False
    )


def evaluate_gate_results(
    token_result: dict[str, Any] | None,
    latent_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Derive criteria from integer counts without trusting persisted floats."""
    baseline_present = token_result is not None
    latent_present = latent_result is not None
    baseline = _fraction(token_result) if token_result is not None else Fraction(0, 1)
    runs = latent_result.get("runs", []) if latent_result is not None else []
    run_fractions = [_fraction(run) for run in runs if isinstance(run, dict)]
    latent_mean = sum(run_fractions, Fraction(0, 1)) / len(run_fractions) if run_fractions else Fraction(0, 1)
    criteria = {
        "criterion_baseline_present": baseline_present,
        "criterion_baseline_meaningful": (
            baseline_present
            and token_result.get("total") == 520
            and token_result.get("identity", {}).get("development_only") is False
            and baseline >= BASELINE_FLOOR
        ),
        "criterion_latent_present": latent_present,
        "criterion_latent_meets_baseline": (
            baseline_present and latent_present and bool(run_fractions) and latent_mean >= baseline
        ),
        "criterion_min_seeds": _exact_default_seeds(latent_result),
    }
    token_display = (
        None
        if token_result is None
        else {
            **token_result,
            "accuracy": float(baseline),
        }
    )
    latent_display = None
    if latent_result is not None:
        floats = [float(value) for value in run_fractions]
        mean = float(latent_mean)
        variance = sum((value - mean) ** 2 for value in floats) / (len(floats) - 1) if len(floats) > 1 else 0.0
        latent_display = {
            **latent_result,
            "accuracies": floats,
            "mean": mean,
            "std": variance**0.5,
        }
    return {
        "schema_version": 2,
        "results": {"token_cot": token_display, "latent_eval": latent_display},
        "criteria": criteria,
        "pass": all(criteria.values()),
    }


def _selected_records(records: tuple[AsdivRecord, ...], ordered_ids: list[str]) -> tuple[AsdivRecord, ...]:
    by_id = {record.id: record for record in records}
    try:
        return tuple(by_id[item_id] for item_id in ordered_ids)
    except KeyError as exc:
        raise result_cache.GateResultError(f"cached selection contains unknown item {exc.args[0]!r}") from exc


def build_report(
    results_dir: Path,
    *,
    records: tuple[AsdivRecord, ...] | None = None,
    expected_token_identity: Mapping[str, Any] | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Read token and latent results through the strict cache validators."""
    token_path = results_dir / "token_cot.json"
    latent_path = results_dir / "latent_eval.json"
    if not token_path.exists() or not latent_path.exists():
        report = evaluate_gate_results(None, None)
        report["results_dir"] = str(results_dir)
        return report

    if expected_token_identity is None:
        raise result_cache.GateResultError("current expected token identity is required to read cached results")

    source_records = records or tuple(load_asdiv_a_record_split("test"))
    selection = expected_token_identity.get("selection")
    if not isinstance(selection, dict) or not isinstance(selection.get("ordered_item_ids"), list):
        raise result_cache.GateResultError("expected token selection identity is missing")
    selected = _selected_records(source_records, selection["ordered_item_ids"])
    token_result = result_cache.read_token_result(
        token_path, expected_identity=expected_token_identity, records=selected
    )

    if expected_checkpoint_sha256 is None or expected_seeds is None:
        raise result_cache.GateResultError(
            "current expected checkpoint digest and seeds are required to read latent results"
        )
    latent_identity = copy.deepcopy(dict(expected_token_identity))
    latent_identity.update(
        {
            "checkpoint_sha256": expected_checkpoint_sha256,
            "seeds": list(expected_seeds),
        }
    )
    latent_result = result_cache.read_latent_result(
        latent_path,
        expected_identity=latent_identity,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_seeds=expected_seeds,
        records=selected,
    )
    report = evaluate_gate_results(token_result, latent_result)
    report["results_dir"] = str(results_dir)
    return report


def _print_summary(report: dict[str, Any]) -> None:
    print("=== Stage 0 gate summary ===")
    token_result = report["results"].get("token_cot")
    latent_result = report["results"].get("latent_eval")
    if token_result is None:
        print("token-CoT baseline: missing")
    else:
        print(f"token-CoT baseline accuracy: {token_result['accuracy']:.4f}")
    if latent_result is None:
        print("latent eval: missing")
    else:
        print(f"latent eval mean accuracy: {latent_result['mean']:.4f} (std={latent_result['std']:.4f})")
    for name, passed in report["criteria"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"GATE: {'PASS' if report['pass'] else 'FAIL'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect strict Stage 0 Calc-ASDiv_A/Qwen results and render a verdict."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Current safe version-2 checkpoint whose identity must match latent results.",
    )
    parser.add_argument(
        "--device",
        required=True,
        help="Current local device used to reconstruct the token request identity.",
    )
    parser.add_argument("--development-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    standalone_checkpoint.configure_deterministic_runtime()
    args = _parse_args()
    token_path = args.results_dir / "token_cot.json"
    latent_path = args.results_dir / "latent_eval.json"
    if token_path.exists() and latent_path.exists():
        records = tuple(load_asdiv_a_record_split("test"))
        prepared = token_cot_baseline.prepare_baseline_request(
            device=args.device,
            development_limit=args.development_limit,
            records=records,
            runtime_configurer=lambda: None,
        )
        report = build_report(
            args.results_dir,
            records=records,
            expected_token_identity=prepared.identity,
            expected_checkpoint_sha256=result_cache.checkpoint_sha256(args.checkpoint),
            expected_seeds=DEFAULT_SEEDS,
        )
    else:
        report = build_report(args.results_dir)
    result_cache.write_json_atomic(args.output, report)
    _print_summary(report)
    print(f"report written to {args.output}")


if __name__ == "__main__":
    main()
