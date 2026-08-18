"""Gate 8.5: collect Stage 0 gate results and render a pass/fail determination.

Reads whichever of `token_cot.json`, `latent_eval.json`, `slot_ablation.json`,
and `cycling_sweep.json` exist under `--results-dir` (written by
`eval.gate.token_cot_baseline`, `eval.gate.latent_eval`,
`eval.gate.slot_ablation`, and `eval.gate.cycling_sweep` respectively),
and determines the gate's pass condition: the latent reasoning subject's
mean GSM8K accuracy must be at least the token chain-of-thought
baseline's accuracy. A missing baseline or latent result fails the gate
outright, since the comparison the gate exists to make cannot be made
without both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_RESULTS_DIR = Path("gate_results")
DEFAULT_OUTPUT = Path("gate_results/gate_report.json")

RESULT_FILES = {
    "token_cot": "token_cot.json",
    "latent_eval": "latent_eval.json",
    "slot_ablation": "slot_ablation.json",
    "cycling_sweep": "cycling_sweep.json",
}


def _load_results(results_dir: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for key, filename in RESULT_FILES.items():
        path = results_dir / filename
        if path.exists():
            results[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            results[key] = None
    return results


def _evaluate_criteria(results: dict[str, Any]) -> dict[str, bool]:
    token_cot = results.get("token_cot")
    latent_eval = results.get("latent_eval")

    criterion_baseline_present = token_cot is not None
    criterion_latent_present = latent_eval is not None
    criterion_latent_meets_baseline = (
        criterion_baseline_present
        and criterion_latent_present
        and latent_eval["mean"] >= token_cot["accuracy"]
    )
    criterion_min_seeds = (
        criterion_latent_present and len(latent_eval.get("seeds", [])) >= 5
    )

    return {
        "criterion_baseline_present": criterion_baseline_present,
        "criterion_latent_present": criterion_latent_present,
        "criterion_latent_meets_baseline": criterion_latent_meets_baseline,
        "criterion_min_seeds": criterion_min_seeds,
    }


def build_report(results_dir: Path) -> dict[str, Any]:
    results = _load_results(results_dir)
    criteria = _evaluate_criteria(results)

    return {
        "results_dir": str(results_dir),
        "results": results,
        "criteria": criteria,
        "pass": all(criteria.values()),
    }


def _print_summary(report: dict[str, Any]) -> None:
    print("=== Stage 0 gate summary ===")
    token_cot = report["results"].get("token_cot")
    latent_eval = report["results"].get("latent_eval")
    if token_cot is not None:
        print(f"token-CoT baseline accuracy: {token_cot['accuracy']:.4f}")
    else:
        print("token-CoT baseline: missing")
    if latent_eval is not None:
        print(f"latent eval mean accuracy: {latent_eval['mean']:.4f} (std={latent_eval['std']:.4f})")
    else:
        print("latent eval: missing")

    for name, passed in report["criteria"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"GATE: {'PASS' if report['pass'] else 'FAIL'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate 8.5: collect Stage 0 gate results and render a pass/fail determination."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    report = build_report(args.results_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_summary(report)
    print(f"report written to {args.output}")


if __name__ == "__main__":
    main()
