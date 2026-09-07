"""Run the strict Stage 0 Calc-MAWPS/Qwen gate."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from eval.gate.answer_scoring import MIN_MEANINGFUL_ACCURACY
from eval.gate.gate_report import build_report
from eval.gate.latent_eval import run_eval
from eval.gate.result_cache import (
    checkpoint_sha256,
    read_latent_result,
    read_token_result,
    write_gate_result,
    write_json_atomic,
)
from eval.gate.token_cot_baseline import (
    PreparedBaselineRequest,
    prepare_baseline_request,
    run_baseline,
)
from eval.stage0_identity import LATENT_TAP_LAYER
from eval.stream.generators.calc_mawps import CalcMawpsRecord, load_calc_mawps_record_split
from eval.subjects.latent_core import DEFAULT_NUM_STEPS
from train.standalone_checkpoint import configure_deterministic_runtime
from workspace.concept_slots import DEFAULT_SLOT_COUNT


DEFAULT_RESULTS_DIR = Path("gate_results/calc_mawps_qwen")
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
LOG_EVERY = 50


def _ts() -> str:
    now = time.localtime()
    return f"[{now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}]"


def _log(message: str) -> None:
    print(f"{_ts()} {message}", flush=True)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h{minutes:02d}m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the strict Stage 0 Calc-MAWPS/Qwen gate evaluation."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Safe version-2 .ckpt checkpoint under evaluation.",
    )
    parser.add_argument("--slot-count", type=int, default=DEFAULT_SLOT_COUNT)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--development-limit",
        "--problem-count",
        dest="development_limit",
        type=int,
        default=None,
        help="Development-only prefix length. The default uses all 520 test items.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seeds. The gate requires exactly 0,1,2,3,4.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def _records() -> tuple[CalcMawpsRecord, ...]:
    return tuple(load_calc_mawps_record_split("test"))


def _accuracy(result: Mapping[str, Any]) -> float:
    total = result["total"]
    return result["correct"] / total if total else 0.0


def _run_baseline(
    args: argparse.Namespace,
    records: Sequence[CalcMawpsRecord] | None = None,
    prepared_request: PreparedBaselineRequest | None = None,
) -> dict[str, Any]:
    _log("=" * 60)
    _log("  PHASE 1: Frozen-Qwen token chain-of-thought baseline")
    _log("=" * 60)
    source = tuple(records) if records is not None else _records()
    prepared = prepared_request or prepare_baseline_request(
        device=args.device,
        development_limit=args.development_limit,
        records=source,
    )
    selection = prepared.selection
    results_path = args.results_dir / "token_cot.json"
    if results_path.exists():
        from eval.gate import result_cache

        result = read_token_result(
            results_path,
            expected_identity=prepared.identity,
            records=selection.records,
        )
        _log(
            f"reused validated token cache: {_accuracy(result):.4f} "
            f"({result['correct']}/{result['total']})"
        )
        return result

    phase_start = time.monotonic()

    def on_problem(_index: int, total: int, _correct: bool, correct: int, done: int) -> None:
        if done % LOG_EVERY == 0 or done == total:
            elapsed = time.monotonic() - phase_start
            remaining = (elapsed / done) * (total - done)
            _log(
                f"  baseline [{done}/{total}] acc={correct / done:.4f} "
                f"({correct}/{done}) ETA {_format_duration(remaining)}"
            )

    result = run_baseline(
        device=args.device,
        development_limit=args.development_limit,
        records=source,
        on_problem=on_problem,
        runtime_configurer=lambda: None,
        prepared_request=prepared,
    )
    write_gate_result(results_path, result)
    _log(
        f"baseline done ({_format_duration(time.monotonic() - phase_start)}): "
        f"{_accuracy(result):.4f} ({result['correct']}/{result['total']})"
    )
    return result


def _expected_latent_identity(
    token_request_identity: Mapping[str, Any],
    args: argparse.Namespace,
    seeds: list[int],
) -> dict[str, Any]:
    identity = copy.deepcopy(dict(token_request_identity))
    identity.update(
        {
            "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
            "seeds": list(seeds),
            "slot_count": args.slot_count,
            "latent_step_count": args.num_steps,
            "tap_tuple_index": LATENT_TAP_LAYER,
        }
    )
    return identity


def _run_latent_eval(
    args: argparse.Namespace,
    seeds: list[int],
    token_result: Mapping[str, Any] | None = None,
    records: Sequence[CalcMawpsRecord] | None = None,
    prepared_request: PreparedBaselineRequest | None = None,
) -> dict[str, Any]:
    _log("=" * 60)
    _log("  PHASE 2: Latent reasoning evaluation")
    _log("=" * 60)
    source = tuple(records) if records is not None else _records()
    prepared = prepared_request or prepare_baseline_request(
        device=args.device,
        development_limit=args.development_limit,
        records=source,
    )
    if token_result is None:
        token_result = _run_baseline(args, source, prepared)
    results_path = args.results_dir / "latent_eval.json"
    expected_identity = _expected_latent_identity(prepared.identity, args, seeds)
    selected_ids = expected_identity["selection"]["ordered_item_ids"]
    by_id = {record.id: record for record in source}
    selected = tuple(by_id[item_id] for item_id in selected_ids)
    digest = expected_identity["checkpoint_sha256"]
    if results_path.exists():
        result = read_latent_result(
            results_path,
            expected_identity=expected_identity,
            expected_checkpoint_sha256=digest,
            expected_seeds=seeds,
            records=selected,
        )
        _log("reused validated latent cache")
        return result

    phase_start = time.monotonic()
    result = run_eval(
        checkpoint_path=args.checkpoint,
        seeds=seeds,
        development_limit=None,
        slot_count=args.slot_count,
        num_steps=args.num_steps,
        device=args.device,
        token_result=token_result,
        records=source,
        runtime_configurer=lambda: None,
        backbone_loader=lambda **_kwargs: prepared.backbone,
    )
    write_gate_result(results_path, result)
    _log(f"latent eval done ({_format_duration(time.monotonic() - phase_start)})")
    return result


def _run_verdict(
    args: argparse.Namespace,
    baseline: dict[str, Any],
    latent: dict[str, Any],
    records: Sequence[CalcMawpsRecord],
    prepared_request: PreparedBaselineRequest,
    seeds: Sequence[int],
) -> bool:
    del baseline, latent
    _log("=" * 60)
    _log("  PHASE 3: Gate verdict")
    _log("=" * 60)
    report = build_report(
        args.results_dir,
        records=tuple(records),
        expected_token_identity=prepared_request.identity,
        expected_checkpoint_sha256=checkpoint_sha256(args.checkpoint),
        expected_seeds=seeds,
    )
    report_path = args.results_dir / "gate_report.json"
    write_json_atomic(report_path, report)
    token = report["results"]["token_cot"]
    latent_result = report["results"]["latent_eval"]
    _log(f"  token-CoT baseline:    {token['accuracy']:.4f}")
    _log(
        f"  latent mean (5 seeds): {latent_result['mean']:.4f} "
        f"(std={latent_result['std']:.4f})"
    )
    for name, passed in report["criteria"].items():
        _log(f"  {name}: {'PASS' if passed else 'FAIL'}")
    _log(f"  GATE: {'PASS' if report['pass'] else 'FAIL'}")
    return bool(report["pass"])


def main() -> None:
    configure_deterministic_runtime()
    args = _parse_args()
    seeds = [int(token) for token in args.seeds.split(",") if token.strip()]
    if not args.checkpoint.is_file():
        _log(f"checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if seeds != DEFAULT_SEEDS:
        _log("gate seeds must be exactly 0,1,2,3,4 in that order")
        sys.exit(1)
    if args.slot_count != DEFAULT_SLOT_COUNT or args.num_steps != DEFAULT_NUM_STEPS:
        _log(
            f"official gate geometry requires slots={DEFAULT_SLOT_COUNT} "
            f"and latent steps={DEFAULT_NUM_STEPS}"
        )
        sys.exit(1)
    source = _records()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    gate_start = time.monotonic()
    prepared = prepare_baseline_request(
        device=args.device,
        development_limit=args.development_limit,
        records=source,
        runtime_configurer=lambda: None,
    )
    baseline = _run_baseline(args, source, prepared)
    if _accuracy(baseline) < MIN_MEANINGFUL_ACCURACY:
        _log(
            f"baseline accuracy {_accuracy(baseline):.4f} is below the "
            f"documented {MIN_MEANINGFUL_ACCURACY:.0%} floor"
        )
        _log("GATE: INVALID - select a simpler dataset before latent evaluation")
        sys.exit(1)
    latent = _run_latent_eval(args, seeds, baseline, source, prepared)
    passed = _run_verdict(
        args,
        baseline,
        latent,
        prepared.selection.records,
        prepared,
        seeds,
    )
    _log(f"total gate time: {_format_duration(time.monotonic() - gate_start)}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
