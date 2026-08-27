"""Convert structured alternating-training logs into CSV and an HTML graph."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class ProgressRecord:
    """One graphable training or held-out evaluation record."""

    record_type: str
    global_step: int
    epoch: int
    example_position: int
    phase_step: int
    cycle: int
    update_method: str
    language_model_loss: float
    shared_variance: float
    answer_exact_match: float | None


CSV_FIELDS = tuple(ProgressRecord.__dataclass_fields__)
METHOD_LABELS = {"eggroll": "Eggroll", "gradient": "Gradient"}


def _decode_log(path: Path) -> str:
    payload = path.read_bytes()
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        return payload.decode("utf-16")
    return payload.decode("utf-8-sig")


def _finite_number(value: object, field: str, line_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"line {line_number}: {field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"line {line_number}: {field} must be a finite number")
    return result


def _integer(value: object, field: str, line_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"line {line_number}: {field} must be an integer")
    return value


def _progress_record(value: dict[str, object], line_number: int) -> ProgressRecord:
    record_type = value.get("record_type")
    update_method = value.get("update_method")
    if record_type not in {"training", "evaluation"}:
        raise ValueError(f"line {line_number}: unsupported record type")
    if update_method not in METHOD_LABELS:
        raise ValueError(f"line {line_number}: update_method must be eggroll or gradient")
    exact_match = value.get("answer_exact_match")
    if record_type == "evaluation" and exact_match is None:
        raise ValueError(f"line {line_number}: evaluation requires answer_exact_match")
    parsed_exact_match = (
        None
        if exact_match is None
        else _finite_number(exact_match, "answer_exact_match", line_number)
    )
    if parsed_exact_match is not None and not 0.0 <= parsed_exact_match <= 1.0:
        raise ValueError(f"line {line_number}: answer_exact_match must be in [0, 1]")
    shared_variance = _finite_number(
        value.get("shared_variance"), "shared_variance", line_number
    )
    if shared_variance < 0.0:
        raise ValueError(f"line {line_number}: shared_variance must be non-negative")
    return ProgressRecord(
        record_type=str(record_type),
        global_step=_integer(value.get("global_step"), "global_step", line_number),
        epoch=_integer(value.get("epoch"), "epoch", line_number),
        example_position=_integer(
            value.get("example_position"), "example_position", line_number
        ),
        phase_step=_integer(value.get("phase_step"), "phase_step", line_number),
        cycle=_integer(value.get("cycle"), "cycle", line_number),
        update_method=str(update_method),
        language_model_loss=_finite_number(
            value.get("language_model_loss"), "language_model_loss", line_number
        ),
        shared_variance=shared_variance,
        answer_exact_match=parsed_exact_match,
    )


def read_progress_records(path: Path) -> list[ProgressRecord]:
    """Read graphable JSON records from a UTF-8 or UTF-16 mixed training log."""
    records: list[ProgressRecord] = []
    for line_number, line in enumerate(_decode_log(path).splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or value.get("record_type") not in {
            "training",
            "evaluation",
        }:
            continue
        records.append(_progress_record(value, line_number))
    if not records:
        raise ValueError(f"{path}: no training or evaluation records found")
    return records


def write_metrics_csv(records: Sequence[ProgressRecord], path: Path) -> None:
    """Write graphable records as a conventional CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {field: getattr(record, field) for field in CSV_FIELDS}
            )


def _extent(values: Sequence[float], *, include_zero: bool = False) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if include_zero:
        low = min(0.0, low)
    padding = max(abs(low) * 0.05, 0.5) if low == high else (high - low) * 0.08
    return low - padding, high + padding


def _format_number(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 10_000:
        return f"{value:.1e}"
    return f"{value:.3g}"


def _line_path(points: Sequence[tuple[float, float]]) -> str:
    return " ".join(
        ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(points)
    )


def _chart(
    *,
    chart_id: str,
    title: str,
    records: Sequence[ProgressRecord],
    value: Callable[[ProgressRecord], float | None],
    y_label: str,
    log_scale: bool = False,
    percent: bool = False,
    reference_lines: Sequence[tuple[float, str, str]] = (),
) -> str:
    width, height = 960.0, 270.0
    left, right, top, bottom = 78.0, 24.0, 24.0, 48.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    points = [(record, value(record)) for record in records]
    points = [(record, item) for record, item in points if item is not None]
    if not points:
        return ""
    x_values = [float(record.global_step) for record, _ in points]
    y_values = [float(item) for _, item in points]
    x_low, x_high = min(x_values), max(x_values)
    if x_low == x_high:
        x_low -= 1.0
        x_high += 1.0

    if log_scale:
        positive = [item for item in y_values if item > 0.0]
        floor = min(positive) / 10.0 if positive else 1e-9

        def transform(item: float) -> float:
            return math.log10(max(item, floor))

        transformed = [transform(item) for item in y_values]
        transformed.extend(transform(item) for item, _, _ in reference_lines if item > 0.0)
        y_low, y_high = _extent(transformed)
    else:
        floor = 0.0

        def transform(item: float) -> float:
            return item

        transformed = y_values + [item for item, _, _ in reference_lines]
        y_low, y_high = _extent(transformed, include_zero=percent)
        if percent:
            y_low, y_high = 0.0, 1.0

    def sx(item: float) -> float:
        return left + (item - x_low) / (x_high - x_low) * plot_width

    def sy(item: float) -> float:
        return top + (y_high - transform(item)) / (y_high - y_low) * plot_height

    x_ticks = [x_low + index * (x_high - x_low) / 4 for index in range(5)]
    y_ticks = [y_low + index * (y_high - y_low) / 4 for index in range(5)]
    grid = []
    labels = []
    for tick in x_ticks:
        x = sx(tick)
        grid.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        labels.append(f'<text class="tick" x="{x:.2f}" y="{height - 20}" text-anchor="middle">{round(tick)}</text>')
    for tick in y_ticks:
        y = top + (y_high - tick) / (y_high - y_low) * plot_height
        grid.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        original = 10 ** tick if log_scale else tick
        label = f"{original * 100:.0f}%" if percent else _format_number(original)
        labels.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{label}</text>')

    references = []
    for item, label, css_class in reference_lines:
        if log_scale and item <= 0.0:
            continue
        y = sy(item)
        references.append(
            f'<line class="reference {css_class}" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>'
            f'<text class="reference-label {css_class}" x="{left + plot_width - 4}" y="{y - 5:.2f}" text-anchor="end">{escape(label)}</text>'
        )

    marks = []
    for method in METHOD_LABELS:
        runs: list[list[tuple[float, float]]] = []
        current_run: list[tuple[float, float]] = []
        for record, item in points:
            if record.record_type != "training":
                continue
            if record.update_method == method:
                current_run.append(
                    (sx(float(record.global_step)), sy(float(item)))
                )
            elif current_run:
                runs.append(current_run)
                current_run = []
        if current_run:
            runs.append(current_run)
        for run in runs:
            if len(run) > 1:
                marks.append(f'<path class="series {method}" d="{_line_path(run)}"/>')
            for x, y in run:
                marks.append(
                    f'<circle class="point {method}" cx="{x:.2f}" cy="{y:.2f}" r="3"/>'
                )
    evaluation_points = [
        (sx(float(record.global_step)), sy(float(item)))
        for record, item in points
        if record.record_type == "evaluation"
    ]
    if len(evaluation_points) > 1:
        marks.append(f'<path class="series evaluation" d="{_line_path(evaluation_points)}"/>')
    for x, y in evaluation_points:
        marks.append(
            f'<rect class="point evaluation" x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" height="8"/>'
        )

    return f"""
    <section class="chart-section">
      <h2>{escape(title)}</h2>
      <svg viewBox="0 0 {int(width)} {int(height)}" role="img" aria-labelledby="{chart_id}-title {chart_id}-desc">
        <title id="{chart_id}-title">{escape(title)}</title>
        <desc id="{chart_id}-desc">{escape(y_label)} plotted against global training step.</desc>
        {''.join(grid)}
        <rect class="frame" x="{left}" y="{top}" width="{plot_width}" height="{plot_height}"/>
        {''.join(references)}
        {''.join(marks)}
        {''.join(labels)}
        <text class="axis-title" x="{left + plot_width / 2}" y="{height - 3}" text-anchor="middle">Global step</text>
        <text class="axis-title" transform="translate(18 {top + plot_height / 2}) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>
      </svg>
    </section>"""


def write_training_graph(
    records: Sequence[ProgressRecord],
    path: Path,
    *,
    title: str,
    variance_lower_threshold: float = 0.01,
    variance_upper_threshold: float = 0.02,
    baseline_exact_match: float | None = None,
) -> None:
    """Write a self-contained browser-viewable training graph."""
    if not records:
        raise ValueError("at least one progress record is required")
    baseline_lines = (
        []
        if baseline_exact_match is None
        else [
            (
                baseline_exact_match,
                f"{baseline_exact_match * 100:.2f}% baseline",
                "baseline",
            )
        ]
    )
    charts = [
        _chart(
            chart_id="loss",
            title="Language-model loss",
            records=records,
            value=lambda record: record.language_model_loss,
            y_label="Cross-entropy loss",
        ),
        _chart(
            chart_id="variance",
            title="Shared slot variance (log scale)",
            records=records,
            value=lambda record: record.shared_variance,
            y_label="Population variance",
            log_scale=True,
            reference_lines=[
                (variance_lower_threshold, "lower hysteresis threshold", "lower"),
                (variance_upper_threshold, "upper hysteresis threshold", "upper"),
            ],
        ),
        _chart(
            chart_id="exact-match",
            title="Held-out exact match",
            records=records,
            value=lambda record: record.answer_exact_match,
            y_label="Exact match",
            percent=True,
            reference_lines=baseline_lines,
        ),
    ]
    training_count = sum(record.record_type == "training" for record in records)
    evaluation_count = sum(record.record_type == "evaluation" for record in records)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: light dark; --bg: #ffffff; --fg: #172033; --muted: #657087; --grid: #d9deea; --eggroll: #c56816; --gradient: #2867b2; --evaluation: #292d38; --lower: #7a57a5; --upper: #3a855d; --baseline: #a23c45; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --bg: #11151d; --fg: #edf1f8; --muted: #aeb7c8; --grid: #343b49; --eggroll: #f1a055; --gradient: #6fa8eb; --evaluation: #e6e9ef; --lower: #b79bd2; --upper: #71bf91; --baseline: #e57b84; }} }}
    body {{ max-width: 1100px; margin: 0 auto; padding: 24px; background: var(--bg); color: var(--fg); font: 15px/1.4 system-ui, sans-serif; }}
    h1 {{ margin: 0 0 4px; font-size: 1.55rem; font-weight: 600; }}
    h2 {{ margin: 26px 0 4px; font-size: 1.05rem; font-weight: 600; }}
    .summary {{ margin: 0 0 12px; color: var(--muted); }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 18px; color: var(--muted); }}
    .legend span::before {{ content: ""; display: inline-block; width: 18px; height: 3px; margin: 0 6px 3px 0; background: currentColor; }}
    .legend .eggroll {{ color: var(--eggroll); }} .legend .gradient {{ color: var(--gradient); }} .legend .evaluation {{ color: var(--evaluation); }}
    svg {{ display: block; width: 100%; height: auto; overflow: visible; }}
    svg text {{ fill: var(--fg); font: 12px system-ui, sans-serif; }}
    .frame {{ fill: none; stroke: var(--grid); }} .grid {{ stroke: var(--grid); stroke-width: 1; }}
    .tick {{ fill: var(--muted); }} .axis-title {{ font-weight: 600; }}
    .series {{ fill: none; stroke-width: 2; }} .point {{ stroke-width: 1; }}
    .eggroll {{ stroke: var(--eggroll); fill: var(--eggroll); }} .gradient {{ stroke: var(--gradient); fill: var(--gradient); }} .evaluation {{ stroke: var(--evaluation); fill: var(--bg); }}
    .reference {{ stroke-width: 1.5; stroke-dasharray: 6 5; }} .reference-label {{ font-size: 11px; }}
    .lower {{ stroke: var(--lower); fill: var(--lower); }} .upper {{ stroke: var(--upper); fill: var(--upper); }} .baseline {{ stroke: var(--baseline); fill: var(--baseline); }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="summary">{training_count} training samples and {evaluation_count} held-out evaluations from structured progress records.</p>
  <div class="legend" aria-label="Series legend"><span class="eggroll">Eggroll</span><span class="gradient">Gradient</span><span class="evaluation">Evaluation</span></div>
  {''.join(charts)}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a structured alternating-training log to CSV and HTML."
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--variance-lower", type=float, default=0.01)
    parser.add_argument("--variance-upper", type=float, default=0.02)
    parser.add_argument("--baseline-exact-match", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if not 0.0 <= args.variance_lower < args.variance_upper:
        raise ValueError("variance thresholds must satisfy 0 <= lower < upper")
    if args.baseline_exact_match is not None and not 0.0 <= args.baseline_exact_match <= 1.0:
        raise ValueError("baseline exact match must be in [0, 1]")
    records = read_progress_records(args.log)
    output_dir = args.output_dir or args.log.with_suffix("").with_name(
        args.log.stem + "-metrics"
    )
    csv_path = output_dir / "training-metrics.csv"
    html_path = output_dir / "training-graph.html"
    write_metrics_csv(records, csv_path)
    write_training_graph(
        records,
        html_path,
        title=args.title or f"Training results: {args.log.name}",
        variance_lower_threshold=args.variance_lower,
        variance_upper_threshold=args.variance_upper,
        baseline_exact_match=args.baseline_exact_match,
    )
    print(csv_path)
    print(html_path)


if __name__ == "__main__":
    main()
