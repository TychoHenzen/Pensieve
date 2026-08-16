"""Time to first use: how many stream positions elapse before a taught
fact is first answered correctly.

Facts are identified by the (probe_id, teaching_position) pair, since a
probe_id could in principle be re-taught at a different position. For
each such fact, `time_to_first_use` walks the log in order and records
the distance (probe position - teaching position) at the first correct
probe. Facts never answered correctly within the log are assigned the
`cutoff` distance instead, so they still contribute to the median while
being flagged via `cutoff_share`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median as _median

from eval.metrics import ProbeLogEntry


@dataclass(frozen=True)
class FirstUseResult:
    median: float
    cutoff_share: float
    per_fact: dict[str, int] = field(default_factory=dict)


def time_to_first_use(log: list[ProbeLogEntry], cutoff: int) -> FirstUseResult:
    first_use: dict[tuple[str, int], int | None] = {}

    for entry in log:
        fact = (entry.probe_id, entry.teaching_position)
        if fact not in first_use:
            first_use[fact] = None
        if entry.correct and first_use[fact] is None:
            first_use[fact] = entry.position - entry.teaching_position

    values: list[int] = []
    per_fact: dict[str, int] = {}
    cutoff_hits = 0

    for (probe_id, _teaching_position), distance in first_use.items():
        if distance is None:
            distance = cutoff
            cutoff_hits += 1
        values.append(distance)
        per_fact[probe_id] = distance

    total_facts = len(first_use)
    cutoff_share = (cutoff_hits / total_facts) if total_facts else 0.0
    median_value = float(_median(values)) if values else 0.0

    return FirstUseResult(
        median=median_value,
        cutoff_share=cutoff_share,
        per_fact=per_fact,
    )
