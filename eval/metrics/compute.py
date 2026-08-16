from __future__ import annotations

from dataclasses import dataclass

from eval.metrics import ProbeLogEntry


@dataclass(frozen=True)
class ComputePerInput:
    steps: list[int]
    flops: list[int]
    wall_seconds: list[float]


def compute_per_input(log: list[ProbeLogEntry]) -> ComputePerInput:
    steps: list[int] = []
    flops: list[int] = []
    wall_seconds: list[float] = []

    previous = None
    for entry in log:
        counters = entry.cost_counters
        if previous is None:
            steps.append(counters.steps)
            flops.append(counters.flops)
            wall_seconds.append(counters.wall_seconds)
        else:
            steps.append(counters.steps - previous.steps)
            flops.append(counters.flops - previous.flops)
            wall_seconds.append(counters.wall_seconds - previous.wall_seconds)
        previous = counters

    return ComputePerInput(steps=steps, flops=flops, wall_seconds=wall_seconds)
