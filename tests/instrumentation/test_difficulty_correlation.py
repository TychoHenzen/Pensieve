"""Positive control: compute spent by VariableComputeOracle tracks difficulty.

Runs the difficulty-mix generator, feeds every probe through the oracle,
and checks that steps spent per probe correlate strongly (Spearman) with
the difficulty label carried on ProbeTruth. The oracle never reads
ProbeTruth, so any correlation reflects genuine work proportional to
chain length.
"""

from __future__ import annotations

import random

from eval.stream.config import StreamConfig
from eval.stream.events import Probe
from eval.stream.generators.difficulty_mix import DifficultyMixGenerator
from eval.subject.oracles.variable_compute import VariableComputeOracle


def _spearman(xs: list[float], ys: list[float]) -> float:
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        for position, index in enumerate(order):
            ranks[index] = float(position)
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    n = len(xs)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry, strict=True))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    return cov / (var_x * var_y) ** 0.5


def _spearman_p_value(
    xs: list[float], ys: list[float], n_permutations: int = 1000
) -> float:
    """Permutation p-value for a positive Spearman correlation.

    Shuffles the pairing of ``ys`` against ``xs`` and counts how often a
    random pairing produces a correlation at least as large as the observed
    one. Deterministic (seeded) and dependency-free, so the spec's
    "statistically significant" clause is tested without scipy.
    """
    observed = _spearman(xs, ys)
    rng = random.Random(1234)
    indices = list(range(len(ys)))
    count = 0
    for _ in range(n_permutations):
        permuted = indices[:]
        rng.shuffle(permuted)
        permuted_ys = [ys[i] for i in permuted]
        if _spearman(xs, permuted_ys) >= observed:
            count += 1
    return (count + 1) / (n_permutations + 1)


# covers: eval/instrumentation :: Difficulty-correlation test :: positive correlation
def test_variable_compute_oracle_steps_correlate_with_difficulty() -> None:
    config = StreamConfig(
        generator="difficulty-mix",
        params={
            "num_items": 40,
            "difficulty_levels": [1, 2, 3, 4, 5],
            "probe_rate": 1.0,
        },
    )
    generator = DifficultyMixGenerator()
    oracle = VariableComputeOracle()

    difficulties: list[float] = []
    step_deltas: list[float] = []
    for item in generator.generate(config, seed=1234):
        if not isinstance(item.event, Probe):
            continue
        before = oracle.cost().steps
        oracle.answer(item.event)
        after = oracle.cost().steps
        assert item.truth is not None
        difficulties.append(float(item.truth.difficulty))
        step_deltas.append(float(after - before))

    assert len(difficulties) == 40
    correlation = _spearman(difficulties, step_deltas)
    assert correlation > 0.95
    p_value = _spearman_p_value(difficulties, step_deltas)
    assert p_value < 0.01
