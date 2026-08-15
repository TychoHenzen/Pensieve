"""Chance-rate tests: a random answerer must land near each generator's chance_rate.

Every generator exposes `chance_rate(config)`. That is the accuracy a
random answerer should reach on its probes. Forward transfer and the
Chance oracle both score against that number. A wrong `chance_rate` is
therefore a silent scoring bug rather than a loud one.

These tests prove the number by measurement. They never restate the
formula that produced it. A seeded answerer draws uniformly from each
generator's legal answer set, meaning the vocabulary, the class labels,
or the range of chain answers. It never draws from the expression
`chance_rate` itself evaluates. Its measured accuracy is then checked
against a tolerance. That tolerance comes from the binomial standard
deviation of the sample actually drawn.

Two of the three answer spaces are dense enough for a plain run of the
real generator to give a meaningful check. Those are 1 in 299 for assoc,
and 1 in 5 or 1 in 2 for split-classify. The third space holds 1000
integers. A difficulty-mix run at difficulty 1 is cheap, so it reaches a
large probe count directly and needs no shortcut.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from eval.stream import vocab
from eval.stream.config import StreamConfig
from eval.stream.events import Probe
from eval.stream.generators.assoc import AssocGenerator
from eval.stream.generators.difficulty_mix import CHAIN_MODULUS, DifficultyMixGenerator
from eval.stream.generators.split_classify import SplitClassifyGenerator

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# How many standard deviations of slack the tolerance allows. It is wide
# enough that a correct implementation never flakes across the fixed seeds
# below. It stays tight enough to fail a chance_rate off by any noticeable
# factor.
Z_SCORE = 4.0


@pytest.fixture(autouse=True)
def _corpus_dir(monkeypatch):
    monkeypatch.setenv("PENSIVE_CORPUS_DIR", str(FIXTURE_DIR))


def _binomial_tolerance(n: int, p: float) -> float:
    """Return the fractional tolerance around `p` for a sample of size `n`."""
    std = math.sqrt(n * p * (1 - p))
    return Z_SCORE * std / n


def _assert_within_chance(hits: int, total: int, chance_rate: float) -> None:
    measured = hits / total
    tolerance = _binomial_tolerance(total, chance_rate)
    assert abs(measured - chance_rate) <= tolerance, (
        f"measured accuracy {measured:.6f} over {total} probes is outside "
        f"chance_rate {chance_rate:.6f} +/- {tolerance:.6f}"
    )


# --------------------------------------------------------------------------
# assoc: legal answer set is the full vocabulary. num_pairs=100 with five
# recall distances gives 500 probes per stream. filler_density=0.5 leaves
# the scheduler slack, so it never has to fail a run. 40 seeds accumulate
# 20000 probes. That puts the tolerance near 0.0016 around a chance of
# 0.00334, tight enough to catch a chance_rate off by any sizeable factor.
# --------------------------------------------------------------------------


def test_assoc_random_answerer_matches_chance_rate():
    config = StreamConfig(
        generator="assoc",
        params={
            "num_pairs": 100,
            "recall_distances": [1, 2, 3, 4, 5],
            "filler_density": 0.5,
            "max_distance": 5,
            "corpus": "corpus_fixture",
        },
    )
    generator = AssocGenerator()
    chance_rate = generator.chance_rate(config)
    assert chance_rate == pytest.approx(1.0 / len(vocab.VOCAB))

    answerer = random.Random("assoc-answerer")
    hits = 0
    total = 0
    for seed in range(40):
        for item in generator.generate(config=config, seed=seed):
            if not isinstance(item.event, Probe):
                continue
            guess = answerer.choice(vocab.VOCAB)
            hits += guess == item.truth.answer
            total += 1

    assert total == 100 * 5 * 40
    _assert_within_chance(hits, total, chance_rate)


# --------------------------------------------------------------------------
# split-classify: legal answer set is the class labels of the one probed
# task. num_tasks=1 keeps every probe on the same task, so
# probes_per_task alone controls the sample size cheaply (no filler, no
# corpus). Run at two values of classes_per_task to prove the parameter
# actually moves the measured rate, not just the declared one.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("classes_per_task", [2, 5])
def test_split_classify_random_answerer_matches_chance_rate(classes_per_task):
    config = StreamConfig(
        generator="split-classify",
        params={
            "num_tasks": 1,
            "classes_per_task": classes_per_task,
            "examples_per_task": 4,
            "probes_per_task": 5000,
        },
    )
    generator = SplitClassifyGenerator()
    chance_rate = generator.chance_rate(config)
    assert chance_rate == pytest.approx(1.0 / classes_per_task)

    legal_answers = [f"task0-class{c}" for c in range(classes_per_task)]
    answerer = random.Random(f"split-classify-answerer-{classes_per_task}")
    hits = 0
    total = 0
    for seed in range(5):
        for item in generator.generate(config=config, seed=seed):
            if not isinstance(item.event, Probe):
                continue
            guess = answerer.choice(legal_answers)
            hits += guess == item.truth.answer
            total += 1

    assert total == 5000 * 5
    _assert_within_chance(hits, total, chance_rate)


def test_split_classify_chance_rate_changes_with_classes_per_task():
    generator = SplitClassifyGenerator()
    low = generator.chance_rate(
        StreamConfig(generator="split-classify", params={"classes_per_task": 2})
    )
    high = generator.chance_rate(
        StreamConfig(generator="split-classify", params={"classes_per_task": 5})
    )
    assert low != high
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(0.2)


# --------------------------------------------------------------------------
# difficulty-mix: legal answer set is every integer mod CHAIN_MODULUS,
# which is 1000, so chance is only 1/1000. probe_rate=1.0 and a single
# difficulty level of 1 make every position a cheap, real probe. A plain
# generator run therefore reaches 300000 probes across 5 seeds. It needs
# no direct-answer-set shortcut.
# --------------------------------------------------------------------------


def test_difficulty_mix_random_answerer_matches_chance_rate():
    config = StreamConfig(
        generator="difficulty-mix",
        params={
            "num_items": 60000,
            "difficulty_levels": [1],
            "probe_rate": 1.0,
        },
    )
    generator = DifficultyMixGenerator()
    chance_rate = generator.chance_rate(config)
    assert chance_rate == pytest.approx(1.0 / CHAIN_MODULUS)

    answerer = random.Random("difficulty-mix-answerer")
    hits = 0
    total = 0
    for seed in range(5):
        for item in generator.generate(config=config, seed=seed):
            if not isinstance(item.event, Probe):
                continue
            guess = answerer.randrange(CHAIN_MODULUS)
            hits += guess == item.truth.answer
            total += 1

    assert total == 60000 * 5
    _assert_within_chance(hits, total, chance_rate)
