"""Reproduction gate pass-condition contract tests.

`scripts/run_gate.py` is the Stage -1 reproduction gate and has no direct
test file. Running the full gate needs MNIST data and takes minutes per
seed, so the pass condition written down before the first run - per-method
tolerance bands, a five-seed minimum, and the paper's method ordering - is
pinned here against the module's own constants and `_evaluate_criteria`
logic, without executing a gate run.
"""

from __future__ import annotations

import scripts.run_gate as gate


def _summary(means: dict[str, float], seed_count: int = 5) -> dict:
    """Return a gate summary with every method at `means` across `seed_count` seeds."""
    return {method: {"mean": mean, "accuracies": [mean] * seed_count} for method, mean in means.items()}


def _reproducibility(ok: bool = True) -> dict[str, bool]:
    return dict.fromkeys(gate.METHODS, ok)


# covers: eval/reproduction-gate::tolerance is declared
def test_pass_condition_declares_a_numerical_tolerance_per_method():
    assert set(gate.PASS_BANDS) == set(gate.METHODS)
    for method in gate.METHODS:
        low, high = gate.PASS_BANDS[method]
        assert isinstance(low, (int, float))
        assert isinstance(high, (int, float))
        assert low <= high


# covers: eval/reproduction-gate::five seeds
def test_min_seeds_requires_at_least_five():
    assert gate.MIN_SEEDS >= 5


# covers: eval/reproduction-gate::five seeds
def test_seed_count_criterion_rejects_fewer_than_five_seeds():
    four_seed_summary = _summary({"naive": 20.0, "ewc": 20.0, "replay": 85.0}, seed_count=4)
    four_seed_criteria = gate._evaluate_criteria(four_seed_summary, [0, 1, 2, 3], _reproducibility())
    assert four_seed_criteria["criterion_4_seed_count"] is False

    five_seed_summary = _summary({"naive": 20.0, "ewc": 20.0, "replay": 85.0})
    five_seed_criteria = gate._evaluate_criteria(five_seed_summary, [0, 1, 2, 3, 4], _reproducibility())
    assert five_seed_criteria["criterion_4_seed_count"] is True


# covers: eval/reproduction-gate::five seeds
def test_seed_count_criterion_requires_five_accuracies_per_method():
    short_summary = _summary({"naive": 20.0, "ewc": 20.0, "replay": 85.0}, seed_count=4)
    criteria = gate._evaluate_criteria(short_summary, [0, 1, 2, 3, 4], _reproducibility())
    assert criteria["criterion_4_seed_count"] is False


# covers: eval/reproduction-gate::ordering check
def test_ordering_criterion_requires_replay_above_ewc_and_naive():
    paper_order = _summary({"naive": 20.0, "ewc": 20.0, "replay": 85.0})
    criteria = gate._evaluate_criteria(paper_order, [0, 1, 2, 3, 4], _reproducibility())
    assert criteria["criterion_3_ordering"] is True

    reversed_order = _summary({"naive": 20.0, "ewc": 85.0, "replay": 20.0})
    criteria = gate._evaluate_criteria(reversed_order, [0, 1, 2, 3, 4], _reproducibility())
    assert criteria["criterion_3_ordering"] is False
