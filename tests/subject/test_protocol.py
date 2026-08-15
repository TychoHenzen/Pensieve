from __future__ import annotations

import pytest

from eval.stream.events import Observe, Probe
from eval.subject import CostCounters, Subject
from eval.subject.oracles.perfect_memory import PerfectMemoryOracle


# covers: eval/subject/protocol::Subject is an abstract base class::direct instantiation rejected
def test_subject_abc_not_instantiable():
    with pytest.raises(TypeError):
        Subject()


# covers: eval/subject/protocol::Subject is an abstract base class::partial implementation rejected
def test_partial_implementation_not_instantiable():
    class Incomplete(Subject):
        def observe(self, event):
            return None

    with pytest.raises(TypeError):
        Incomplete()


# covers: eval/subject/protocol::CostCounters is a frozen dataclass with zero defaults::frozen after construction
def test_cost_counters_frozen():
    c = CostCounters(steps=1, flops=2, wall_seconds=3.0)
    with pytest.raises(AttributeError):
        c.steps = 10  # type: ignore


# covers: eval/subject/protocol::CostCounters is a frozen dataclass with zero defaults::zero defaults
def test_cost_counters_defaults():
    c = CostCounters()
    assert c.steps == 0
    assert c.flops == 0
    assert c.wall_seconds == 0.0


# covers: eval/subject/protocol::Cost monotonicity::steps increase after answer
def test_cost_monotonicity():
    oracle = PerfectMemoryOracle()
    c0 = oracle.cost()

    oracle.observe(Observe(position=0, payload={"key": "a", "value": "b"}))
    c1 = oracle.cost()
    assert c1.steps >= c0.steps

    oracle.answer(Probe(position=1, probe_id="p1", task_id="t", query="a"))
    c2 = oracle.cost()
    assert c2.steps >= c1.steps
    assert c2.steps > c0.steps
