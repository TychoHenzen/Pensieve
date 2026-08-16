from __future__ import annotations

import pytest

from eval.stream.events import Observe, Probe
from eval.subject import CostCounters, Subject
from eval.subject.oracles.perfect_memory import PerfectMemoryOracle


# covers: eval/subject::Subject is an abstract base class::direct instantiation rejected
def test_subject_abc_not_instantiable():
    with pytest.raises(TypeError):
        Subject()


# covers: eval/subject::Subject is an abstract base class::partial implementation rejected
def test_partial_implementation_not_instantiable():
    class Incomplete(Subject):
        def observe(self, event):
            return None

    with pytest.raises(TypeError):
        Incomplete()


# covers: eval/subject::Subject is an abstract base class::full implementation accepted
def test_full_implementation_instantiable():
    class Complete(Subject):
        def observe(self, event):
            return None

        def answer(self, probe):
            return ""

        def idle(self, budget):
            return None

        def snapshot(self):
            return None

        def restore(self, state):
            return None

        def cost(self):
            return CostCounters()

    assert isinstance(Complete(), Subject)


# covers: eval/subject::Subject is an abstract base class::Subject is an ABC
def test_subject_is_an_abc():
    import abc

    assert issubclass(Subject, abc.ABC) or isinstance(Subject, abc.ABCMeta)


# covers: eval/subject::CostCounters is a frozen dataclass with zero defaults::frozen after construction
def test_cost_counters_frozen():
    c = CostCounters(steps=1, flops=2, wall_seconds=3.0)
    with pytest.raises(AttributeError):
        c.steps = 10  # type: ignore


# covers: eval/subject::CostCounters is a frozen dataclass with zero defaults::zero defaults
def test_cost_counters_defaults():
    c = CostCounters()
    assert c.steps == 0
    assert c.flops == 0
    assert c.wall_seconds == 0.0


# covers: eval/subject::CostCounters is a frozen dataclass with zero defaults::steps defaults to zero
def test_cost_counters_steps_defaults_to_zero():
    c = CostCounters()
    assert c.steps == 0


# covers: eval/subject::CostCounters is a frozen dataclass with zero defaults::flops defaults to zero
def test_cost_counters_flops_defaults_to_zero():
    c = CostCounters()
    assert c.flops == 0


# covers: eval/subject::CostCounters is a frozen dataclass with zero defaults::wall_seconds defaults to zero
def test_cost_counters_wall_seconds_defaults_to_zero():
    c = CostCounters()
    assert c.wall_seconds == 0.0


# covers: eval/subject::Cost monotonicity::steps increase after answer
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


# covers: eval/subject::Cost monotonicity::steps non-decreasing across observe calls
def test_cost_steps_non_decreasing_across_observes():
    oracle = PerfectMemoryOracle()
    prev = oracle.cost().steps
    for key, value in [("a", "1"), ("b", "2"), ("c", "3")]:
        oracle.observe(Observe(position=0, payload={"key": key, "value": value}))
        current = oracle.cost().steps
        assert current >= prev
        prev = current


# covers: eval/subject::Cost monotonicity::steps strictly increase after answer vs initial
def test_cost_steps_strictly_increase_after_answer_vs_initial():
    oracle = PerfectMemoryOracle()
    initial = oracle.cost().steps
    oracle.answer(Probe(position=0, probe_id="p1", task_id="t", query="a"))
    assert oracle.cost().steps > initial
