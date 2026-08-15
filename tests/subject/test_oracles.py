from __future__ import annotations

import math

from eval.stream.events import Boundary, BoundaryKind, Observe, Probe
from eval.stream.vocab import VOCAB
from eval.subject.oracles.chance import ChanceOracle
from eval.subject.oracles.cheater import CheaterOracle
from eval.subject.oracles.forgetful import ForgetfulOracle
from eval.subject.oracles.perfect_memory import PerfectMemoryOracle
from eval.subject.oracles.task_wiper import TaskWiperOracle


def _assoc_stream():
    pairs = [("cat", "meow"), ("dog", "bark"), ("cow", "moo")]
    events = []
    pos = 0
    for key, value in pairs:
        events.append(Observe(position=pos, payload={"key": key, "value": value}))
        pos += 1
    probes = []
    for key, value in pairs:
        probes.append(
            Probe(position=pos, probe_id=f"p-{key}", task_id="assoc", query=key)
        )
        pos += 1
    return events, probes, {k: v for k, v in pairs}


# covers: eval/subject/oracles::PerfectMemoryOracle recalls all observations::recalls everything
def test_perfect_memory_recalls_everything():
    events, probes, truth = _assoc_stream()
    oracle = PerfectMemoryOracle()
    for e in events:
        oracle.observe(e)
    for p in probes:
        assert oracle.answer(p) == truth[p.query]


# covers: eval/subject/oracles::PerfectMemoryOracle recalls all observations::unknown key returns empty
def test_perfect_memory_unknown_returns_empty():
    oracle = PerfectMemoryOracle()
    probe = Probe(position=0, probe_id="p", task_id="t", query="never_taught")
    assert oracle.answer(probe) == ""


def test_forgetful_recalls_only_last():
    events, probes, truth = _assoc_stream()
    oracle = ForgetfulOracle()
    for e in events:
        oracle.observe(e)

    last_key = "cow"
    for p in probes:
        answer = oracle.answer(p)
        if p.query == last_key:
            assert answer == truth[last_key]
        else:
            assert answer == ""


# covers: eval/subject/oracles::ForgetfulOracle retains only the most recent observation::immediate recall correct
def test_forgetful_correct_on_immediate_recall():
    oracle = ForgetfulOracle()
    oracle.observe(Observe(position=0, payload={"key": "x", "value": "y"}))
    probe = Probe(position=1, probe_id="p", task_id="t", query="x")
    assert oracle.answer(probe) == "y"


# covers: eval/subject/oracles::ForgetfulOracle retains only the most recent observation::earlier keys forgotten after intervening observation
def test_forgetful_wrong_after_intervening():
    oracle = ForgetfulOracle()
    oracle.observe(Observe(position=0, payload={"key": "a", "value": "1"}))
    oracle.observe(Observe(position=1, payload={"key": "b", "value": "2"}))
    oracle.observe(Observe(position=2, payload={"key": "c", "value": "3"}))
    probe = Probe(position=3, probe_id="p", task_id="t", query="a")
    assert oracle.answer(probe) != "1"


# covers: eval/subject/oracles::ChanceOracle answers randomly from VOCAB::accuracy near chance rate
def test_chance_near_chance_rate():
    n_probes = 2000
    oracle = ChanceOracle(seed=42)
    vocab_size = len(VOCAB)
    expected_rate = 1.0 / vocab_size

    correct = 0
    for i in range(n_probes):
        probe = Probe(
            position=i, probe_id=f"p{i}", task_id="t", query=VOCAB[i % vocab_size]
        )
        answer = oracle.answer(probe)
        if answer == VOCAB[i % vocab_size]:
            correct += 1

    mu = n_probes * expected_rate
    sigma = math.sqrt(n_probes * expected_rate * (1 - expected_rate))
    z = abs(correct - mu) / sigma if sigma > 0 else 0
    assert z < 3.0, (
        f"chance oracle accuracy {correct}/{n_probes} = {correct/n_probes:.4f} "
        f"is more than 3 sigma from expected p={expected_rate:.4f} (z={z:.2f})"
    )


# covers: eval/subject/oracles::TaskWiperOracle forgets on task boundaries::current task correct
def test_task_wiper_current_task_correct():
    oracle = TaskWiperOracle(seed=0)
    oracle.observe(Observe(position=0, payload={"key": "a", "value": "1"}))
    oracle.observe(Observe(position=1, payload={"key": "b", "value": "2"}))
    assert oracle.answer(Probe(position=2, probe_id="pa", task_id="t0", query="a")) == "1"
    assert oracle.answer(Probe(position=3, probe_id="pb", task_id="t0", query="b")) == "2"


# covers: eval/subject/oracles::TaskWiperOracle forgets on task boundaries::drops on boundary
def test_task_wiper_drops_on_boundary():
    oracle = TaskWiperOracle(seed=0)
    oracle.observe(Observe(position=0, payload={"key": "a", "value": "1"}))
    oracle.observe(Boundary(position=1, kind=BoundaryKind.TASK_SWITCH))
    oracle.observe(Observe(position=2, payload={"key": "b", "value": "2"}))

    assert oracle.answer(Probe(position=3, probe_id="pb", task_id="t1", query="b")) == "2"
    answer_a = oracle.answer(Probe(position=4, probe_id="pa", task_id="t0", query="a"))
    assert answer_a != "1", "the task wiper should have forgotten task 0 after the switch"


# covers: eval/subject/oracles::CheaterOracle exploits probe truth map::learns from probes
def test_cheater_learns_from_probes():
    truth_map = {"x": "correct"}
    oracle = CheaterOracle(truth_map)

    probe = Probe(position=0, probe_id="p", task_id="t", query="x")
    first = oracle.answer(probe)
    assert first == "", "first probe returns empty because cheater has not learned yet"
    second = oracle.answer(probe)
    assert second == "correct", "second probe returns the truth learned from the first"


# covers: eval/subject/oracles::CheaterOracle exploits probe truth map::unknown without truth map
def test_cheater_unknown_without_truth_map():
    oracle = CheaterOracle({})
    probe = Probe(position=0, probe_id="p", task_id="t", query="x")
    assert oracle.answer(probe) == ""
