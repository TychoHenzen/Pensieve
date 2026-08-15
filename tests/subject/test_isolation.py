from __future__ import annotations

from eval.stream.events import Observe, Probe
from eval.subject.isolation import isolated_answer
from eval.subject.oracles.cheater import CheaterOracle
from eval.subject.oracles.perfect_memory import PerfectMemoryOracle


def _make_stream():
    return [
        Observe(position=0, payload={"key": "cat", "value": "meow"}),
        Probe(position=1, probe_id="p0", task_id="t", query="cat"),
        Observe(position=2, payload={"key": "dog", "value": "bark"}),
        Probe(position=3, probe_id="p1", task_id="t", query="cat"),
        Probe(position=4, probe_id="p2", task_id="t", query="dog"),
    ]


# covers: eval/subject/isolation::isolated_answer prevents probe-based learning::cheater detected by isolation
def test_cheater_detected_by_isolation():
    truth_map = {"cat": "meow", "dog": "bark"}

    stream = [
        Probe(position=0, probe_id="p0", task_id="t", query="cat"),
        Probe(position=1, probe_id="p1", task_id="t", query="cat"),
        Probe(position=2, probe_id="p2", task_id="t", query="dog"),
        Probe(position=3, probe_id="p3", task_id="t", query="dog"),
    ]

    iso = CheaterOracle(truth_map)
    non_iso = CheaterOracle(truth_map)

    iso_answers = []
    non_iso_answers = []

    for event in stream:
        iso_answers.append(isolated_answer(iso, event))
        non_iso_answers.append(non_iso.answer(event))

    assert non_iso_answers != iso_answers, (
        "the cheater should produce different results with and without "
        "isolation, proving isolation caught the probe learning"
    )


# covers: eval/subject/isolation::isolated_answer prevents probe-based learning::honest subject unaffected
def test_honest_subject_identical_with_and_without_isolation():
    stream = _make_stream()

    isolated_oracle = PerfectMemoryOracle()
    non_isolated_oracle = PerfectMemoryOracle()

    isolated_answers = []
    non_isolated_answers = []

    for event in stream:
        if isinstance(event, Observe):
            isolated_oracle.observe(event)
            non_isolated_oracle.observe(event)
        elif isinstance(event, Probe):
            isolated_answers.append(isolated_answer(isolated_oracle, event))
            non_isolated_answers.append(non_isolated_oracle.answer(event))

    assert isolated_answers == non_isolated_answers


# covers: eval/subject/isolation::CheaterOracle learns from probes when unprotected::learns on second probe
def test_cheater_learns_from_probes_when_unprotected():
    truth_map = {"cat": "meow"}
    oracle = CheaterOracle(truth_map)

    probe = Probe(position=0, probe_id="p0", task_id="t", query="cat")
    first = oracle.answer(probe)
    assert first == "", "first probe should return empty because cheater has not learned yet"
    second = oracle.answer(probe)
    assert second == "meow", "second probe should return the truth learned from the first"


# covers: eval/subject/isolation::CheaterOracle learns from probes when unprotected::cannot learn when isolated
def test_cheater_cannot_learn_from_probes_when_isolated():
    truth_map = {"cat": "meow"}
    oracle = CheaterOracle(truth_map)

    probe = Probe(position=0, probe_id="p0", task_id="t", query="cat")
    first = isolated_answer(oracle, probe)
    assert first == "", "first isolated probe returns empty because cheater has not learned yet"

    second = isolated_answer(oracle, probe)
    assert second == "", (
        "second isolated probe also returns empty because isolation "
        "restored state after the first probe, erasing what it learned"
    )
