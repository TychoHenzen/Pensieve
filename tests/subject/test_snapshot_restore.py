from __future__ import annotations

from eval.stream.events import Observe, Probe
from eval.subject.oracles.forgetful import ForgetfulOracle
from eval.subject.oracles.perfect_memory import PerfectMemoryOracle


def _run_snapshot_noise_restore(oracle, signal_events, noise_events, probes):
    for event in signal_events:
        oracle.observe(event)
    state = oracle.snapshot()
    for event in noise_events:
        oracle.observe(event)
    oracle.restore(state)
    return [oracle.answer(p) for p in probes]


def _run_clean(oracle, signal_events, probes):
    for event in signal_events:
        oracle.observe(event)
    return [oracle.answer(p) for p in probes]


# covers: eval/subject::Snapshot captures full state::restore erases intermediate observations on PerfectMemoryOracle
def test_perfect_memory_restore_erases_noise():
    signal = [
        Observe(position=0, payload={"key": "a", "value": "1"}),
        Observe(position=1, payload={"key": "b", "value": "2"}),
    ]
    noise = [
        Observe(position=2, payload={"key": "a", "value": "WRONG"}),
        Observe(position=3, payload={"key": "c", "value": "3"}),
    ]
    probes = [
        Probe(position=4, probe_id="pa", task_id="t", query="a"),
        Probe(position=5, probe_id="pb", task_id="t", query="b"),
        Probe(position=6, probe_id="pc", task_id="t", query="c"),
    ]

    restored = _run_snapshot_noise_restore(
        PerfectMemoryOracle(), signal, noise, probes
    )
    clean = _run_clean(PerfectMemoryOracle(), signal, probes)
    assert restored == clean


# covers: eval/subject::Snapshot captures full state::restore erases intermediate observations on ForgetfulOracle
def test_forgetful_restore_erases_noise():
    signal = [
        Observe(position=0, payload={"key": "a", "value": "1"}),
        Observe(position=1, payload={"key": "b", "value": "2"}),
    ]
    noise = [
        Observe(position=2, payload={"key": "z", "value": "99"}),
    ]
    probes = [
        Probe(position=3, probe_id="pb", task_id="t", query="b"),
        Probe(position=4, probe_id="pz", task_id="t", query="z"),
    ]

    restored = _run_snapshot_noise_restore(
        ForgetfulOracle(), signal, noise, probes
    )
    clean = _run_clean(ForgetfulOracle(), signal, probes)
    assert restored == clean
