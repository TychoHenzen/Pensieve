"""Integration tests: run each oracle on a hand-built stream through the
full pipeline (subject.observe/answer -> ProbeLogEntry log -> metrics) and
check the results against paper-derived expectations.

Unlike the per-metric unit tests, these tests do not hand-construct
ProbeLogEntry lists directly. They drive a real Subject implementation
over Observe/Probe events and build the log from its answers, so the
path from oracle behavior through to metric output is exercised end to
end.
"""

from __future__ import annotations

from eval.metrics import ProbeLogEntry
from eval.metrics.accuracy import task_accuracy
from eval.metrics.compute import compute_per_input
from eval.metrics.first_use import time_to_first_use
from eval.metrics.retention import retention_matrix
from eval.stream.events import Boundary, BoundaryKind, Observe, Probe
from eval.stream.vocab import VOCAB
from eval.subject import Subject
from eval.subject.oracles.chance import ChanceOracle
from eval.subject.oracles.forgetful import ForgetfulOracle
from eval.subject.oracles.perfect_memory import PerfectMemoryOracle
from eval.subject.oracles.task_wiper import TaskWiperOracle


def _run(subject: Subject, events: list) -> list[ProbeLogEntry]:
    """Drive *subject* over *events* and build a probe log.

    Observe/Boundary events update the subject's state; Probe events are
    answered and scored but never fed back into observe(), matching the
    isolation requirement that probes are held out from learning.
    """
    log: list[ProbeLogEntry] = []
    for event in events:
        if isinstance(event, Probe):
            answer = subject.answer(event)
            correct = answer == event.narration
            log.append(
                ProbeLogEntry(
                    position=event.position,
                    probe_id=event.probe_id,
                    task_id=event.task_id,
                    teaching_position=event.teaching_position or 0,
                    correct=correct,
                    cost_counters=subject.cost(),
                )
            )
        else:
            subject.observe(event)
    return log


# covers: eval/metrics :: Oracle metric validation :: perfect-memory oracle scores 1.0 pooled accuracy
def test_perfect_memory_scores_full_accuracy_and_positive_compute():
    facts = [("cat", "able"), ("dog", "acid"), ("cow", "aged")]
    events = []
    for pos, (key, value) in enumerate(facts):
        events.append(Observe(position=pos, payload={"key": key, "value": value}))
    for pos, (key, value) in enumerate(facts, start=len(facts)):
        events.append(
            Probe(
                position=pos,
                probe_id=f"p-{key}",
                task_id="assoc",
                query=key,
                narration=value,
                teaching_position=facts.index((key, value)),
            )
        )

    subject = PerfectMemoryOracle()
    log = _run(subject, events)

    accuracy = task_accuracy(log)
    assert accuracy.pooled == 1.0

    result = time_to_first_use(log, cutoff=100)
    assert result.cutoff_share == 0.0
    for teaching_pos, (probe_pos, (key, _value)) in enumerate(
        zip(range(len(facts), len(facts) * 2), facts, strict=True)
    ):
        expected_distance = probe_pos - teaching_pos
        assert result.per_fact[f"p-{key}"] == expected_distance

    compute = compute_per_input(log)
    assert len(compute.steps) == len(facts)
    assert all(step > 0 for step in compute.steps)


# covers: eval/metrics :: Oracle metric validation :: forgetful oracle retains only the last fact taught
def test_forgetful_oracle_only_recalls_most_recent_fact():
    facts = [("cat", "able"), ("dog", "acid"), ("cow", "aged")]
    events = []
    for pos, (key, value) in enumerate(facts):
        events.append(Observe(position=pos, payload={"key": key, "value": value}))
    for pos, (key, value) in enumerate(facts, start=len(facts)):
        events.append(
            Probe(
                position=pos,
                probe_id=f"p-{key}",
                task_id="assoc",
                query=key,
                narration=value,
                teaching_position=facts.index((key, value)),
            )
        )

    subject = ForgetfulOracle()
    log = _run(subject, events)

    accuracy = task_accuracy(log)
    # Only the last taught fact ("cow") survives; the first two are lost.
    assert accuracy.pooled == 1 / 3
    assert accuracy.pooled < 1.0


# covers: eval/metrics :: Oracle metric validation :: forgetful oracle retention drops to chance beyond window
def test_forgetful_oracle_retention_drops_to_chance_beyond_window():
    # Task 0: teach one fact, probe immediately after teaching (phase 0).
    # Task 1: teach a second fact, which evicts task 0's fact from the
    # forgetful oracle's single-slot memory. Probe task 0 again in phase 1
    # (distance > 1 from its teaching) and task 1 in phase 1.
    task0_key, task0_value = "cat", "able"
    task1_key, task1_value = "dog", "acid"

    subject = ForgetfulOracle()

    phase0_events = [
        Observe(position=0, payload={"key": task0_key, "value": task0_value}),
        Probe(
            position=1,
            probe_id="p-cat",
            task_id="task0",
            query=task0_key,
            narration=task0_value,
            teaching_position=0,
        ),
    ]
    phase0_log = _run(subject, phase0_events)

    # Phase 0: probed immediately after teaching, within the 1-event window.
    assert phase0_log[0].correct is True

    phase1_events = [
        Observe(position=2, payload={"key": task1_key, "value": task1_value}),
        Probe(
            position=3,
            probe_id="p-cat",
            task_id="task0",
            query=task0_key,
            narration=task0_value,
            teaching_position=0,
        ),
        Probe(
            position=4,
            probe_id="p-dog",
            task_id="task1",
            query=task1_key,
            narration=task1_value,
            teaching_position=2,
        ),
    ]
    phase1_log = _run(subject, phase1_events)

    task0_probe = next(e for e in phase1_log if e.task_id == "task0")
    task1_probe = next(e for e in phase1_log if e.task_id == "task1")

    # Beyond the 1-event window, task 0's fact was evicted: accuracy is at
    # chance (the oracle returns "" for unknown keys, never the vocab word).
    assert task0_probe.correct is False
    # Task 1 was just taught and is still within the window.
    assert task1_probe.correct is True

    phases = [phase0_log, phase1_log]
    R = retention_matrix(phases, task_order=["task0", "task1"])
    assert R[0][0] == 1.0
    assert R[0][1] == 0.0


# covers: eval/metrics :: Oracle metric validation :: chance oracle accuracy tracks the vocabulary chance rate
def test_chance_oracle_accuracy_matches_chance_rate_within_tolerance():
    chance_rate = 1.0 / len(VOCAB)
    num_probes = 500
    events = [
        Probe(
            position=pos,
            probe_id=f"p-{pos}",
            task_id="assoc",
            query="cat",
            narration="able",
            teaching_position=0,
        )
        for pos in range(num_probes)
    ]

    subject = ChanceOracle(seed=42)
    log = _run(subject, events)

    accuracy = task_accuracy(log)
    # With a 299-word vocabulary, expected accuracy is ~0.33%. Allow a
    # generous absolute tolerance given the small sample size.
    assert abs(accuracy.pooled - chance_rate) < 0.02


# covers: eval/metrics :: Oracle metric validation :: task-wiper oracle forgets prior tasks on TASK_SWITCH boundary
def test_task_wiper_forgets_on_task_switch_but_not_within_task():
    task0_key, task0_value = "cat", "able"
    task1_key, task1_value = "dog", "acid"

    subject = TaskWiperOracle(seed=0)

    events = [
        Observe(position=0, payload={"key": task0_key, "value": task0_value}),
        Boundary(position=1, kind=BoundaryKind.TASK_SWITCH),
        Observe(position=2, payload={"key": task1_key, "value": task1_value}),
        # Task 0 probe after the task-switch boundary: facts were wiped.
        Probe(
            position=3,
            probe_id="p-cat",
            task_id="task0",
            query=task0_key,
            narration=task0_value,
            teaching_position=0,
        ),
        # Task 1 probe immediately after teaching, no wipe in between.
        Probe(
            position=4,
            probe_id="p-dog",
            task_id="task1",
            query=task1_key,
            narration=task1_value,
            teaching_position=2,
        ),
    ]

    log = _run(subject, events)

    task0_probe = next(e for e in log if e.task_id == "task0")
    task1_probe = next(e for e in log if e.task_id == "task1")

    assert task0_probe.correct is False
    assert task1_probe.correct is True

    accuracy = task_accuracy(log)
    assert accuracy.per_task["task0"] == 0.0
    assert accuracy.per_task["task1"] == 1.0
