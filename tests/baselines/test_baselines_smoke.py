"""Smoke tests: every baseline satisfies the Subject protocol end to end.

Each of the five baselines in `eval.baselines` is driven through a tiny
synthetic `split-classify` stream (2 tasks, 2 classes per task, 4 taught
examples per task, 2 probes per task). The point is not accuracy, only
that a baseline implements the full `Subject` protocol without crashing:
`observe`/`answer` process every event, `answer` always returns a string,
`cost()` reports steps taken, and `snapshot`/`restore` round-trip state
so that feeding noise after a snapshot and then restoring reproduces the
pre-noise answer.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from eval.baselines.ewc import EWCBaseline
from eval.baselines.frozen import FrozenBaseline
from eval.baselines.joint import JointBaseline
from eval.baselines.naive import NaiveBaseline
from eval.baselines.replay import ReplayBaseline
from eval.stream.config import StreamConfig
from eval.stream.events import Boundary, BoundaryKind, Observe, Probe
from eval.stream.generators.split_classify import SplitClassifyGenerator
from eval.stream.render import format_features
from eval.subject import CostCounters, Subject

INPUT_DIM = 2
OUTPUT_DIM = 4
TRAIN_ITERATIONS = 10

_STREAM_CONFIG = StreamConfig(
    generator="split-classify",
    params={
        "num_tasks": 2,
        "classes_per_task": 2,
        "examples_per_task": 4,
        "probes_per_task": 2,
    },
)

BASELINE_FACTORIES: dict[str, Callable[[], Subject]] = {
    "naive": lambda: NaiveBaseline(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM),
    "joint": lambda: JointBaseline(
        input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, train_iterations=TRAIN_ITERATIONS
    ),
    "frozen": lambda: FrozenBaseline(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM),
    "ewc": lambda: EWCBaseline(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM),
    "replay": lambda: ReplayBaseline(
        input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, train_iterations=TRAIN_ITERATIONS
    ),
}

BASELINE_NAMES = sorted(BASELINE_FACTORIES)


def _stream_items(seed: int = 0):
    return list(SplitClassifyGenerator().generate(_STREAM_CONFIG, seed=seed))


def _drive(baseline: Subject, items) -> list[str]:
    """Feed `items` to `baseline` in order, returning every probe answer."""
    answers: list[str] = []
    for item in items:
        event = item.event
        if isinstance(event, Probe):
            answers.append(baseline.answer(event))
        elif isinstance(event, (Observe, Boundary)):
            baseline.observe(event)
    return answers


def _noise_events(start_position: int, count: int = 3) -> list[Observe]:
    """Build fabricated `Observe` events that perturb, but don't crash, a baseline.

    Reuses the real stream's label vocabulary (`task0-class0`/`task0-class1`)
    rather than inventing new labels: `JointBaseline`/`ReplayBaseline` cap
    `_label_to_index` at `output_dim`, which the tiny stream already fills,
    so a brand-new label would sit in the example buffer with no index and
    blow up the next retrain.
    """
    labels = ["task0-class0", "task0-class1"]
    events = []
    for i in range(count):
        events.append(
            Observe(
                position=start_position + i,
                payload={
                    "features": [9.9 + i, -9.9 - i],
                    "label": labels[i % len(labels)],
                    "source": None,
                },
            )
        )
    return events


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baseline_implements_subject_protocol(name: str) -> None:
    baseline = BASELINE_FACTORIES[name]()
    assert isinstance(baseline, Subject)


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baseline_processes_tiny_stream_without_crashing(name: str) -> None:
    baseline = BASELINE_FACTORIES[name]()
    items = _stream_items()

    answers = _drive(baseline, items)

    probe_count = sum(1 for item in items if isinstance(item.event, Probe))
    assert len(answers) == probe_count
    assert probe_count > 0
    for answer in answers:
        assert isinstance(answer, str)


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baseline_cost_reports_steps_after_processing(name: str) -> None:
    baseline = BASELINE_FACTORIES[name]()
    items = _stream_items()

    _drive(baseline, items)

    cost = baseline.cost()
    assert isinstance(cost, CostCounters)
    assert cost.steps > 0


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baseline_snapshot_restore_round_trips_answer(name: str) -> None:
    baseline = BASELINE_FACTORIES[name]()
    items = _stream_items()
    _drive(baseline, items)

    probe = Probe(
        position=10_000,
        probe_id="snapshot-probe",
        task_id="task0",
        query=format_features([5.0, 0.0]),
    )
    answer_before = baseline.answer(probe)

    snapshot = baseline.snapshot()

    for noise_event in _noise_events(start_position=10_001):
        baseline.observe(noise_event)
    baseline.observe(Boundary(position=10_010, kind=BoundaryKind.TASK_SWITCH))
    baseline.answer(probe)  # perturb whatever state answer() itself mutates

    baseline.restore(snapshot)

    answer_after_restore = baseline.answer(probe)
    assert answer_after_restore == answer_before
