import dataclasses
import itertools
from pathlib import Path

import pytest

from eval.stream.config import StreamConfig
from eval.stream.events import Boundary, BoundaryKind, Idle, Observe, Probe
from eval.stream.generators.assoc import AssocGenerator
from eval.stream.generators.difficulty_mix import DifficultyMixGenerator
from eval.stream.generators.split_classify import SplitClassifyGenerator
from eval.stream.truth import ProbeTruth, StreamItem, harness_view, subject_view


def _probe(position=1, probe_id="p1", task_id="t1", query="q"):
    return Probe(position=position, probe_id=probe_id, task_id=task_id, query=query)


def test_probe_truth_holds_answer():
    truth = ProbeTruth(answer="42")
    assert truth.answer == "42"


# covers: eval/events::ProbeTruth immutability and optional difficulty::difficulty is optional
def test_probe_truth_difficulty_defaults_to_none():
    truth = ProbeTruth(answer="42")
    assert truth.difficulty is None


def test_probe_truth_is_immutable():
    truth = ProbeTruth(answer="42")
    with pytest.raises(dataclasses.FrozenInstanceError):
        truth.answer = "7"


def test_stream_item_wrapping_probe_carries_truth():
    truth = ProbeTruth(answer="42")
    item = StreamItem(event=_probe(), truth=truth)
    assert item.truth is truth


def test_stream_item_wrapping_observe_carries_none():
    item = StreamItem(event=Observe(position=1, payload="hi"), truth=None)
    assert item.truth is None


def test_stream_item_wrapping_idle_carries_none():
    item = StreamItem(event=Idle(position=1, budget=5), truth=None)
    assert item.truth is None


def test_stream_item_wrapping_boundary_carries_none():
    item = StreamItem(
        event=Boundary(position=1, kind=BoundaryKind.SESSION_END), truth=None
    )
    assert item.truth is None


def test_stream_item_is_immutable():
    item = StreamItem(event=Observe(position=1, payload="hi"), truth=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.truth = None


# covers: eval/events::StreamItem truth pairing invariant::probe without truth
def test_probe_without_truth_is_rejected():
    with pytest.raises(ValueError, match="must carry a ProbeTruth"):
        StreamItem(event=_probe(), truth=None)


# covers: eval/events::StreamItem truth pairing invariant::non-probe with truth
def test_non_probe_with_truth_is_rejected():
    with pytest.raises(ValueError, match="may carry a ProbeTruth"):
        StreamItem(
            event=Observe(position=1, payload="hi"), truth=ProbeTruth(answer="42")
        )


def test_subject_view_never_yields_probe_truth():
    items = [
        StreamItem(event=Observe(position=1, payload="hi"), truth=None),
        StreamItem(event=_probe(position=2), truth=ProbeTruth(answer="42")),
    ]
    result = list(subject_view(items))
    assert all(not isinstance(entry, ProbeTruth) for entry in result)


# covers: eval/events::subject_view hides truth and hidden boundaries::probe answer never leaks
def test_subject_view_never_leaks_answer_string():
    items = [
        StreamItem(event=_probe(position=1), truth=ProbeTruth(answer="secret-42")),
    ]
    result = list(subject_view(items))
    assert "secret-42" not in repr(result)


# covers: eval/events::subject_view hides truth and hidden boundaries::hidden boundary omitted
def test_subject_view_omits_hidden_boundary():
    hidden = Boundary(
        position=1, kind=BoundaryKind.TASK_SWITCH, hidden_from_subject=True
    )
    items = [StreamItem(event=hidden, truth=None)]
    result = list(subject_view(items))
    assert result == []


# covers: eval/events::harness_view yields everything unfiltered::hidden boundary included for harness
def test_harness_view_includes_hidden_boundary():
    hidden = Boundary(
        position=1, kind=BoundaryKind.TASK_SWITCH, hidden_from_subject=True
    )
    items = [StreamItem(event=hidden, truth=None)]
    result = list(harness_view(items))
    assert result == [items[0]]


def test_visible_boundary_appears_in_both_views():
    visible = Boundary(position=1, kind=BoundaryKind.TASK_SWITCH)
    items = [StreamItem(event=visible, truth=None)]
    assert list(subject_view(items)) == [visible]
    assert list(harness_view(items)) == [items[0]]


def test_views_agree_on_ordering_and_positions():
    hidden = Boundary(
        position=2, kind=BoundaryKind.TASK_SWITCH, hidden_from_subject=True
    )
    items = [
        StreamItem(event=Observe(position=1, payload="a"), truth=None),
        StreamItem(event=hidden, truth=None),
        StreamItem(event=Observe(position=3, payload="b"), truth=None),
    ]
    subject_positions = [entry.position for entry in subject_view(items)]
    harness_positions = [entry.event.position for entry in harness_view(items)]
    assert subject_positions == [1, 3]
    assert harness_positions == [1, 2, 3]


# covers: eval/events::StreamItem truth pairing invariant::Idle with truth rejected
def test_idle_with_truth_is_rejected():
    with pytest.raises(ValueError, match="may carry a ProbeTruth"):
        StreamItem(event=Idle(position=1, budget=5), truth=ProbeTruth(answer="42"))


# covers: eval/events::StreamItem truth pairing invariant::Boundary with truth rejected
def test_boundary_with_truth_is_rejected():
    with pytest.raises(ValueError, match="may carry a ProbeTruth"):
        StreamItem(
            event=Boundary(position=1, kind=BoundaryKind.SESSION_END),
            truth=ProbeTruth(answer="42"),
        )


# covers: eval/events::StreamItem truth pairing invariant::Probe with truth accepted
def test_probe_with_truth_is_accepted():
    truth = ProbeTruth(answer="42")
    item = StreamItem(event=_probe(), truth=truth)
    assert item.truth == truth


# covers: eval/events::ProbeTruth immutability and optional difficulty::ProbeTruth is frozen
def test_probe_truth_frozen_raises_on_assignment():
    truth = ProbeTruth(answer="42")
    with pytest.raises(dataclasses.FrozenInstanceError):
        truth.answer = "changed"


# covers: eval/events::ProbeTruth immutability and optional difficulty::difficulty defaults to None
def test_probe_truth_difficulty_default_is_none():
    truth = ProbeTruth(answer="x")
    assert truth.difficulty is None


# covers: eval/events::ProbeTruth immutability and optional difficulty::difficulty can be set explicitly
def test_probe_truth_difficulty_can_be_set():
    truth = ProbeTruth(answer="x", difficulty=3)
    assert truth.difficulty == 3


# covers: eval/events::subject_view hides truth and hidden boundaries::yields bare Events not StreamItems
def test_subject_view_yields_bare_events():
    items = [
        StreamItem(event=Observe(position=1, payload="hi"), truth=None),
        StreamItem(event=_probe(position=2), truth=ProbeTruth(answer="42")),
    ]
    result = list(subject_view(items))
    assert all(isinstance(entry, (Observe, Probe, Idle, Boundary)) for entry in result)
    assert all(not isinstance(entry, StreamItem) for entry in result)


# covers: eval/events::subject_view hides truth and hidden boundaries::preserves position order
def test_subject_view_preserves_position_order():
    items = [
        StreamItem(event=Observe(position=0, payload="a"), truth=None),
        StreamItem(event=Observe(position=1, payload="b"), truth=None),
        StreamItem(event=Observe(position=2, payload="c"), truth=None),
        StreamItem(event=Observe(position=3, payload="d"), truth=None),
    ]
    result = list(subject_view(items))
    positions = [entry.position for entry in result]
    assert all(a < b for a, b in itertools.pairwise(positions))


# covers: eval/events::subject_view hides truth and hidden boundaries::visible boundary included
def test_subject_view_includes_visible_boundary():
    visible = Boundary(
        position=1, kind=BoundaryKind.TASK_SWITCH, hidden_from_subject=False
    )
    items = [StreamItem(event=visible, truth=None)]
    result = list(subject_view(items))
    assert visible in result


# covers: eval/events::harness_view yields everything unfiltered::truth preserved in harness view
def test_harness_view_preserves_truth():
    truth = ProbeTruth(answer="42")
    items = [StreamItem(event=_probe(), truth=truth)]
    result = list(harness_view(items))
    assert result[0].truth == truth


# covers: eval/events::narration and hostile flags exist for future stages::no current consumer
def test_no_generator_sets_narration_or_hostile(monkeypatch):
    fixture_dir = Path(__file__).parent / "fixtures"
    monkeypatch.setenv("PENSIVE_CORPUS_DIR", str(fixture_dir))

    configs = {
        AssocGenerator: StreamConfig(
            generator="assoc",
            params={
                "num_pairs": 2,
                "recall_distances": [1],
                "filler_density": 1.0,
                "max_distance": 5,
                "corpus": "corpus_fixture",
            },
        ),
        SplitClassifyGenerator: StreamConfig(
            generator="split-classify",
            params={
                "num_tasks": 2,
                "classes_per_task": 2,
                "examples_per_task": 2,
                "probes_per_task": 1,
            },
        ),
        DifficultyMixGenerator: StreamConfig(
            generator="difficulty-mix",
            params={
                "num_items": 3,
                "difficulty_levels": [1, 2],
                "probe_rate": 0.5,
            },
        ),
    }

    for generator_class, config in configs.items():
        generator = generator_class()
        for item in generator.generate(config=config, seed=0):
            assert item.event.narration is None, (
                f"{generator.name} set narration on position {item.event.position}"
            )
            assert item.event.hostile is False, (
                f"{generator.name} set hostile on position {item.event.position}"
            )
