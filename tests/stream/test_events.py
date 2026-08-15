import dataclasses

import pytest

from eval.stream.events import (
    BoundaryKind,
    Boundary,
    Event,
    Idle,
    Observe,
    Probe,
)
from eval.stream.render import render_event


def test_observe_constructs_with_required_fields():
    event = Observe(position=1, payload="hello")
    assert event.position == 1
    assert event.payload == "hello"


def test_probe_constructs_with_required_fields():
    event = Probe(
        position=5,
        probe_id="p1",
        task_id="t1",
        query="what is key-1?",
        teaching_position=2,
    )
    assert event.position == 5
    assert event.probe_id == "p1"
    assert event.task_id == "t1"
    assert event.query == "what is key-1?"
    assert event.teaching_position == 2


def test_idle_constructs_with_required_fields():
    event = Idle(position=10, budget=100)
    assert event.position == 10
    assert event.budget == 100


def test_boundary_constructs_with_required_fields():
    event = Boundary(position=20, kind=BoundaryKind.SESSION_END)
    assert event.position == 20
    assert event.kind == BoundaryKind.SESSION_END


# covers: eval/events::Event immutability::attribute assignment on a constructed event
def test_events_are_immutable():
    event = Observe(position=1, payload="hello")
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.position = 2


# covers: eval/events::Event optional-field defaults::default construction
def test_narration_defaults_to_none():
    event = Observe(position=1, payload="hello")
    assert event.narration is None


def test_hostile_defaults_to_false():
    event = Observe(position=1, payload="hello")
    assert event.hostile is False


def test_hidden_from_subject_defaults_to_false():
    event = Boundary(position=20, kind=BoundaryKind.TASK_SWITCH)
    assert event.hidden_from_subject is False


# covers: eval/events::Event required fields::omitted required field
def test_probe_requires_probe_id():
    with pytest.raises(TypeError):
        Probe(position=1, task_id="t1", query="q")


def test_probe_requires_task_id():
    with pytest.raises(TypeError):
        Probe(position=1, probe_id="p1", query="q")


def test_probe_requires_query():
    with pytest.raises(TypeError):
        Probe(position=1, probe_id="p1", task_id="t1")


def test_idle_requires_budget():
    with pytest.raises(TypeError):
        Idle(position=1)


def test_observe_requires_payload():
    with pytest.raises(TypeError):
        Observe(position=1)


def test_boundary_requires_kind():
    with pytest.raises(TypeError):
        Boundary(position=1)


# covers: eval/events::BoundaryKind string values::rendered boundary text
def test_boundary_kind_value_strings():
    assert BoundaryKind.SESSION_END.value == "session_end"
    assert BoundaryKind.TASK_SWITCH.value == "task_switch"
    assert BoundaryKind.DISTRIBUTION_SHIFT.value == "distribution_shift"


# covers: eval/events::BoundaryKind string values::rendered boundary text
def test_boundary_kind_value_is_literal_string():
    for member in BoundaryKind:
        assert isinstance(member.value, str)
        assert member.value == member.name.lower()


# covers: eval/events::Event type union::isinstance dispatch
def test_event_union_isinstance_dispatch():
    events: list[Event] = [
        Observe(position=1, payload="hello"),
        Probe(position=2, probe_id="p1", task_id="t1", query="q", teaching_position=None),
        Idle(position=3, budget=50),
        Boundary(position=4, kind=BoundaryKind.DISTRIBUTION_SHIFT),
    ]
    types = (Observe, Probe, Idle, Boundary)
    for event in events:
        matches = [isinstance(event, t) for t in types]
        assert sum(matches) == 1


def test_event_union_accepts_all_four_kinds():
    events: list[Event] = [
        Observe(position=1, payload="hello"),
        Probe(position=2, probe_id="p1", task_id="t1", query="q", teaching_position=None),
        Idle(position=3, budget=50),
        Boundary(position=4, kind=BoundaryKind.DISTRIBUTION_SHIFT),
    ]
    assert len(events) == 4
    assert all(isinstance(event.position, int) for event in events)


# covers: eval/events::BoundaryKind string values::rendered boundary text
def test_rendered_boundary_contains_kind_string():
    event = Boundary(position=1, kind=BoundaryKind.TASK_SWITCH)
    rendered = render_event(event)
    assert "task_switch" in rendered
