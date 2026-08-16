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


# covers: eval/events::Event immutability::attribute assignment on a constructed Observe
def test_observe_is_immutable():
    event = Observe(position=1, payload="hello")
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.payload = "changed"


# covers: eval/events::Event immutability::attribute assignment on a constructed Probe
def test_probe_is_immutable():
    event = Probe(position=1, probe_id="p1", task_id="t1", query="q")
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.query = "changed"


# covers: eval/events::Event immutability::attribute assignment on a constructed Idle
def test_idle_is_immutable():
    event = Idle(position=1, budget=10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.budget = 20


# covers: eval/events::Event immutability::attribute assignment on a constructed Boundary
def test_boundary_is_immutable():
    event = Boundary(position=1, kind=BoundaryKind.SESSION_END)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.kind = BoundaryKind.TASK_SWITCH


# covers: eval/events::Event required fields::Probe without probe_id
def test_probe_without_probe_id_raises_type_error():
    with pytest.raises(TypeError):
        Probe(position=1, task_id="t1", query="q")


# covers: eval/events::Event required fields::Probe without task_id
def test_probe_without_task_id_raises_type_error():
    with pytest.raises(TypeError):
        Probe(position=1, probe_id="p1", query="q")


# covers: eval/events::Event required fields::Probe without query
def test_probe_without_query_raises_type_error():
    with pytest.raises(TypeError):
        Probe(position=1, probe_id="p1", task_id="t1")


# covers: eval/events::Event required fields::Idle without budget
def test_idle_without_budget_raises_type_error():
    with pytest.raises(TypeError):
        Idle(position=1)


# covers: eval/events::Event required fields::Observe without payload
def test_observe_without_payload_raises_type_error():
    with pytest.raises(TypeError):
        Observe(position=1)


# covers: eval/events::Event required fields::Boundary without kind
def test_boundary_without_kind_raises_type_error():
    with pytest.raises(TypeError):
        Boundary(position=1)


# covers: eval/events::Event optional-field defaults::narration defaults to None
def test_observe_narration_defaults_to_none():
    event = Observe(position=1, payload="hello")
    assert event.narration is None


# covers: eval/events::Event optional-field defaults::hostile defaults to False
def test_observe_hostile_defaults_to_false():
    event = Observe(position=1, payload="hello")
    assert event.hostile is False


# covers: eval/events::Event optional-field defaults::hidden_from_subject defaults to False
def test_boundary_hidden_from_subject_defaults_to_false():
    event = Boundary(position=1, kind=BoundaryKind.SESSION_END)
    assert event.hidden_from_subject is False


# covers: eval/events::Event optional-field defaults::narration defaults to None on Probe
def test_probe_narration_defaults_to_none():
    event = Probe(position=1, probe_id="p1", task_id="t1", query="q")
    assert event.narration is None


# covers: eval/events::BoundaryKind string values::session_end string value
def test_boundary_kind_session_end_value():
    assert BoundaryKind.SESSION_END.value == "session_end"


# covers: eval/events::BoundaryKind string values::task_switch string value
def test_boundary_kind_task_switch_value():
    assert BoundaryKind.TASK_SWITCH.value == "task_switch"


# covers: eval/events::BoundaryKind string values::distribution_shift string value
def test_boundary_kind_distribution_shift_value():
    assert BoundaryKind.DISTRIBUTION_SHIFT.value == "distribution_shift"


# covers: eval/events::Event type union::isinstance dispatch covers all four types
def test_event_union_isinstance_dispatch_covers_all_four_types():
    events: list[Event] = [
        Observe(position=1, payload="hello"),
        Probe(position=2, probe_id="p1", task_id="t1", query="q"),
        Idle(position=3, budget=50),
        Boundary(position=4, kind=BoundaryKind.DISTRIBUTION_SHIFT),
    ]
    types = (Observe, Probe, Idle, Boundary)
    for event in events:
        matches = [isinstance(event, t) for t in types]
        assert sum(matches) == 1


# covers: eval/events::Event type union::Event alias accepts all four types
def test_event_alias_accepts_all_four_types():
    instances: list[Event] = [
        Observe(position=1, payload="hello"),
        Probe(position=2, probe_id="p1", task_id="t1", query="q"),
        Idle(position=3, budget=50),
        Boundary(position=4, kind=BoundaryKind.DISTRIBUTION_SHIFT),
    ]
    for instance in instances:
        assert isinstance(instance, (Observe, Probe, Idle, Boundary))
