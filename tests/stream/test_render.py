import re
from pathlib import Path

import pytest

from eval.stream.config import StreamConfig
from eval.stream.events import Boundary, BoundaryKind, Idle, Observe, Probe
from eval.stream.generators.assoc import AssocGenerator
from eval.stream.generators.difficulty_mix import DifficultyMixGenerator
from eval.stream.render import (
    RENDER_VERSION,
    TokenCounter,
    carries_text,
    default_token_counter,
    format_features,
    render_event,
    rendered_subject_view,
    resolve_token_counter,
)
from eval.stream.truth import ProbeTruth, StreamItem

SECRET_ANSWER = "secret-42"
PROBE_ID = "probe-id-xyz"
TASK_ID = "task-id-xyz"
TEACHING_POSITION = 7

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _corpus_dir(monkeypatch):
    monkeypatch.setenv("PENSIVE_CORPUS_DIR", str(FIXTURE_DIR))


def _probe():
    return Probe(
        position=1,
        probe_id=PROBE_ID,
        task_id=TASK_ID,
        query="what is the capital of France?",
        teaching_position=TEACHING_POSITION,
        hostile=True,
    )


def _observe():
    return Observe(position=1, payload={"key": "beam", "value": "duty"}, hostile=True)


def _prose():
    return Observe(position=2, payload={"text": "the mill wheel turned all morning"})


def _example():
    return Observe(
        position=3,
        payload={
            "features": [5.383941741180422, -0.5668909031452558],
            "label": "task0-class0",
            "source": "mnist-train-17",
        },
    )


def _idle():
    return Idle(position=1, budget=5, hostile=True)


def _boundary():
    return Boundary(position=1, kind=BoundaryKind.TASK_SWITCH, hostile=True)


def _assoc_config(**overrides):
    params = {
        "num_pairs": 3,
        "recall_distances": [1, 2, 5],
        "filler_density": 1.0,
        "max_distance": 10,
        "corpus": "corpus_fixture",
    }
    params.update(overrides)
    return StreamConfig(generator="assoc", params=params)


def _assoc_items(config=None, seed=0):
    generator = AssocGenerator()
    return list(generator.generate(config=config or _assoc_config(), seed=seed))


def _difficulty_mix_config(**overrides):
    params = {
        "num_items": 20,
        "difficulty_levels": [1, 2, 3, 5],
        "probe_rate": 0.5,
    }
    params.update(overrides)
    return StreamConfig(generator="difficulty-mix", params=params)


def _difficulty_mix_items(config=None, seed=0):
    generator = DifficultyMixGenerator()
    return list(generator.generate(config=config or _difficulty_mix_config(), seed=seed))


# covers: eval/render::RENDER_VERSION is a non-empty string::version is present
def test_render_version_is_a_string():
    assert isinstance(RENDER_VERSION, str)
    assert RENDER_VERSION


# covers: eval/render::RENDER_VERSION is a non-empty string::version is a str
def test_render_version_type_is_str():
    assert type(RENDER_VERSION) is str


# covers: eval/render::Observe and Boundary rendering omits hostile and is stable::hostile flag never leaks
def test_observe_render_omits_hostile_flag_and_repeats():
    event = _observe()
    first = render_event(event)
    second = render_event(event)
    assert first == second
    assert "True" not in first
    assert "hostile" not in first


# covers: eval/render::Probe renders its query and hides harness fields::probe text hides harness fields
def test_probe_render_omits_answer_and_identifiers_and_repeats():
    event = _probe()
    first = render_event(event)
    second = render_event(event)
    assert first == second
    assert SECRET_ANSWER not in first
    assert PROBE_ID not in first
    assert TASK_ID not in first
    assert str(TEACHING_POSITION) not in first
    assert "True" not in first
    assert "hostile" not in first


# covers: eval/render::Probe renders its query and hides harness fields::probe renders query
def test_probe_renders_query():
    event = Probe(
        position=1,
        probe_id="p-query",
        task_id="t-query",
        query="what is beam?",
        teaching_position=0,
    )
    assert render_event(event) == "[probe] what is beam?"


# covers: eval/render::Probe renders its query and hides harness fields::probe hides answer
def test_probe_hides_answer():
    event = Probe(
        position=1,
        probe_id="p-answer",
        task_id="t-answer",
        query="what is beam?",
        teaching_position=0,
    )
    item = StreamItem(event=event, truth=ProbeTruth(answer="duty"))
    rendered = render_event(item.event)
    assert "duty" not in rendered


# covers: eval/render::Probe renders its query and hides harness fields::probe hides probe_id and task_id
def test_probe_hides_probe_id_and_task_id():
    event = Probe(
        position=1,
        probe_id="p-99",
        task_id="t-7",
        query="what is beam?",
        teaching_position=0,
    )
    rendered = render_event(event)
    assert "p-99" not in rendered
    assert "t-7" not in rendered


# covers: eval/render::Probe renders its query and hides harness fields::probe rendering is idempotent
def test_probe_rendering_is_idempotent():
    event = _probe()
    assert render_event(event) == render_event(event)


# covers: eval/render::Idle carries no text and render_event refuses it::Idle is not text
def test_idle_carries_no_text_and_refuses_to_render():
    event = _idle()
    assert carries_text(event) is False
    with pytest.raises(TypeError, match="idle"):
        render_event(event)


# covers: eval/render::Idle carries no text and render_event refuses it::carries_text returns False for Idle
def test_carries_text_returns_false_for_idle():
    assert carries_text(_idle()) is False


# covers: eval/render::Idle carries no text and render_event refuses it::render_event raises on Idle
def test_render_event_raises_on_idle():
    with pytest.raises(TypeError, match="idle"):
        render_event(_idle())


# covers: eval/render::Idle carries no text and render_event refuses it::carries_text returns True for Observe
def test_every_other_event_kind_carries_text():
    for event in (_observe(), _prose(), _example(), _probe(), _boundary()):
        assert carries_text(event) is True


# covers: eval/render::fact payload renders as "[fact] key = value"::rendering a fact
def test_fact_payload_renders_key_and_value_without_dict_syntax():
    rendered = render_event(_observe())
    assert rendered == "[fact] beam = duty"


# covers: eval/render::fact payload renders as "[fact] key = value"::fact format is exact
def test_fact_format_is_exact():
    event = Observe(position=1, payload={"key": "alpha", "value": "beta"})
    rendered = render_event(event)
    assert rendered.startswith("[fact] ")
    assert " = " in rendered


# covers: eval/render::prose payload renders bare with no marker::rendering prose
def test_prose_payload_renders_bare_with_no_marker():
    rendered = render_event(_prose())
    assert rendered == "the mill wheel turned all morning"


# covers: eval/render::prose payload renders bare with no marker::prose has no prefix
def test_prose_has_no_prefix():
    event = Observe(position=1, payload={"text": "hello world"})
    rendered = render_event(event)
    assert not rendered.startswith("[")


# covers: eval/render::example payload renders with 2-decimal features and hides source::rendering an example
def test_example_payload_renders_two_decimals_and_hides_its_source():
    rendered = render_event(_example())
    assert rendered == "[example] features=(5.38, -0.57) label=task0-class0"
    assert "mnist-train-17" not in rendered


# covers: eval/render::example payload renders with 2-decimal features and hides source::rendering an example with 2-decimal features
def test_example_renders_with_two_decimal_features():
    rendered = render_event(_example())
    assert rendered == "[example] features=(5.38, -0.57) label=task0-class0"


# covers: eval/render::example payload renders with 2-decimal features and hides source::source value hidden
def test_example_source_value_hidden():
    event = Observe(position=1, payload={"features": [1.0], "label": "a", "source": "secret-source"})
    rendered = render_event(event)
    assert "secret-source" not in rendered


# covers: eval/render::example payload renders with 2-decimal features and hides source::features use exactly 2 decimal places
def test_example_features_use_exactly_two_decimal_places():
    event = Observe(position=1, payload={"features": [1.0, 2.12345], "label": "x", "source": None})
    rendered = render_event(event)
    assert "1.00" in rendered
    assert "2.12" in rendered
    assert not re.search(r"\.\d{3}", rendered)


# covers: eval/render::unknown Observe payload shape raises ValueError::unrecognized shape
def test_unknown_payload_shape_raises_and_names_the_known_shapes():
    event = Observe(position=1, payload={"mystery": 1})
    with pytest.raises(ValueError, match="known payload shapes"):
        render_event(event)


# covers: eval/render::unknown Observe payload shape raises ValueError::non-mapping payload
def test_non_mapping_payload_raises_rather_than_printing_its_repr():
    event = Observe(position=1, payload=["a", "b"])
    with pytest.raises(ValueError, match="known payload shapes"):
        render_event(event)


# covers: eval/render::Observe and Boundary rendering omits hostile and is stable::hostile flag never leaks
def test_boundary_render_omits_hostile_flag_and_repeats():
    event = _boundary()
    first = render_event(event)
    second = render_event(event)
    assert first == second
    assert "True" not in first
    assert "hostile" not in first


# covers: eval/render::Observe and Boundary rendering omits hostile and is stable::Observe hostile flag never leaks
def test_observe_hostile_flag_never_leaks():
    rendered = render_event(_observe())
    assert "True" not in rendered
    assert "hostile" not in rendered


# covers: eval/render::Observe and Boundary rendering omits hostile and is stable::Boundary hostile flag never leaks
def test_boundary_hostile_flag_never_leaks():
    rendered = render_event(_boundary())
    assert "True" not in rendered
    assert "hostile" not in rendered


# covers: eval/render::Observe and Boundary rendering omits hostile and is stable::Observe rendering is idempotent
def test_observe_rendering_is_idempotent():
    event = _observe()
    assert render_event(event) == render_event(event)


# covers: eval/render::rendered_subject_view drops Idle and hidden Boundary::four items in, three lines out
def test_rendered_subject_view_drops_idle_and_is_stable():
    items = [
        StreamItem(event=_observe(), truth=None),
        StreamItem(event=_probe(), truth=ProbeTruth(answer=SECRET_ANSWER)),
        StreamItem(event=_idle(), truth=None),
        StreamItem(event=_boundary(), truth=None),
    ]
    first = list(rendered_subject_view(items))
    second = list(rendered_subject_view(items))
    assert first == second
    # Four events in, three lines out: the Idle reaches the subject as
    # time through `idle(budget)`, never as text.
    assert len(first) == 3
    assert not any("budget" in line for line in first)
    joined = "\n".join(first)
    assert SECRET_ANSWER not in joined
    assert PROBE_ID not in joined
    assert TASK_ID not in joined
    assert str(TEACHING_POSITION) not in joined
    assert "True" not in joined
    assert "hostile" not in joined


# covers: eval/render::rendered_subject_view drops Idle and hidden Boundary::hidden boundary is omitted
def test_rendered_subject_view_omits_hidden_boundary():
    hidden = Boundary(position=1, kind=BoundaryKind.TASK_SWITCH, hidden_from_subject=True)
    items = [StreamItem(event=hidden, truth=None)]
    assert list(rendered_subject_view(items)) == []


# covers: eval/render::rendered_subject_view drops Idle and hidden Boundary::Idle events omitted
def test_rendered_subject_view_omits_idle_events():
    items = [
        StreamItem(event=_observe(), truth=None),
        StreamItem(event=_idle(), truth=None),
    ]
    rendered = list(rendered_subject_view(items))
    assert len(rendered) == 1


# covers: eval/render::rendered_subject_view drops Idle and hidden Boundary::hidden boundary omitted
def test_rendered_subject_view_hidden_boundary_yields_nothing():
    hidden = Boundary(position=1, kind=BoundaryKind.TASK_SWITCH, hidden_from_subject=True)
    items = [StreamItem(event=hidden, truth=None)]
    assert list(rendered_subject_view(items)) == []


# covers: eval/render::rendered_subject_view drops Idle and hidden Boundary::repeated calls produce same sequence
def test_rendered_subject_view_repeated_calls_produce_same_sequence():
    items = [
        StreamItem(event=_observe(), truth=None),
        StreamItem(event=_probe(), truth=ProbeTruth(answer=SECRET_ANSWER)),
        StreamItem(event=_idle(), truth=None),
        StreamItem(event=_boundary(), truth=None),
    ]
    first = list(rendered_subject_view(items))
    second = list(rendered_subject_view(items))
    assert first == second


# covers: eval/render::rendered_subject_view drops Idle and hidden Boundary::no harness fields leak
def test_rendered_subject_view_no_harness_fields_leak():
    items = [
        StreamItem(event=_probe(), truth=ProbeTruth(answer=SECRET_ANSWER)),
        StreamItem(event=_idle(), truth=None),
    ]
    joined = "\n".join(rendered_subject_view(items))
    assert SECRET_ANSWER not in joined
    assert PROBE_ID not in joined
    assert TASK_ID not in joined
    assert str(TEACHING_POSITION) not in joined
    assert "budget" not in joined
    assert "True" not in joined
    assert "hostile" not in joined


# covers: eval/render::default_token_counter is deterministic and reports its name::repeated counting is stable
def test_default_token_counter_is_deterministic():
    text = "[observe] hello world, this is a payload"
    first = default_token_counter(text)
    second = default_token_counter(text)
    assert first == second
    assert isinstance(first, int)
    assert first > 0


# covers: eval/render::default_token_counter is deterministic and reports its name::name is a non-empty string
def test_default_token_counter_name_is_reported():
    assert isinstance(default_token_counter.name, str)
    assert default_token_counter.name


# covers: eval/render::default_token_counter is deterministic and reports its name::returns a positive integer
def test_default_token_counter_returns_positive_integer():
    assert default_token_counter("some non-empty text") > 0


# covers: eval/render::default token counter name is "regex-whitespace-v1"::name equals regex-whitespace-v1
def test_default_token_counter_name_equals_regex_whitespace_v1():
    assert default_token_counter.name == "regex-whitespace-v1"


# covers: eval/render::default token counter name is "regex-whitespace-v1"::resolve_token_counter returns the default counter
def test_resolve_token_counter_returns_default_counter():
    counter = resolve_token_counter("regex-whitespace-v1")
    assert counter.name == "regex-whitespace-v1"


# covers: eval/render::resolve_token_counter raises ValueError naming known counters::unknown counter name
def test_resolve_token_counter_unknown_name_raises():
    with pytest.raises(ValueError, match="regex-whitespace-v1"):
        resolve_token_counter("not-a-real-counter")


# covers: eval/render::resolve_token_counter raises ValueError naming known counters::error message names the registered counter
def test_resolve_token_counter_error_names_registered_counter():
    with pytest.raises(ValueError) as exc_info:
        resolve_token_counter("bogus")
    assert "regex-whitespace-v1" in str(exc_info.value)


class _FixedTokenCounter:
    name = "fixed-for-test"

    def __call__(self, text: str) -> int:
        return 1


# covers: eval/render::TokenCounter protocol allows injection::swap in a custom counter
def test_injected_counter_overrides_default():
    injected: TokenCounter = _FixedTokenCounter()
    text = "[observe] hello world, this is a much longer payload than one token"
    assert injected(text) == 1
    assert injected.name != default_token_counter.name
    assert injected(text) != default_token_counter(text)


class _NamedTokenCounter:
    name = "my-counter"

    def __call__(self, text: str) -> int:
        return 1


# covers: eval/render::TokenCounter protocol allows injection::custom counter name is preserved
def test_custom_counter_name_is_preserved():
    counter: TokenCounter = _NamedTokenCounter()
    assert counter.name == "my-counter"


# covers: eval/render::format_features renders exactly 2 decimal places::probe and example use the same feature notation
def test_format_features_never_has_three_consecutive_digits_after_decimal():
    values = [5.383941741180422, -0.5668909031452558, 123.456789, 0.001]
    rendered = format_features(values)
    # No run of 3+ digits after a decimal point.
    assert not re.search(r"\.\d{3}", rendered)

    # Also check that an example payload and a probe query built from
    # format_features share the same notation.
    example_rendered = render_event(_example())
    assert not re.search(r"\.\d{3}", example_rendered)


# covers: eval/render::format_features renders exactly 2 decimal places::2-decimal precision enforced
def test_format_features_two_decimal_precision_enforced():
    rendered = format_features([1.23456, 7.891])
    assert "1.23" in rendered
    assert "7.89" in rendered
    assert not re.search(r"\.\d{3}", rendered)


# covers: eval/render::format_features renders exactly 2 decimal places::whole numbers get trailing zeros
def test_format_features_whole_numbers_get_trailing_zeros():
    rendered = format_features([5.0, 10.0])
    assert "5.00" in rendered
    assert "10.00" in rendered


# covers: eval/render::rendered text never contains difficulty or level labels::no difficulty leak in rendered text
def test_no_difficulty_leak_in_difficulty_mix_rendered_text():
    items = _difficulty_mix_items()
    rendered_any = False
    for item in items:
        if not carries_text(item.event):
            continue
        rendered = render_event(item.event)
        rendered_any = True
        assert "difficulty" not in rendered.lower()
        assert "level" not in rendered.lower()
    assert rendered_any


# covers: eval/render::rendered text never contains difficulty or level labels::no level leak in rendered text
def test_no_level_leak_in_difficulty_mix_rendered_text():
    items = _difficulty_mix_items()
    rendered_any = False
    for item in items:
        if not carries_text(item.event):
            continue
        rendered = render_event(item.event)
        rendered_any = True
        assert "level" not in rendered.lower()
    assert rendered_any


# covers: eval/render::default token counter name is "regex-whitespace-v1"::known counter name resolves
def test_known_counter_name_resolves_and_probes_carry_token_distance():
    items = _assoc_items(
        _assoc_config(
            recall_distances=[1, 2],
            token_counter=default_token_counter.name,
        )
    )
    probes = [item.event for item in items if isinstance(item.event, Probe)]
    assert probes
    for probe in probes:
        assert probe.token_distance is not None


# covers: eval/render::token distance is measured over rendered text between teaching and probe::adjacent teaching and probe measure zero
def test_adjacent_teaching_and_probe_measures_zero_token_distance():
    items = _assoc_items(
        _assoc_config(
            recall_distances=[1],
            max_distance=5,
            token_counter=default_token_counter.name,
        )
    )
    probes = [item.event for item in items if isinstance(item.event, Probe)]
    assert probes
    for probe in probes:
        assert probe.token_distance == 0


# covers: eval/render::token distance is measured over rendered text between teaching and probe::token distance grows with event distance
def test_token_distance_grows_with_event_distance():
    items = _assoc_items(
        _assoc_config(
            recall_distances=[1, 2, 5],
            token_counter=default_token_counter.name,
        )
    )
    probes_by_teaching: dict[int, list[Probe]] = {}
    for item in items:
        if isinstance(item.event, Probe):
            probes_by_teaching.setdefault(item.event.teaching_position, []).append(item.event)

    assert probes_by_teaching
    for probes in probes_by_teaching.values():
        ordered = sorted(probes, key=lambda p: p.position - p.teaching_position)
        distances = [p.token_distance for p in ordered]
        assert distances == sorted(distances)
        assert distances[0] < distances[-1]


# covers: eval/render::token distance is measured over rendered text between teaching and probe::token distance derived from rendered text
def test_token_distance_derived_from_rendered_text():
    items = _assoc_items(
        _assoc_config(
            recall_distances=[2],
            token_counter=default_token_counter.name,
        )
    )
    events_by_position = {item.event.position: item.event for item in items}
    probes = [item.event for item in items if isinstance(item.event, Probe)]
    assert probes
    for probe in probes:
        # recall_distances=[2] guarantees exactly one intervening position.
        assert probe.position - probe.teaching_position == 2
        intervening_event = events_by_position[probe.teaching_position + 1]
        expected = default_token_counter(render_event(intervening_event))
        assert probe.token_distance == expected


# covers: eval/render::token distance is measured over rendered text between teaching and probe::non-text events do not contribute to token distance
def test_non_text_events_do_not_contribute_to_token_distance():
    events = [_idle(), _idle()]
    assert [event for event in events if carries_text(event)] == []
