import re

import pytest

from eval.stream.config import StreamConfig
from eval.stream.events import Idle, Probe
from eval.stream.generator import StreamGenerator
from eval.stream.generators.difficulty_mix import CHAIN_MODULUS, DifficultyMixGenerator
from eval.stream.render import carries_text, render_event

_STEP_PATTERN = re.compile(r"^(Add|Subtract) (\d+)\.$")
_STEP_COUNT_PATTERN = re.compile(r"(?:Add|Subtract) \d+\.")


def _config(**overrides):
    params = {
        "num_items": 12,
        "difficulty_levels": [1, 2, 3, 5],
        "probe_rate": 0.5,
    }
    params.update(overrides)
    return StreamConfig(generator="difficulty-mix", params=params)


def _items(config=None, seed=0):
    generator = DifficultyMixGenerator()
    return list(generator.generate(config=config or _config(), seed=seed))


def _probes(items):
    return [item for item in items if isinstance(item.event, Probe)]


def _evaluate_chain_independently(query: str) -> int:
    """Re-derive the chain answer from the rendered query text alone.

    This parses the same words the generator renders, then runs its own
    add/subtract loop against `CHAIN_MODULUS`, so a pass proves the truth
    channel matches an independent evaluation of the chain rather than
    merely agreeing with the generator's own arithmetic helper.
    """
    suffix = f" What is the result modulo {CHAIN_MODULUS}?"
    assert query.endswith(suffix), f"query has an unexpected tail: {query!r}"
    body = query[: -len(suffix)]

    start_match = re.match(r"^Start at (\d+)\.", body)
    assert start_match, f"query has no parseable start: {query!r}"
    total = int(start_match.group(1))

    rest = body[start_match.end():].strip()
    step_texts = [segment.strip() for segment in rest.split(".") if segment.strip()]
    for step_text in step_texts:
        match = _STEP_PATTERN.match(step_text + ".")
        assert match, f"unparseable step: {step_text!r} in query {query!r}"
        verb, operand_text = match.groups()
        operand = int(operand_text)
        total = (
            (total + operand) % CHAIN_MODULUS
            if verb == "Add"
            else (total - operand) % CHAIN_MODULUS
        )
    return total


# covers: eval/generators/difficulty-mix::Generator identity and version::name and version present
def test_generator_satisfies_stream_generator_protocol():
    generator = DifficultyMixGenerator()
    assert isinstance(generator, StreamGenerator)
    assert generator.name == "difficulty-mix"
    assert generator.version


# covers: eval/generators/difficulty-mix::Every configured difficulty level appears at least once::all levels represented
def test_every_configured_difficulty_level_appears():
    items = _items(_config(num_items=20, difficulty_levels=[1, 2, 3, 5, 8]))
    probes = _probes(items)
    seen_difficulties = {item.truth.difficulty for item in probes}
    assert seen_difficulties == {1, 2, 3, 5, 8}


# covers: eval/generators/difficulty-mix::Truth answer matches independently evaluated chain::chain replay agrees
def test_truth_answer_matches_the_chain_evaluated_independently():
    items = _items(_config(num_items=20, difficulty_levels=[1, 2, 3, 5, 8]))
    probes = _probes(items)
    assert probes
    for item in probes:
        independent_answer = _evaluate_chain_independently(item.event.query)
        assert item.truth.answer == independent_answer


# covers: eval/generators/difficulty-mix::Difficulty equals the number of arithmetic steps::difficulty matches step count
def test_difficulty_matches_the_number_of_steps_in_the_rendered_chain():
    items = _items(_config(num_items=20, difficulty_levels=[1, 2, 3, 5, 8]))
    probes = _probes(items)
    assert probes
    for item in probes:
        step_count = len(_STEP_COUNT_PATTERN.findall(item.event.query))
        assert step_count == item.truth.difficulty


# covers: eval/generators/difficulty-mix::Answer space is bounded by CHAIN_MODULUS::answer is bounded
def test_answer_space_is_bounded_and_explicit():
    assert isinstance(CHAIN_MODULUS, int)
    assert CHAIN_MODULUS > 0
    items = _items(_config(num_items=20, difficulty_levels=[1, 2, 3, 5, 8]))
    for item in _probes(items):
        assert 0 <= item.truth.answer < CHAIN_MODULUS


# covers: eval/generators/difficulty-mix::Rendered query never leaks difficulty or level labels::no difficulty leak
def test_no_rendered_text_carries_the_difficulty_label():
    items = _items(_config(num_items=20, difficulty_levels=[1, 2, 3, 5, 8]))
    for item in items:
        if not carries_text(item.event):
            continue
        rendered = render_event(item.event)
        assert "difficulty" not in rendered.lower()
        assert "level" not in rendered.lower()


# covers: eval/generators/difficulty-mix::Rendered query follows a parseable grammar::query grammar
def test_rendered_query_follows_the_parseable_grammar():
    items = _items(_config(num_items=20, difficulty_levels=[1, 2, 3, 5, 8]))
    probes = _probes(items)
    assert probes
    grammar = re.compile(
        rf"^Start at (\d+)\."
        rf"(?: (Add|Subtract) (\d+)\.)+?"
        rf" What is the result modulo {CHAIN_MODULUS}\?$"
    )
    for item in probes:
        query = item.event.query
        # Full query matches the overall grammar.
        assert grammar.match(query), f"query does not match grammar: {query!r}"
        # Verify each step individually against the step pattern.
        suffix = f" What is the result modulo {CHAIN_MODULUS}?"
        body = query[: -len(suffix)]
        start_match = re.match(r"^Start at (\d+)\.", body)
        assert start_match, f"no start segment: {query!r}"
        rest = body[start_match.end():].strip()
        step_texts = [s.strip() for s in rest.split(".") if s.strip()]
        for step_text in step_texts:
            assert _STEP_PATTERN.match(step_text + "."), (
                f"step does not match pattern: {step_text!r}"
            )


def test_rendered_query_never_leaks_the_step_count_as_a_label():
    items = _items(_config(num_items=20, difficulty_levels=[1, 2, 3, 5, 8]))
    for item in _probes(items):
        assert "level" not in item.event.query.lower()
        assert "difficulty" not in item.event.query.lower()


# covers: eval/generators/difficulty-mix::Non-probe positions are Idle events with no truth::idle filler
def test_non_probe_positions_are_idle_events():
    items = _items()
    for item in items:
        assert isinstance(item.event, (Idle, Probe))
        if isinstance(item.event, Idle):
            assert item.truth is None


# covers: eval/generators/difficulty-mix::Positions are contiguous from zero::contiguous positions
def test_positions_are_consecutive_from_zero_with_no_gaps_or_repeats():
    items = _items()
    positions = [item.event.position for item in items]
    assert positions == list(range(len(positions)))


# covers: eval/generators/difficulty-mix::Probe count equals num_items exactly::exact probe count
def test_probe_count_matches_num_items():
    items = _items(_config(num_items=9, difficulty_levels=[1, 2, 3], probe_rate=0.3))
    assert len(_probes(items)) == 9


# covers: eval/generators/difficulty-mix::chance_rate equals 1/CHAIN_MODULUS::chance rate
def test_chance_rate_equals_inverse_of_chain_modulus():
    generator = DifficultyMixGenerator()
    assert generator.chance_rate(_config()) == 1.0 / CHAIN_MODULUS
    assert generator.chance_rate(_config()) == 0.001


# covers: eval/generators/difficulty-mix::Deterministic in (config, seed) and diverges on different seeds::replay determinism
def test_same_config_and_seed_give_identical_sequence():
    first = _items()
    second = _items()
    assert [item.event for item in first] == [item.event for item in second]
    assert [item.truth for item in first] == [item.truth for item in second]


def test_different_seed_gives_a_different_interleaving_or_chains():
    first = _items(seed=0)
    second = _items(seed=1)
    assert [item.event for item in first] != [item.event for item in second]


def test_num_items_below_difficulty_level_count_raises():
    with pytest.raises(ValueError):
        _items(_config(num_items=2, difficulty_levels=[1, 2, 3, 5]))


# covers: eval/generators/difficulty-mix::Validates num_items, difficulty_levels, and probe_rate::empty difficulty_levels rejected
def test_empty_difficulty_levels_raises():
    with pytest.raises(ValueError):
        _items(_config(difficulty_levels=[]))


# covers: eval/generators/difficulty-mix::Validates num_items, difficulty_levels, and probe_rate::empty difficulty_levels rejected
def test_num_items_zero_raises():
    with pytest.raises(ValueError):
        _items(_config(num_items=0))


# covers: eval/generators/difficulty-mix::Validates num_items, difficulty_levels, and probe_rate::empty difficulty_levels rejected
def test_difficulty_level_below_one_raises():
    with pytest.raises(ValueError):
        _items(_config(difficulty_levels=[0, 1, 2]))


# covers: eval/generators/difficulty-mix::Validates num_items, difficulty_levels, and probe_rate::probe_rate out of range
def test_probe_rate_out_of_range_raises():
    with pytest.raises(ValueError):
        _items(_config(probe_rate=0))
    with pytest.raises(ValueError):
        _items(_config(probe_rate=1.5))
