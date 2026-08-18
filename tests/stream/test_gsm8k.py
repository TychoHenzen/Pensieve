from eval.stream.config import StreamConfig
from eval.stream.events import Observe, Probe
from eval.stream.generator import StreamGenerator
from eval.stream.generators.gsm8k import GSM8KGenerator
from eval.stream.render import render_event
from eval.stream.truth import ProbeTruth, subject_view


def _config(**overrides):
    params = {"split": "test", "problem_count": 5}
    params.update(overrides)
    return StreamConfig(generator="gsm8k", params=params)


def _items(config=None, seed=0):
    generator = GSM8KGenerator()
    return list(generator.generate(config=config or _config(), seed=seed))


def _probes(items):
    return [item for item in items if isinstance(item.event, Probe)]


# covers: eval/generators/gsm8k::Generator identity::satisfies StreamGenerator protocol
def test_generator_satisfies_stream_generator_protocol():
    generator = GSM8KGenerator()
    assert isinstance(generator, StreamGenerator)
    assert generator.name == "gsm8k"
    assert generator.version


# covers: eval/generators/gsm8k::chance_rate reports 0.0::chance rate is zero
def test_chance_rate_is_zero():
    generator = GSM8KGenerator()
    assert generator.chance_rate(_config()) == 0.0


# covers: eval/generators/gsm8k::chance_rate reports 0.0::chance rate is zero regardless of config
def test_chance_rate_is_zero_regardless_of_problem_count():
    generator = GSM8KGenerator()
    assert generator.chance_rate(_config(problem_count=1)) == 0.0
    assert generator.chance_rate(_config(problem_count=None)) == 0.0


# covers: eval/generators/gsm8k::deterministic replay::same seed and config give identical sequence
def test_same_seed_and_config_give_identical_sequence():
    config = _config()
    first = _items(config, seed=7)
    second = _items(config, seed=7)
    assert [item.event for item in first] == [item.event for item in second]
    assert [item.truth for item in first] == [item.truth for item in second]


# covers: eval/generators/gsm8k::deterministic replay::different seeds diverge
def test_different_seeds_produce_different_streams():
    first = _items(seed=0)
    second = _items(seed=1)
    assert [item.event for item in first] != [item.event for item in second]


# covers: eval/generators/gsm8k::truth isolation::subject_view carries no ProbeTruth
def test_subject_view_carries_no_truth():
    items = _items()
    subject_events = list(subject_view(items))
    for event in subject_events:
        assert not isinstance(event, ProbeTruth)
        assert not hasattr(event, "truth")


# covers: eval/generators/gsm8k::truth isolation::rendered probe text contains no answer
def test_rendered_probe_text_contains_no_answer():
    items = _items()
    for item in items:
        if not isinstance(item.event, Probe):
            continue
        rendered = render_event(item.event)
        assert item.truth is not None
        answer = str(item.truth.answer)
        assert answer not in rendered


# covers: eval/generators/gsm8k::event shape::Observe payloads carry text
def test_observe_events_have_text_payload():
    items = _items()
    observes = [item.event for item in items if isinstance(item.event, Observe)]
    assert observes
    for obs in observes:
        assert isinstance(obs.payload, dict)
        assert isinstance(obs.payload["text"], str)
        assert obs.payload["text"]


# covers: eval/generators/gsm8k::event shape::Probe events carry a query string
def test_probe_events_have_query_string():
    items = _items()
    probes = _probes(items)
    assert probes
    for item in probes:
        assert isinstance(item.event, Probe)
        assert isinstance(item.event.query, str)
        assert item.event.query


# covers: eval/generators/gsm8k::event shape::each problem yields one Observe followed by one Probe
def test_each_problem_yields_one_observe_then_one_probe():
    items = _items(_config(problem_count=5))
    assert len(items) == 10
    for index in range(0, len(items), 2):
        assert isinstance(items[index].event, Observe)
        assert isinstance(items[index + 1].event, Probe)


# covers: eval/generators/gsm8k::event shape::probe truth is a ProbeTruth with a non-empty answer
def test_probe_truth_has_non_empty_answer():
    items = _items()
    for item in _probes(items):
        assert isinstance(item.truth, ProbeTruth)
        assert item.truth.answer
