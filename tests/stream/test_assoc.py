from pathlib import Path

import pytest

from eval.stream import vocab
from eval.stream.config import StreamConfig
from eval.stream.corpus import load_corpus, resolve
from eval.stream.events import Observe, Probe
from eval.stream.generator import StreamGenerator
from eval.stream.generators.assoc import AssocGenerator
from eval.stream.hashing import stream_hash
from eval.stream.render import default_token_counter

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _corpus_dir(monkeypatch):
    monkeypatch.setenv("PENSIVE_CORPUS_DIR", str(FIXTURE_DIR))


def _config(**overrides):
    params = {
        "num_pairs": 3,
        "recall_distances": [1, 2, 5],
        "filler_density": 1.0,
        "max_distance": 10,
        "corpus": "corpus_fixture",
    }
    params.update(overrides)
    return StreamConfig(generator="assoc", params=params)


def _items(config=None, seed=0):
    generator = AssocGenerator()
    return list(generator.generate(config=config or _config(), seed=seed))


def _observes_by_position(items):
    return {
        item.event.position: item.event
        for item in items
        if isinstance(item.event, Observe)
    }


# covers: eval/generators/assoc::Probes at exactly the requested distances::probe truth matches the taught value
def test_probe_truth_matches_the_taught_value():
    items = _items()
    observes = _observes_by_position(items)
    for item in items:
        if not isinstance(item.event, Probe):
            continue
        teach = observes[item.event.teaching_position]
        assert item.truth.answer == teach.payload["value"]


def test_probe_teaching_position_points_at_an_earlier_observe_that_taught_it():
    items = _items()
    observes = _observes_by_position(items)
    for item in items:
        if not isinstance(item.event, Probe):
            continue
        teach = observes[item.event.teaching_position]
        assert isinstance(teach, Observe)
        assert teach.position < item.event.position
        assert teach.payload["key"] is not None


# covers: eval/generators/assoc::Probes at exactly the requested distances::probe distance matches configuration
def test_gap_between_teaching_and_probe_matches_requested_distances_exactly():
    items = _items()
    gaps = set()
    for item in items:
        if not isinstance(item.event, Probe):
            continue
        gap = item.event.position - item.event.teaching_position
        gaps.add(gap)
    assert gaps == {1, 2, 5}


# covers: eval/generators/assoc::Each pair is taught exactly once::teaching count matches pair count
def test_each_pair_is_taught_exactly_once():
    items = _items()
    taught_keys = [
        item.event.payload["key"]
        for item in items
        if isinstance(item.event, Observe)
        and any(
            other.event.teaching_position == item.event.position
            for other in items
            if isinstance(other.event, Probe)
        )
    ]
    assert len(taught_keys) == len(set(taught_keys)) == 3


# covers: eval/generators/assoc::Positions are contiguous from zero::no gaps in positions
def test_positions_are_consecutive_from_zero_with_no_gaps_or_repeats():
    items = _items()
    positions = [item.event.position for item in items]
    assert positions == list(range(len(positions)))


def test_no_probe_exceeds_the_configured_max_distance():
    items = _items(_config(max_distance=3, recall_distances=[1, 2, 3]))
    for item in items:
        if not isinstance(item.event, Probe):
            continue
        gap = item.event.position - item.event.teaching_position
        assert gap <= 3


# covers: eval/generators/assoc::Deterministic in (config, seed) and diverges on different seeds::replay determinism
def test_same_config_and_seed_give_identical_sequence():
    first = _items()
    second = _items()
    assert [item.event for item in first] == [item.event for item in second]
    assert [item.truth for item in first] == [item.truth for item in second]


# covers: eval/generators/assoc::Deterministic in (config, seed) and diverges on different seeds::replay determinism
def test_different_seed_gives_different_sequence():
    first = _items(seed=0)
    second = _items(seed=1)
    assert [item.event for item in first] != [item.event for item in second]


def test_different_filler_density_gives_different_streams():
    low = _items(_config(recall_distances=[1, 2, 3], filler_density=0.5))
    high = _items(_config(recall_distances=[1, 2, 3], filler_density=3.0))
    low_events = [item.event for item in low]
    high_events = [item.event for item in high]
    assert low_events != high_events
    assert len(low_events) != len(high_events)


# covers: eval/generators/assoc::Unschedulable configurations raise ValueError::impossible schedule rejected
def test_unschedulable_config_raises_rather_than_clamping():
    with pytest.raises(ValueError):
        _items(_config(num_pairs=10, recall_distances=[1], filler_density=0.0, max_distance=10))


# covers: eval/generators/assoc::Generator identity and version::version is pinned
def test_generator_satisfies_stream_generator_protocol():
    generator = AssocGenerator()
    assert isinstance(generator, StreamGenerator)
    assert generator.name == "assoc"
    assert generator.version == "6"


def test_every_probe_query_names_the_taught_key():
    items = _items()
    observes = _observes_by_position(items)
    for item in items:
        if not isinstance(item.event, Probe):
            continue
        assert item.event.query
        teach = observes[item.event.teaching_position]
        assert item.event.query == teach.payload["key"]


# covers: eval/generators/assoc::Interleaving puts other pairs' events between a teaching and its probe::interleaving occurs
def test_interleaving_puts_another_pairs_teaching_between_a_teaching_and_its_probe():
    items = _items(_config(num_pairs=5, recall_distances=[2], filler_density=1.0, max_distance=10))
    probes = [item for item in items if isinstance(item.event, Probe)]
    observes = [item for item in items if isinstance(item.event, Observe)]

    found_interleaved = False
    for probe_item in probes:
        teach_position = probe_item.event.teaching_position
        probe_position = probe_item.event.position
        for observe_item in observes:
            if teach_position < observe_item.event.position < probe_position:
                if observe_item.event.position != teach_position:
                    is_another_teaching = any(
                        other.event.teaching_position == observe_item.event.position
                        for other in probes
                    )
                    if is_another_teaching:
                        found_interleaved = True
    assert found_interleaved


# covers: eval/generators/assoc::No payload or query text carries a role prefix::no role prefix leak
def test_no_payload_or_query_text_carries_a_role_prefix():
    items = _items(_config(num_pairs=5, recall_distances=[1, 2, 3], filler_density=1.0, max_distance=10))
    for item in items:
        if isinstance(item.event, Observe) and "key" in item.event.payload:
            texts = [item.event.payload["key"], item.event.payload["value"]]
        elif isinstance(item.event, Observe):
            texts = [item.event.payload["text"]]
        elif isinstance(item.event, Probe):
            texts = [item.event.query]
        else:
            texts = []
        for text in texts:
            assert not text.startswith("key-")
            assert not text.startswith("value-")
            assert not text.startswith("filler-")


# covers: eval/generators/assoc::Keys are distinct vocabulary words::unique keys from vocabulary
def test_every_key_and_value_is_a_vocabulary_word():
    items = _items(_config(num_pairs=5, recall_distances=[1, 2, 3], filler_density=1.0, max_distance=10))
    for item in items:
        if isinstance(item.event, Observe) and "key" in item.event.payload:
            assert item.event.payload["key"] in vocab.VOCAB
            assert item.event.payload["value"] in vocab.VOCAB
        elif isinstance(item.event, Probe):
            assert item.event.query in vocab.VOCAB
            assert item.truth.answer in vocab.VOCAB


# covers: eval/generators/assoc::Keys are distinct vocabulary words::unique keys from vocabulary
def test_keys_are_pairwise_distinct_across_the_whole_stream():
    items = _items(_config(num_pairs=5, recall_distances=[1, 2, 3], filler_density=1.0, max_distance=10))
    keys = [
        item.event.payload["key"]
        for item in items
        if isinstance(item.event, Observe) and "key" in item.event.payload
    ]
    assert len(keys) == 5
    assert len(keys) == len(set(keys))


def test_each_probe_query_matches_exactly_one_taught_key():
    items = _items(_config(num_pairs=5, recall_distances=[1, 2, 3], filler_density=1.0, max_distance=10))
    taught_keys = [
        item.event.payload["key"]
        for item in items
        if isinstance(item.event, Observe)
        and any(
            other.event.teaching_position == item.event.position
            for other in items
            if isinstance(other.event, Probe)
        )
    ]
    for item in items:
        if not isinstance(item.event, Probe):
            continue
        matches = [key for key in taught_keys if key == item.event.query]
        assert len(matches) == 1


# covers: eval/generators/assoc::Filler occupies non-teaching, non-probe positions with continuous corpus prose::filler continues one document
def test_filler_spans_continue_one_document_across_the_stream():
    """Consecutive filler must read as one document, not as unrelated fragments.

    Prose filler exists so a taught fact has to survive continuous text.
    Drawing each span from an independent random start satisfied the
    letter of that and not the sense: the text jumped topic every span.
    So the whole run of filler, joined in position order, has to appear
    verbatim in the corpus it came from.
    """
    corpus = load_corpus(resolve("corpus_fixture"))
    items = _items(_config(num_pairs=3, recall_distances=[1, 2], filler_density=3.0, max_distance=10))
    filler_texts = [
        item.event.payload["text"]
        for item in items
        if isinstance(item.event, Observe) and "text" in item.event.payload
    ]
    assert len(filler_texts) > 3
    assert " ".join(filler_texts) in " ".join(corpus.words)


def test_filler_payloads_are_prose_and_differ_from_each_other():
    items = _items(_config(num_pairs=5, recall_distances=[1, 2, 3], filler_density=1.0, max_distance=10))
    filler_texts = [
        item.event.payload["text"]
        for item in items
        if isinstance(item.event, Observe) and "text" in item.event.payload
    ]
    assert len(filler_texts) > 1
    for text in filler_texts:
        assert len(text.split()) > 1
    assert len(set(filler_texts)) > 1


def test_taught_observe_payload_holds_only_key_and_value():
    items = _items(_config(num_pairs=5, recall_distances=[1, 2, 3], filler_density=1.0, max_distance=10))
    taught_positions = {
        item.event.teaching_position for item in items if isinstance(item.event, Probe)
    }
    for item in items:
        if not isinstance(item.event, Observe):
            continue
        if item.event.position in taught_positions:
            assert set(item.event.payload.keys()) == {"key", "value"}
            assert item.event.payload["key"] in vocab.VOCAB
            assert item.event.payload["value"] in vocab.VOCAB
        else:
            assert set(item.event.payload.keys()) == {"text"}


# covers: eval/generators/assoc::token_distance absent without counter, computed with counter::no counter configured
def test_token_distance_is_none_when_no_counter_is_configured():
    items = _items()
    probes = [item.event for item in items if isinstance(item.event, Probe)]
    assert probes
    for probe in probes:
        assert probe.token_distance is None


# covers: eval/generators/assoc::token_distance absent without counter, computed with counter::counter configured, adjacent probe
def test_token_distance_is_zero_for_no_intervening_events():
    items = _items(
        _config(
            recall_distances=[1],
            max_distance=5,
            token_counter=default_token_counter.name,
        )
    )
    probes = [item.event for item in items if isinstance(item.event, Probe)]
    assert probes
    for probe in probes:
        assert probe.token_distance == 0


def test_token_distance_rises_with_the_event_distance():
    items = _items(
        _config(
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
        ordered = sorted(probes, key=lambda probe: probe.position - probe.teaching_position)
        token_distances = [probe.token_distance for probe in ordered]
        assert token_distances == sorted(token_distances)
        assert token_distances[0] < token_distances[-1]


def test_config_naming_a_token_counter_hashes_without_raising():
    config = _config(recall_distances=[1], token_counter=default_token_counter.name)
    digest = stream_hash(
        config,
        seed=0,
        generator_version=AssocGenerator.version,
        render_version="1",
        corpus_id="corpus-id",
    )
    assert isinstance(digest, str)
    assert digest


def test_configs_naming_different_counters_are_both_hashable():
    first = _config(recall_distances=[1], token_counter="regex-whitespace-v1")
    second = _config(recall_distances=[1], token_counter="some-other-counter")
    first_digest = stream_hash(
        first, seed=0, generator_version=AssocGenerator.version,
        render_version="1", corpus_id="corpus-id",
    )
    second_digest = stream_hash(
        second, seed=0, generator_version=AssocGenerator.version,
        render_version="1", corpus_id="corpus-id",
    )
    assert isinstance(first_digest, str) and first_digest
    assert isinstance(second_digest, str) and second_digest
    assert first_digest != second_digest


# covers: eval/generators/assoc::Unknown token_counter name raises ValueError::bad counter name
def test_unknown_token_counter_name_raises_naming_the_known_counters():
    with pytest.raises(ValueError, match="regex-whitespace-v1"):
        _items(_config(recall_distances=[1], token_counter="not-a-real-counter"))


# covers: eval/generators/assoc::Config validation::distance below 1
def test_recall_distance_zero_raises():
    with pytest.raises(ValueError):
        _items(_config(recall_distances=[0], max_distance=10))


# covers: eval/generators/assoc::Config validation::distance below 1
def test_distance_exceeding_max_raises():
    with pytest.raises(ValueError):
        _items(_config(recall_distances=[20], max_distance=10))


# covers: eval/generators/assoc::Config validation::distance below 1
def test_negative_filler_density_raises():
    with pytest.raises(ValueError):
        _items(_config(filler_density=-1.0))


# covers: eval/generators/assoc::Total stream length is a fixed function of config::pinned stream length
def test_pinned_stream_length():
    items = _items(
        _config(
            num_pairs=10,
            recall_distances=[1, 10, 100],
            filler_density=5.0,
            max_distance=100,
        )
    )
    assert len(items) == 240


# covers: eval/generators/assoc::chance_rate equals 1/len(VOCAB)::chance rate
def test_chance_rate_equals_one_over_vocab_size():
    generator = AssocGenerator()
    rate = generator.chance_rate(_config())
    assert rate == 1 / len(vocab.VOCAB)


def test_a_120100_position_stream_builds_without_raising():
    items = _items(
        _config(
            num_pairs=50,
            recall_distances=[100000],
            filler_density=1200.0,
            max_distance=100000,
        )
    )
    assert len(items) == 120100
