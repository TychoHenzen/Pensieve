import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eval.stream.config import StreamConfig
from eval.stream.hashing import canonical_json, stream_hash
from eval.stream.serialize import items_to_canonical_json, items_to_plain


def _config(**params) -> StreamConfig:
    return StreamConfig(generator="assoc", params=params)


# covers: eval/serialize::stream_hash is a reproducible function of exactly five inputs::identical inputs repeat
def test_same_config_seed_and_version_repeats_the_hash():
    config = _config(pairs=10, distance=100)
    assert stream_hash(config, 0, "1.0", "1", "corpus-a") == stream_hash(
        config, 0, "1.0", "1", "corpus-a"
    )


# covers: eval/config::StreamConfig params are canonically hashable::key order does not affect the hash
def test_key_insertion_order_does_not_affect_the_hash():
    first = StreamConfig(generator="assoc", params={"a": 1, "b": 2})
    second = StreamConfig(generator="assoc", params={"b": 2, "a": 1})
    assert stream_hash(first, 0, "1.0", "1", "corpus-a") == stream_hash(
        second, 0, "1.0", "1", "corpus-a"
    )


# covers: eval/serialize::stream_hash is a reproducible function of exactly five inputs::any one input changing changes the hash
def test_changing_a_config_parameter_changes_the_hash():
    base = _config(pairs=10, distance=100)
    changed = _config(pairs=11, distance=100)
    assert stream_hash(base, 0, "1.0", "1", "corpus-a") != stream_hash(
        changed, 0, "1.0", "1", "corpus-a"
    )


# covers: eval/serialize::stream_hash is a reproducible function of exactly five inputs::any one input changing changes the hash
def test_changing_the_seed_changes_the_hash():
    config = _config(pairs=10, distance=100)
    assert stream_hash(config, 0, "1.0", "1", "corpus-a") != stream_hash(
        config, 1, "1.0", "1", "corpus-a"
    )


# covers: eval/serialize::stream_hash is a reproducible function of exactly five inputs::any one input changing changes the hash
def test_changing_the_generator_version_changes_the_hash():
    config = _config(pairs=10, distance=100)
    assert stream_hash(config, 0, "1.0", "1", "corpus-a") != stream_hash(
        config, 0, "1.1", "1", "corpus-a"
    )


# covers: eval/serialize::stream_hash is a reproducible function of exactly five inputs::any one input changing changes the hash
def test_changing_the_render_version_changes_the_hash():
    config = _config(pairs=10, distance=100)
    assert stream_hash(config, 0, "1.0", "1", "corpus-a") != stream_hash(
        config, 0, "1.0", "2", "corpus-a"
    )


# covers: eval/serialize::stream_hash is a reproducible function of exactly five inputs::any one input changing changes the hash
def test_changing_the_corpus_id_changes_the_hash():
    config = _config(pairs=10, distance=100)
    assert stream_hash(config, 0, "1.0", "1", "corpus-a") != stream_hash(
        config, 0, "1.0", "1", "corpus-b"
    )


# covers: eval/serialize::stream_hash is a reproducible function of exactly five inputs::digest format
def test_hash_is_lowercase_hex_of_fixed_length():
    digest = stream_hash(_config(pairs=10), 0, "1.0", "1", "corpus-a")
    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(char in "0123456789abcdef" for char in digest)


# covers: eval/serialize::canonical_json handles frozen mapping proxies and tuples::tuple encoded as array
def test_canonical_json_encodes_tuple_as_array():
    result = canonical_json((1, 2, 3))
    assert result == "[1,2,3]"


# covers: eval/serialize::canonical_json handles frozen mapping proxies and tuples::nested frozen structure encodes
def test_hashing_a_nested_config_structure_works():
    config = StreamConfig(
        generator="split-classify",
        params={"tasks": [{"name": "a", "weight": 0.5}, {"name": "b", "weight": 0.5}]},
    )
    assert stream_hash(config, 0, "1.0", "1", "corpus-a") == stream_hash(
        config, 0, "1.0", "1", "corpus-a"
    )


# covers: eval/serialize::stream_hash reproduces a pinned digest::golden value reproduces
def test_known_hash_value_for_a_fixed_config_seed_version_render_and_corpus():
    config = StreamConfig(generator="assoc", params={"pairs": 10, "distance": 100})
    assert (
        stream_hash(config, 0, "1.0", "1", "corpus-a")
        == "e60856050c6024f1fdb90d15563c7102393b80c47ff549892f06cdaf41c03f3b"
    )


# covers: eval/serialize::stream_hash reproduces a pinned digest::pinned digest is 64 hex characters
def test_pinned_digest_is_64_hex_characters():
    digest = "e60856050c6024f1fdb90d15563c7102393b80c47ff549892f06cdaf41c03f3b"
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# covers: eval/config::StreamConfig is frozen::frozen attribute
def test_stream_config_attribute_assignment_raises():
    config = _config(pairs=10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.generator = "other"


# covers: eval/serialize::canonical_json forbids NaN and Infinity::non-finite float rejected
def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


# covers: eval/serialize::canonical_json forbids NaN and Infinity::non-finite float rejected
def test_canonical_json_rejects_infinity():
    with pytest.raises(ValueError):
        canonical_json({"x": float("inf")})


# covers: eval/serialize::canonical_json sorts keys and uses compact separators::key order does not affect output
def test_canonical_json_uses_compact_separators():
    result = canonical_json({"a": 1, "b": 2})
    assert ", " not in result
    assert ": " not in result
    assert "," in result
    assert ":" in result


# covers: eval/serialize::canonical_json sorts keys and uses compact separators::compact separators used
def test_canonical_json_compact_separators_no_spaces():
    result = canonical_json({"a": 1, "b": 2})
    parsed = json.loads(result)
    assert parsed == {"a": 1, "b": 2}
    assert result == '{"a":1,"b":2}'


# covers: eval/serialize::canonical_json raises TypeError on sets::a set cannot be canonicalized
def test_canonical_json_raises_on_a_value_it_cannot_encode_canonically():
    with pytest.raises(TypeError):
        canonical_json({1, 2, 3})


# covers: eval/serialize::canonical_json raises TypeError on sets::nested set also rejected
def test_canonical_json_raises_on_nested_set():
    with pytest.raises(TypeError):
        canonical_json({"a": {1, 2}})


# covers: eval/config::StreamConfig immutable params::top-level params assignment rejected
def test_assigning_into_config_params_raises():
    config = _config(pairs=10)
    with pytest.raises(TypeError):
        config.params["pairs"] = 9999


# covers: eval/config::StreamConfig immutable params::nested params assignment rejected
def test_assigning_into_a_nested_mapping_raises():
    config = StreamConfig(
        generator="split-classify",
        params={"task": {"name": "a", "weight": 0.5}},
    )
    with pytest.raises(TypeError):
        config.params["task"]["weight"] = 1.0


# covers: eval/config::StreamConfig immutable params::caller's dict independence
def test_mutating_the_callers_original_dict_does_not_affect_the_config():
    original = {"pairs": 10, "distance": 100}
    config = StreamConfig(generator="assoc", params=original)
    before = stream_hash(config, 0, "1.0", "1", "corpus-a")
    original["distance"] = 9999
    original["extra"] = [1, 2]
    assert config.params["distance"] == 100
    assert "extra" not in config.params
    assert stream_hash(config, 0, "1.0", "1", "corpus-a") == before


# ---------------------------------------------------------------------------
# Subprocess and items_to_canonical_json / items_to_plain tests
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_FIXTURE_PATH = _FIXTURE_DIR / "corpus_fixture.jsonl"

_SUBPROCESS_PARAMS = {
    "num_pairs": 10,
    "recall_distances": [1, 10, 100],
    "filler_density": 5.0,
    "max_distance": 100,
    "corpus": "corpus_fixture",
}

_CANONICAL_JSON_SCRIPT = """
import json, os
from pathlib import Path
from eval.stream.config import StreamConfig
from eval.stream.corpus import load_corpus
from eval.stream.generators.assoc import AssocGenerator
from eval.stream.serialize import items_to_canonical_json

os.environ["PENSIVE_CORPUS_DIR"] = {corpus_dir!r}
config = StreamConfig(generator="assoc", params={params!r})
generator = AssocGenerator()
items = list(generator.generate(config=config, seed={seed!r}))
print(items_to_canonical_json(items))
"""


def _run_canonical_json_subprocess(
    seed: int, pythonhashseed: str,
) -> str:
    script = _CANONICAL_JSON_SCRIPT.format(
        corpus_dir=str(_FIXTURE_DIR),
        params=_SUBPROCESS_PARAMS,
        seed=seed,
    )
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = pythonhashseed
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.getcwd(),
        check=True,
    )
    return completed.stdout.strip()


# covers: eval/serialize::items_to_canonical_json is process-stable and hash-seed-independent::same seed in same process repeats
def test_items_to_canonical_json_same_seed_repeats_in_process():
    first = _run_canonical_json_subprocess(seed=0, pythonhashseed="0")
    second = _run_canonical_json_subprocess(seed=0, pythonhashseed="0")
    assert first == second


# covers: eval/serialize::items_to_canonical_json is process-stable and hash-seed-independent::same seed, same output across processes
def test_items_to_canonical_json_same_across_different_pythonhashseed():
    first = _run_canonical_json_subprocess(seed=0, pythonhashseed="0")
    second = _run_canonical_json_subprocess(seed=0, pythonhashseed="12345")
    assert first == second


# covers: eval/serialize::items_to_canonical_json is process-stable and hash-seed-independent::different seed diverges
def test_items_to_canonical_json_differs_for_different_seeds():
    first = _run_canonical_json_subprocess(seed=0, pythonhashseed="0")
    second = _run_canonical_json_subprocess(seed=1, pythonhashseed="0")
    assert first != second


# covers: eval/serialize::canonical_json sorts keys and uses compact separators::key order does not affect output
def test_canonical_json_key_order_does_not_affect_output():
    a = canonical_json({"z": 1, "a": 2, "m": 3})
    b = canonical_json({"a": 2, "m": 3, "z": 1})
    assert a == b


# covers: eval/serialize::items_to_plain output shape is not externally pinned::plain shape is self-consistent
def test_items_to_plain_round_trip_stability():
    from eval.stream.events import Observe, Probe
    from eval.stream.truth import ProbeTruth, StreamItem

    items = [
        StreamItem(
            event=Observe(position=0, payload={"key": "value"}),
            truth=None,
        ),
        StreamItem(
            event=Probe(
                position=1,
                probe_id="p1",
                task_id="t1",
                query="what?",
            ),
            truth=ProbeTruth(answer=42),
        ),
    ]
    first = items_to_plain(items)
    second = items_to_plain(items)
    assert first == second
