import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.stream.config import StreamConfig
from eval.stream.hashing import canonical_json, stream_hash
from eval.stream.registry import REGISTRY, build

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _corpus_dir(monkeypatch):
    monkeypatch.setenv("PENSIVE_CORPUS_DIR", str(FIXTURE_DIR))


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


# covers: eval/generator::Three generators are registered under fixed names::registry contents
def test_assoc_is_registered_by_name():
    assert "assoc" in REGISTRY


def test_build_resolves_a_config_alone_into_a_non_empty_stream():
    items = list(build(_assoc_config(), seed=0))
    assert len(items) > 0


# covers: eval/generator::Registry resolves a config's generator name::same config and seed replay identically
def test_build_with_same_config_and_seed_gives_identical_sequence():
    first = list(build(_assoc_config(), seed=0))
    second = list(build(_assoc_config(), seed=0))
    assert [item.event for item in first] == [item.event for item in second]
    assert [item.truth for item in first] == [item.truth for item in second]


# covers: eval/generator::Three generators are registered under fixed names::registry contents
def test_split_classify_and_difficulty_mix_in_registry():
    assert "split-classify" in REGISTRY
    assert "difficulty-mix" in REGISTRY


# covers: eval/generator::Registry resolves a config's generator name::unknown generator name rejected
def test_unknown_generator_name_raises_and_lists_known_names():
    config = StreamConfig(generator="does-not-exist", params={})
    with pytest.raises(ValueError) as excinfo:
        build(config, seed=0)
    message = str(excinfo.value)
    assert "does-not-exist" in message
    for name in REGISTRY:
        assert name in message


# -- Configs keyed by generator name, enough to call chance_rate and generate.

_MINIMAL_CONFIGS = {
    "assoc": {
        "num_pairs": 3,
        "recall_distances": [1],
        "filler_density": 1.0,
        "max_distance": 10,
        "corpus": "corpus_fixture",
    },
    "split-classify": {
        "num_tasks": 1,
        "classes_per_task": 2,
        "examples_per_task": 2,
        "probes_per_task": 2,
    },
    "difficulty-mix": {
        "num_items": 4,
        "difficulty_levels": [1],
        "probe_rate": 1.0,
    },
    "gsm8k": {
        "problem_count": 3,
        "split": "test",
    },
    "asdiv-a": {
        "problem_count": 3,
        "split": "test",
    },
}


# covers: eval/generator::Every generator exposes chance_rate::chance_rate callable on every registered generator
@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_chance_rate_callable_on_every_registered_generator(name):
    config = StreamConfig(generator=name, params=_MINIMAL_CONFIGS[name])
    generator = REGISTRY[name]()
    rate = generator.chance_rate(config)
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0


# covers: eval/generator::Every generator exposes chance_rate::chance_rate callable on every registered generator
def test_chance_rate_callable_on_all_generators():
    for name, cls in REGISTRY.items():
        config = StreamConfig(generator=name, params=_MINIMAL_CONFIGS[name])
        generator = cls()
        rate = generator.chance_rate(config)
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0


# covers: eval/generator::Every generator exposes chance_rate::chance_rate returns a float
def test_chance_rate_returns_float():
    config = StreamConfig(generator="assoc", params=_MINIMAL_CONFIGS["assoc"])
    generator = REGISTRY["assoc"]()
    result = generator.chance_rate(config)
    assert type(result) is float


# covers: eval/generator::Registry resolves a config's generator name::error message names all registered generators
def test_unknown_generator_error_names_all_registered():
    config = StreamConfig(generator="bogus", params={})
    with pytest.raises(ValueError) as excinfo:
        build(config, seed=0)
    message = str(excinfo.value)
    for name in REGISTRY:
        assert name in message


# covers: eval/generator::Three generators are registered under fixed names::assoc is registered
def test_assoc_is_registered():
    assert "assoc" in REGISTRY


# covers: eval/generator::Three generators are registered under fixed names::split-classify is registered
def test_split_classify_is_registered():
    assert "split-classify" in REGISTRY


# covers: eval/generator::Three generators are registered under fixed names::difficulty-mix is registered
def test_difficulty_mix_is_registered():
    assert "difficulty-mix" in REGISTRY


# covers: eval/generator::Cross-process and cross-hash-seed replay determinism::same-process replay matches
def test_same_process_replay_matches():
    config = _assoc_config()
    first = list(build(config, seed=0))
    second = list(build(config, seed=0))
    assert [dataclasses.asdict(i.event) for i in first] == [dataclasses.asdict(i.event) for i in second]
    assert [dataclasses.asdict(i.truth) if i.truth else None for i in first] == [
        dataclasses.asdict(i.truth) if i.truth else None for i in second
    ]


# covers: eval/generator::Cross-process and cross-hash-seed replay determinism::subprocess replay matches
def test_subprocess_replay_matches():
    config = _assoc_config()
    seed = 0

    # Generate the stream in this process.
    items = list(build(config, seed=seed))
    events = [canonical_json(dataclasses.asdict(item.event)) for item in items]
    truths = [canonical_json(dataclasses.asdict(item.truth)) if item.truth else "null" for item in items]
    parent_sequence = json.dumps({"events": events, "truths": truths})

    generator_cls = REGISTRY[config.generator]
    parent_hash = stream_hash(config, seed, generator_cls.version, "1", "corpus_fixture")

    # Run the same generation in a subprocess with a different PYTHONHASHSEED.
    script = (
        "import dataclasses, json, sys, os;"
        "sys.path.insert(0, os.environ['PROJECT_ROOT']);"
        "from eval.stream.config import StreamConfig;"
        "from eval.stream.registry import REGISTRY, build;"
        "from eval.stream.hashing import canonical_json, stream_hash;"
        f"config = StreamConfig(generator={config.generator!r}, params=dict({dict(config.params)!r}));"
        f"items = list(build(config, seed={seed}));"
        "events = [canonical_json(dataclasses.asdict(item.event)) for item in items];"
        "truths = [canonical_json(dataclasses.asdict(item.truth)) if item.truth else 'null' for item in items];"
        "sequence = json.dumps({'events': events, 'truths': truths});"
        f"gen_cls = REGISTRY[{config.generator!r}];"
        f"h = stream_hash(config, {seed}, gen_cls.version, '1', 'corpus_fixture');"
        "print(sequence);"
        "print(h)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env={
            **dict(__import__("os").environ),
            "PYTHONHASHSEED": "12345",
            "PENSIVE_CORPUS_DIR": str(FIXTURE_DIR),
            "PROJECT_ROOT": str(Path(__file__).resolve().parents[2]),
        },
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    lines = result.stdout.strip().splitlines()
    child_sequence = lines[0]
    child_hash = lines[1]

    assert parent_sequence == child_sequence
    assert parent_hash == child_hash
