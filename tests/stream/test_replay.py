"""End-to-end replay check: the same config and seed give the same stream.

This is the phase 2 end condition from `docs/STAGE-MINUS-1.md`: a stream
that replays the same way twice. Same-process replay alone would not catch
a bug where ordering or hashing quietly depends on Python's per-process
string hash salt, so this also replays across fresh subprocesses and across
different `PYTHONHASHSEED` values.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eval.stream.config import StreamConfig
from eval.stream.corpus import load_corpus
from eval.stream.generators.assoc import AssocGenerator
from eval.stream.hashing import stream_hash
from eval.stream.render import RENDER_VERSION
from eval.stream.serialize import items_to_canonical_json

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_FIXTURE_PATH = _FIXTURE_DIR / "corpus_fixture.jsonl"
_CORPUS_ID = load_corpus(_FIXTURE_PATH).corpus_id

_PARAMS = {
    "num_pairs": 10,
    "recall_distances": [1, 10, 100],
    "filler_density": 5.0,
    "max_distance": 100,
    "corpus": "corpus_fixture",
}


@pytest.fixture(autouse=True)
def _corpus_dir(monkeypatch):
    monkeypatch.setenv("PENSIVE_CORPUS_DIR", str(_FIXTURE_DIR))


_SUBPROCESS_SCRIPT = """
import json
from pathlib import Path

from eval.stream.config import StreamConfig
from eval.stream.corpus import load_corpus
from eval.stream.generators.assoc import AssocGenerator
from eval.stream.hashing import stream_hash
from eval.stream.render import RENDER_VERSION
from eval.stream.serialize import items_to_canonical_json

fixture_path = Path({fixture_path!r})
corpus_id = load_corpus(fixture_path).corpus_id

config = StreamConfig(generator="assoc", params={params!r})
generator = AssocGenerator()
items = list(generator.generate(config=config, seed={seed!r}))
result = {{
    "hash": stream_hash(config, {seed!r}, generator.version, RENDER_VERSION, corpus_id),
    "sequence": items_to_canonical_json(items),
}}
print(json.dumps(result))
"""


def _config(seed_params=None):
    return StreamConfig(generator="assoc", params=seed_params or _PARAMS)


def _run_subprocess(seed: int, pythonhashseed: str | None) -> dict:
    """Run the assoc generator in a fresh interpreter and return its hash and sequence."""
    script = _SUBPROCESS_SCRIPT.format(fixture_path=str(_FIXTURE_PATH), params=_PARAMS, seed=seed)
    env = dict(os.environ)
    if pythonhashseed is None:
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = pythonhashseed
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.getcwd(),
        check=True,
    )
    return json.loads(completed.stdout)


def test_same_config_and_seed_give_identical_events_and_truth_within_one_process():
    generator = AssocGenerator()
    config = _config()
    first = list(generator.generate(config=config, seed=0))
    second = list(generator.generate(config=config, seed=0))
    assert [item.event for item in first] == [item.event for item in second]
    assert [item.truth for item in first] == [item.truth for item in second]
    assert len(first) == 240


def test_subprocess_replay_matches_the_parent_processs_hash_and_sequence():
    config = _config()
    generator = AssocGenerator()
    parent_hash = stream_hash(config, 0, generator.version, RENDER_VERSION, _CORPUS_ID)
    parent_sequence = items_to_canonical_json(generator.generate(config=config, seed=0))

    child = _run_subprocess(seed=0, pythonhashseed=None)

    assert child["hash"] == parent_hash
    assert child["sequence"] == parent_sequence


def test_replay_matches_across_different_pythonhashseed_values():
    first = _run_subprocess(seed=0, pythonhashseed="0")
    second = _run_subprocess(seed=0, pythonhashseed="1")

    assert first["hash"] == second["hash"]
    assert first["sequence"] == second["sequence"]


def test_different_seed_gives_a_different_sequence_and_hash():
    first = _run_subprocess(seed=0, pythonhashseed="0")
    second = _run_subprocess(seed=1, pythonhashseed="0")

    assert first["hash"] != second["hash"]
    assert first["sequence"] != second["sequence"]
