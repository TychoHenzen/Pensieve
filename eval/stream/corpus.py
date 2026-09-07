"""A pinned prose corpus for filler text.

Filler events used to be synthetic key/value pairs behind a `filler-`
prefix. That prefix handed a subject a rule for skipping filler, and a run
of synthetic pairs read like nothing a deployed memory would ever face.
This module replaces that with contiguous spans of real prose, drawn from
a snapshot pinned by `scripts/fetch_corpus.py`.

The repository stores the fetch script and the snapshot's hash, never the
snapshot itself. `resolve` finds the snapshot on disk and raises a clear
error naming the fetch script when it is absent. `load_corpus` reads it
and identifies it by the sha256 of its raw bytes, so a stream's identity
never depends on where a machine happened to put the file.

`Corpus.walk` hands out the prose. It returns a
`eval.stream.corpus_walk.CorpusWalk`, which reads forward from one random
start, so a generator's consecutive filler spans continue one document.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from eval.stream.corpus_walk import CorpusWalk

CORPUS_DIR_ENV = "PENSIVE_CORPUS_DIR"
DEFAULT_CORPUS_DIR = Path("data/corpus")
FETCH_SCRIPT = "scripts/fetch_corpus.py"


def _corpus_dir() -> Path:
    override = os.environ.get(CORPUS_DIR_ENV)
    return Path(override) if override else DEFAULT_CORPUS_DIR


def resolve(name: str) -> Path:
    """Resolve `name` to a snapshot path under the corpus directory.

    The corpus directory defaults to `data/corpus/` and is overridable by
    the `PENSIVE_CORPUS_DIR` environment variable. Raises
    `FileNotFoundError` naming `scripts/fetch_corpus.py` when the snapshot
    is not there, so a missing download fails loudly instead of falling
    back to something silent.
    """
    path = _corpus_dir() / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"corpus snapshot not found at {path}. Run {FETCH_SCRIPT} to fetch "
            f"it, or point {CORPUS_DIR_ENV} at a directory that already holds it."
        )
    return path


@dataclass(frozen=True)
class Corpus:
    """Prose loaded from one snapshot, identified by the hash of its bytes."""

    corpus_id: str
    words: tuple[str, ...]

    def walk(self, source: random.Random) -> CorpusWalk:
        """Start a forward read at a random position in the corpus.

        `source` is the `random.Random` that picks the starting word, so
        the same source seed always yields the same walk and a different
        seed almost always yields a different one.
        """
        return CorpusWalk(self.words, source.randrange(len(self.words)))


def load_corpus(path: Path) -> Corpus:
    """Load a jsonl snapshot from `path` into a `Corpus`.

    Each line is a JSON object with a `text` field, the same shape
    `scripts/fetch_corpus.py` writes. `corpus_id` is the sha256 of the raw
    file bytes, so identity is content-based and never depends on the
    path a machine stored the file at.
    """
    data = Path(path).read_bytes()
    corpus_id = hashlib.sha256(data).hexdigest()

    words: list[str] = []
    for raw_line in data.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        doc = json.loads(line)
        words.extend(doc["text"].split())

    if not words:
        raise ValueError(f"corpus at {path} contains no words")

    return Corpus(corpus_id=corpus_id, words=tuple(words))
