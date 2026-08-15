"""Canonical serialization and content hashing for a stream identity.

A stream is addressed by a hash of its config, its seed, its generator
version, its render version and its corpus id. All five go into the hash
together. Config alone would not tell two runs with different seeds apart.
Leaving out the generator version would let a changed generator reuse an
old stream's identity. Two streams that differ only in wording are two
different benchmarks, so the render version enters the hash too. Two
streams drawn from different prose snapshots are two different benchmarks
as well, so the corpus id enters the hash. The corpus id is content-based
(the sha256 of the snapshot bytes, from `eval.stream.corpus.load_corpus`),
so the stream hash never depends on where a machine stored the snapshot
file.

`sha256` is used for the same reason `generator.derive` uses it: the digest
must come out identical in a fresh process, not just within one run.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any

from eval.stream.config import StreamConfig


class _CanonicalEncoder(json.JSONEncoder):
    """Encode `MappingProxyType` as an object and `tuple` as an array.

    `StreamConfig` freezes its parameters into these two immutable types,
    which the standard encoder does not know how to serialize on its own.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, MappingProxyType):
            return dict(o)
        return super().default(o)


def canonical_json(value: Any) -> str:
    """Serialize `value` to a canonical JSON string.

    Object keys are sorted. So two equal configs built in a different key
    order produce the same string. Floats use Python's round-tripping
    `repr`, via the standard encoder. A tuple serializes as a JSON array,
    the same as a list. `StreamConfig` freezes nested lists into tuples,
    which does not change what they mean.

    A value that cannot be encoded this way raises `TypeError`. A `set` is
    one, because its iteration order is not stable. The alternative would
    be to drop back to `repr`, but that could quietly give two different
    configs the same digest.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        cls=_CanonicalEncoder,
    )


def stream_hash(
    config: StreamConfig,
    seed: int,
    generator_version: str,
    render_version: str,
    corpus_id: str,
) -> str:
    """Return a hex digest identifying `config`, `seed`, `generator_version`,
    `render_version` and `corpus_id`.

    All five inputs are hashed together. Config alone would not tell two
    seeds apart. Leaving out the generator version would let a changed
    generator reuse an old stream identity. Leaving out the render version
    would let two streams that differ only in wording share an identity.
    Leaving out the corpus id would let two streams drawn from different
    prose snapshots share an identity.
    """
    payload = {
        "generator": config.generator,
        "params": config.params,
        "seed": seed,
        "generator_version": generator_version,
        "render_version": render_version,
        "corpus_id": corpus_id,
    }
    canonical = canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
