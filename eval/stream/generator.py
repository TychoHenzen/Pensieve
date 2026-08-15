"""The stream generator protocol and seed derivation for Stage -1.

A stream generator takes a config and a run seed, and yields `StreamItem`
values in a deterministic order. `derive` turns one run seed into many
independent named random sources, so a generator's draw order never
determines its numbers: `derive(seed, "keys")` always yields the same
sequence regardless of what else the generator draws from `derive(seed,
"order")`, in this process or a fresh one.
"""

from __future__ import annotations

import hashlib
import random
import typing
from collections.abc import Iterator

from eval.stream.config import StreamConfig
from eval.stream.truth import StreamItem


def derive(seed: int, name: str) -> random.Random:
    """Build an independent random source from a run seed and a source name.

    The source seed comes from hashing `name` together with `seed`, so
    adding a new named source never shifts the numbers an existing source
    produces, and the result never depends on Python's per-process string
    hash salt.
    """
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


@typing.runtime_checkable
class StreamGenerator(typing.Protocol):
    """A named, versioned source of a deterministic stream of `StreamItem`."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def generate(self, config: StreamConfig, seed: int) -> Iterator[StreamItem]: ...
