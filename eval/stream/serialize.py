"""Turn a stream item sequence into a stable, comparable form.

A run record will need to write a probe log later, and that log is the same
operation as comparing two replayed streams here: turn each event plus its
truth into plain data, then hand it to `hashing.canonical_json` so the
result never drifts from the digest the harness already trusts.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable
from typing import Any

from eval.stream.hashing import canonical_json
from eval.stream.truth import StreamItem


def _plain(value: Any) -> Any:
    """Recursively convert `value` into data `canonical_json` can encode."""
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        return {field.name: _plain(getattr(value, field.name)) for field in fields}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def items_to_plain(items: Iterable[StreamItem]) -> list[dict]:
    """Turn a `StreamItem` sequence into plain data, event and truth included."""
    plain = []
    for item in items:
        plain.append(
            {
                "event": {"type": type(item.event).__name__, **_plain(item.event)},
                "truth": _plain(item.truth) if item.truth is not None else None,
            }
        )
    return plain


def items_to_canonical_json(items: Iterable[StreamItem]) -> str:
    """Serialize a `StreamItem` sequence into a canonical, comparable string."""
    return canonical_json(items_to_plain(items))
