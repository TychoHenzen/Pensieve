"""Typed stream configuration for Stage -1.

A `StreamConfig` names one generator and carries its parameters. It reads
nothing from the environment, so the same config object always means the
same stream, and `hashing.stream_hash` can turn it into a stable identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _freeze(value: Any) -> Any:
    """Recursively convert `value` into an immutable equivalent.

    Dicts become `MappingProxyType` over a frozen copy of their contents,
    and lists become tuples of frozen items. Other values pass through
    unchanged.
    """
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, kw_only=True)
class StreamConfig:
    """The generator name plus its parameters, with no environment lookups."""

    generator: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", _freeze(dict(self.params)))
