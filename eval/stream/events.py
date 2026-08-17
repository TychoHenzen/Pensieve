"""Event kinds for a Stage -1 stream.

A stream is a deterministic, seeded sequence of these events. Every event
carries a stream position, an optional narration hole for the Stage 0
narration decoder, and a hostile flag for Stage 2 attack streams.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True, kw_only=True)
class _EventBase:
    position: int
    narration: str | None = None
    hostile: bool = False


@dataclass(frozen=True, kw_only=True)
class Observe(_EventBase):
    """Input the subject may learn from."""

    payload: object


@dataclass(frozen=True, kw_only=True)
class Probe(_EventBase):
    """A question with a known answer, scored and isolated from learning."""

    probe_id: str
    task_id: str
    query: str
    teaching_position: int | None = None
    token_distance: int | None = None


@dataclass(frozen=True, kw_only=True)
class Idle(_EventBase):
    """A block of wall time or step budget with no input."""

    budget: int


class BoundaryKind(enum.Enum):
    """The reason a Boundary event marks a break in the stream."""

    SESSION_END = "session_end"
    TASK_SWITCH = "task_switch"
    TASK_TRAINED = "task_trained"
    DISTRIBUTION_SHIFT = "distribution_shift"


@dataclass(frozen=True, kw_only=True)
class Boundary(_EventBase):
    """A marker for a fake session end, a task switch, or a distribution shift."""

    kind: BoundaryKind
    hidden_from_subject: bool = False


Event = Union[Observe, Probe, Idle, Boundary]
