"""The truth side channel for a Stage -1 stream.

A stream generator yields events plus the ground truth needed to score each
probe, on a side channel the subject never sees. `StreamItem` pairs one event
with its truth and enforces, at construction, that only a `Probe` carries
truth. `subject_view` strips the channel and hides any `Boundary` marked
`hidden_from_subject`. `harness_view` yields everything.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from eval.stream.events import Boundary, Event, Probe


@dataclass(frozen=True, kw_only=True)
class ProbeTruth:
    """The correct answer to a probe, plus whatever the scorer needs.

    `difficulty` is optional so existing callers (and existing generators)
    keep working unchanged. `difficulty-mix` is the first generator to set
    it: there, difficulty is an arithmetic chain length, chosen so a hard
    probe genuinely costs more work to answer, not just a label. It lives
    only on this truth channel. A subject that could read `difficulty`
    off the rendered event could fake the correlation between compute and
    difficulty rather than actually doing more work on harder probes.
    """

    answer: object
    difficulty: int | None = None


@dataclass(frozen=True, kw_only=True)
class StreamItem:
    """One stream event paired with its truth on the isolated side channel."""

    event: Event
    truth: ProbeTruth | None

    def __post_init__(self) -> None:
        is_probe = isinstance(self.event, Probe)
        if is_probe and self.truth is None:
            raise ValueError("a Probe StreamItem must carry a ProbeTruth")
        if not is_probe and self.truth is not None:
            raise ValueError("only a Probe StreamItem may carry a ProbeTruth")


def subject_view(items: Iterable[StreamItem]) -> Iterator[Event]:
    """Yield only what the subject is allowed to see: no truth, no hidden boundary."""
    for item in items:
        if isinstance(item.event, Boundary) and item.event.hidden_from_subject:
            continue
        yield item.event


def harness_view(items: Iterable[StreamItem]) -> Iterator[StreamItem]:
    """Yield every stream item, truth included, for the harness."""
    yield from items
