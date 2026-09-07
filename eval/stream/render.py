"""Render stream events into the text every subject sees.

Rendering is the harness's job, not the subject's. Two subjects reading the
same stream therefore see identical wording. `eval.stream.serialize` exists
for hashing and deliberately includes the truth channel. It is not a subject
view and must not be reused as one. This module calls
`eval.stream.truth.subject_view` to strip that channel first. It then calls
`render_event` to turn one event into text. That text never carries an
identifier, a teaching position, or the hostile flag, because those exist
for the harness alone.

An `Observe` renders by the shape of its payload, not by its event class.
Payload shapes vary per generator while the event class does not. One
branch per class would therefore print whatever a payload happened to be.
That branch used to print a Python dict repr, quotes and all, plus 17
digits of float noise per feature. `_PAYLOAD_RENDERERS` names each shape
instead. An unknown payload then fails loudly rather than leaking a repr
into the benchmark.

An `Idle` renders to nothing at all. `eval.stream.events.Idle` is defined
as a block of time with no input, and the subject protocol delivers it
through `idle(budget)`. Rendering it as text would make it input, against
that definition. So `carries_text` returns False for it and `render_event`
refuses it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Protocol, runtime_checkable

from eval.stream.events import Boundary, Event, Idle, Observe, Probe
from eval.stream.truth import StreamItem, subject_view

RENDER_VERSION = "2"

_WORD_PATTERN = re.compile(r"\S+")

# Feature values carry Gaussian noise around a class center. The class
# separation is an order of magnitude wider than that noise, so two
# decimals preserve every bit of signal a subject could use. The digits
# past them are float representation, and printing them would spend tokens
# on nothing.
_FEATURE_DECIMALS = 2


@runtime_checkable
class TokenCounter(Protocol):
    """A callable that counts tokens in rendered text.

    A run record stores `name` alongside the count it produced, so a
    later reader knows which counter produced it. Stage -1 ships only
    the regex-based approximation below. Stage 0 injects a real
    tokenizer through this same seam.
    """

    name: str

    def __call__(self, text: str) -> int: ...


class _RegexTokenCounter:
    """Approximate token count by splitting text on whitespace runs.

    This is not a real tokenizer: it counts whitespace-separated words,
    not model tokens. Stage -1 must not depend on a tokenizer package,
    so this stands in until Stage 0 injects a real one through the same
    `TokenCounter` seam.
    """

    name = "regex-whitespace-v1"

    def __call__(self, text: str) -> int:
        return len(_WORD_PATTERN.findall(text))


default_token_counter: TokenCounter = _RegexTokenCounter()

# A config's params must stay JSON-serializable, so that `stream_hash` can
# hash them. A counter object cannot cross that boundary. So a config names
# its counter, and a generator resolves that name here.
TOKEN_COUNTERS: dict[str, TokenCounter] = {
    default_token_counter.name: default_token_counter,
}


def resolve_token_counter(name: str) -> TokenCounter:
    """Resolve a counter name (as stored in `config.params`) to a counter.

    Raises `ValueError` naming the known counters when `name` is not
    registered, matching how `eval.stream.registry.build` reports an
    unknown generator name.
    """
    counter = TOKEN_COUNTERS.get(name)
    if counter is None:
        known = ", ".join(sorted(TOKEN_COUNTERS))
        raise ValueError(f"unknown token counter {name!r}; known counters: {known}")
    return counter


def _render_fact(payload: Mapping[str, object]) -> str:
    """Render a taught key/value fact, the `assoc` generator's payload.

    The marker keeps a fact visually separate from the prose around it.
    `docs/STREAM-REWORK.md` accepts that on purpose: what gets measured
    is whether a fact survives the distance, not whether a subject can
    find it in the first place.
    """
    return f"[fact] {payload['key']} = {payload['value']}"


def _render_prose(payload: Mapping[str, object]) -> str:
    """Render a corpus prose span as bare text, with no marker at all.

    Filler exists so a taught fact has to survive continuous text. A
    marker around every span would rebuild the run of labeled boxes that
    the prose corpus replaced.
    """
    return str(payload["text"])


def format_features(values: Iterable[float]) -> str:
    """Format a feature vector the one way every subject reads it.

    A `split-classify` probe states its features inside `Probe.query`.
    The generator writes that. A taught example states its features in an
    `Observe` payload, which this module writes. Both go through here. A
    probe and the examples it is compared against therefore never differ
    in wording or in precision.
    """
    inner = ", ".join(f"{float(value):.{_FEATURE_DECIMALS}f}" for value in values)
    return f"features=({inner})"


def _render_example(payload: Mapping[str, object]) -> str:
    """Render a labeled feature vector, the `split-classify` payload.

    A payload's `source` names the dataset item its features came from.
    It stays out of the text. The harness tracks where data came from,
    and a subject must classify on the features alone.
    """
    features = format_features(payload["features"])  # type: ignore[arg-type]
    return f"[example] {features} label={payload['label']}"


_PAYLOAD_RENDERERS = {
    frozenset({"key", "value"}): _render_fact,
    frozenset({"text"}): _render_prose,
    frozenset({"features", "label", "source"}): _render_example,
}


def _render_observe(payload: object) -> str:
    """Dispatch an `Observe` payload to the renderer for its exact shape.

    Raises `ValueError` naming the known shapes when the payload is not a
    mapping, or when its keys match no shape. A new generator then has to
    state how its payload reads. It cannot inherit a dict repr.
    """
    if isinstance(payload, Mapping):
        renderer = _PAYLOAD_RENDERERS.get(frozenset(payload))
        if renderer is not None:
            return renderer(payload)
    known = "; ".join("{" + ", ".join(sorted(shape)) + "}" for shape in _PAYLOAD_RENDERERS)
    raise ValueError(f"no renderer for Observe payload with shape {payload!r}; known payload shapes: {known}")


def carries_text(event: Event) -> bool:
    """Report whether `event` reaches the subject as text.

    False for `Idle` alone. An `Idle` is a block of time with no input,
    delivered through the subject's `idle(budget)` call, so it has no
    place in the rendered stream.
    """
    return not isinstance(event, Idle)


def render_event(event: Event) -> str:
    """Render one event into the text a subject sees.

    The text never carries a `probe_id`, a `task_id`, a
    `teaching_position`, or the `hostile` flag. It never carries a `Probe`
    answer either, because the event holds no answer at all. An answer
    lives only on the truth side channel.

    Raises `ValueError` for an `Idle`, which carries no text. Call
    `carries_text` first when the caller may hold one.
    """
    if isinstance(event, Observe):
        return _render_observe(event.payload)
    if isinstance(event, Probe):
        return f"[probe] {event.query}"
    if isinstance(event, Boundary):
        return f"[boundary] {event.kind.value}"
    if isinstance(event, Idle):
        raise ValueError(  # noqa: TRY004 - Idle is a valid event but has no renderable text
            "an Idle carries no subject text; it reaches the subject through "
            "idle(budget). Filter on render.carries_text before rendering."
        )
    raise TypeError(f"unknown event kind: {type(event).__name__}")


def rendered_subject_view(items: Iterable[StreamItem]) -> Iterator[str]:
    """Render the subject-visible slice of a `StreamItem` sequence.

    Built on `truth.subject_view`, so a hidden `Boundary` never appears
    and no truth ever reaches the renderer. An `Idle` is dropped here too,
    because it reaches the subject as time rather than as text.
    """
    for event in subject_view(items):
        if carries_text(event):
            yield render_event(event)
