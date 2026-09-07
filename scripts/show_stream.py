"""Print what each generator actually shows a subject, and what it measures.

Every other view of a stream is indirect. Tests assert on fields, and
`stream_hash` reduces a whole run to one digest. Neither shows the text a
subject reads. This script builds a stream from an example config and
prints it: the rendered subject line for each event, plus the harness-only
truth for each probe on an indented line below it. Above that it states
what the generator teaches, what it asks, and what a wrong answer means.

The truth lines are marked and indented because no subject ever sees them.
They come from `StreamItem.truth`, not from `render_event`.

An `Idle` prints as a dimmed marker with no text beside it. That is not a
display choice. `render.carries_text` returns False for an `Idle`, so no
subject ever receives one as text.

Usage:

    python scripts/show_stream.py                          # every generator
    python scripts/show_stream.py --generator assoc --all
    python scripts/show_stream.py --generator difficulty-mix --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

from eval.stream import corpus, registry
from eval.stream.config import StreamConfig
from eval.stream.events import Boundary, Probe
from eval.stream.hashing import stream_hash
from eval.stream.render import (
    RENDER_VERSION,
    carries_text,
    default_token_counter,
    render_event,
)
from eval.stream.truth import StreamItem


class Example:
    """One generator's example config plus the plain-language account of it."""

    def __init__(self, params: dict, teaches: str, asks: str, wrong_means: str) -> None:
        self.params = params
        self.teaches = teaches
        self.asks = asks
        self.wrong_means = wrong_means


# Example params match the configs the tests build. They are small on
# purpose: an excerpt is meant to be read, not scrolled.
EXAMPLES: dict[str, Example] = {
    "assoc": Example(
        params={
            "num_pairs": 3,
            "recall_distances": [1, 2, 5],
            "filler_density": 1.0,
            "max_distance": 10,
            "token_counter": default_token_counter.name,
        },
        teaches="One fact per pair, stated once, as 'key = value' over a closed "
        "word list. Prose from a pinned corpus runs between the facts, so a "
        "fact has to survive continuous text.",
        asks="The key alone, at a fixed number of events after the teaching. "
        "Every pair is asked at every requested distance.",
        wrong_means="The fact was lost. It cannot mean a garbled copy, because "
        "the answer is one word from a list both sides know.",
    ),
    "split-classify": Example(
        params={
            "num_tasks": 3,
            "classes_per_task": 2,
            "examples_per_task": 4,
            "probes_per_task": 2,
        },
        teaches="Labeled feature vectors, one task at a time. A task's classes "
        "sit far apart relative to their noise, so each task is easy alone.",
        asks="An unlabeled vector from every task seen so far, after each task "
        "finishes. That fills a retention matrix: rows are the task asked, "
        "columns are the task just learned.",
        wrong_means="Learning the new task overwrote the old one. This is the "
        "generator that hosts the reproduction gate.",
    ),
    "difficulty-mix": Example(
        params={
            "num_items": 12,
            "difficulty_levels": [1, 2, 3, 5],
            "probe_rate": 0.5,
        },
        teaches="Nothing. Each probe is self-contained, so this measures compute rather than memory.",
        asks="An arithmetic chain, stated in full, answered modulo 1000. Chain "
        "length is the difficulty and it lives on the truth channel alone.",
        wrong_means="The subject did not do the arithmetic. Pair the answers "
        "with the cost counters: compute should rise with difficulty.",
    ),
}

# The committed corpus fixture the tests draw from. The real snapshot is a
# multi-million-word download, so the script falls back to this when the
# snapshot is absent and says so in its header.
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "stream" / "fixtures"
FIXTURE_CORPUS = "corpus_fixture"
DEFAULT_CORPUS = "pile-val"

# Streams with no corpus still need a corpus id for `stream_hash`. Only
# `assoc` reads prose, so the other two hash under this fixed marker.
NO_CORPUS_ID = "none"

_WRAP = textwrap.TextWrapper(width=78, initial_indent="  ", subsequent_indent="  ")


def _select_corpus(name: str) -> tuple[str, str, str]:
    """Pick the corpus to draw filler prose from, preferring the real snapshot.

    Returns `(corpus_name, corpus_id, note)`. When `name` is not on disk,
    this falls back to the committed test fixture and returns a note
    saying so, so nobody mistakes fixture prose for the pinned snapshot.
    """
    try:
        path = corpus.resolve(name)
    except FileNotFoundError:
        os.environ[corpus.CORPUS_DIR_ENV] = str(FIXTURE_DIR)
        path = corpus.resolve(FIXTURE_CORPUS)
        note = (
            f"corpus {name!r} is not on disk, so this excerpt draws filler from "
            f"the committed test fixture. Run {corpus.FETCH_SCRIPT} for the real one."
        )
        return FIXTURE_CORPUS, corpus.load_corpus(path).corpus_id, note
    return name, corpus.load_corpus(path).corpus_id, ""


def _build_config(generator: str, corpus_name: str) -> StreamConfig:
    params = dict(EXAMPLES[generator].params)
    if generator == "assoc":
        params["corpus"] = corpus_name
    return StreamConfig(generator=generator, params=params)


def _truncate(text: str, width: int) -> str:
    if width <= 0 or len(text) <= width:
        return text
    return text[: width - 3] + "..."


def _truth_line(item: StreamItem) -> str:
    """Describe a probe's harness-only truth: answer, distances, difficulty."""
    event = item.event
    if not isinstance(event, Probe) or item.truth is None:
        raise ValueError("truth line requires a probe with harness truth")
    parts = [f"answer={item.truth.answer!r}"]
    if event.teaching_position is not None:
        parts.append(f"taught_at={event.teaching_position}")
    if event.token_distance is not None:
        parts.append(f"token_distance={event.token_distance}")
    if item.truth.difficulty is not None:
        parts.append(f"difficulty={item.truth.difficulty}")
    parts.append(f"probe_id={event.probe_id}")
    return "  ".join(parts)


def _print_purpose(name: str) -> None:
    example = EXAMPLES[name]
    print("teaches")
    print(_WRAP.fill(example.teaches))
    print("asks")
    print(_WRAP.fill(example.asks))
    print("a wrong answer means")
    print(_WRAP.fill(example.wrong_means))
    print()


def _print_identity(config: StreamConfig, seed: int, items: list[StreamItem], ids: dict) -> None:
    generator = registry.REGISTRY[config.generator]()
    digest = stream_hash(
        config=config,
        seed=seed,
        generator_version=generator.version,
        render_version=RENDER_VERSION,
        corpus_id=ids["corpus_id"],
    )
    probes = sum(1 for item in items if isinstance(item.event, Probe))
    print(f"  seed {seed}   params {json.dumps(dict(config.params), sort_keys=True)}")
    print(f"  render v{RENDER_VERSION}   corpus {ids['corpus_id'][:12]}   hash {digest[:12]}")
    chance_rate = getattr(generator, "chance_rate", None)
    if not callable(chance_rate):
        raise TypeError(f"generator {config.generator!r} does not expose chance_rate")
    print(f"  {len(items)} events, {probes} probes, chance {chance_rate(config):.4f}")
    if ids["note"]:
        print(_WRAP.fill(ids["note"]))
    print()


def _event_line(item: StreamItem, width: int) -> list[str]:
    """Format one event as its subject line plus any harness-only truth line."""
    event = item.event
    if not carries_text(event):
        return [f"{event.position:>4}  (idle, budget={getattr(event, 'budget', 0)}, no text)"]
    hidden = isinstance(event, Boundary) and event.hidden_from_subject
    marker = "   [hidden from subject]" if hidden else ""
    lines = [f"{event.position:>4}  {_truncate(render_event(event), width)}{marker}"]
    if isinstance(event, Probe):
        lines.append(f"      truth: {_truncate(_truth_line(item), width)}")
    return lines


def _print_events(items: list[StreamItem], limit: int, width: int) -> None:
    shown = items if limit <= 0 else items[:limit]
    for item in shown:
        for line in _event_line(item, width):
            print(line)
    remaining = len(items) - len(shown)
    if remaining:
        print(f"\n  ... {remaining} more events. Pass --all to print them.")


def _show(name: str, args: argparse.Namespace) -> None:
    corpus_name, corpus_id, note = _select_corpus(args.corpus) if name == "assoc" else (args.corpus, NO_CORPUS_ID, "")
    config = _build_config(name, corpus_name)
    items = list(registry.build(config, seed=args.seed))
    generator = registry.REGISTRY[name]()

    print("=" * 78)
    print(f"{name}  v{generator.version}")
    print("=" * 78)
    _print_purpose(name)
    _print_identity(config, args.seed, items, {"corpus_id": corpus_id, "note": note})
    _print_events(items, 0 if args.all else args.limit, args.width)
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator",
        choices=[*sorted(EXAMPLES), "all"],
        default="all",
        help="one generator, or every registered one",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20, help="events to print")
    parser.add_argument("--all", action="store_true", help="print every event")
    parser.add_argument("--width", type=int, default=96, help="0 to disable truncation")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, help="corpus snapshot name")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    names = sorted(EXAMPLES) if args.generator == "all" else [args.generator]
    print("Subject sees the event lines. An indented 'truth:' line is")
    print("harness-only and never reaches the subject.\n")
    for name in names:
        _show(name, args)


if __name__ == "__main__":
    main()
