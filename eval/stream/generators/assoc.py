"""Synthetic fact-learning generator: teach key -> value, probe recall.

Teaches `num_pairs` key/value pairs, each at one scheduled position in a
single interleaved timeline. Each pair gets one `Probe` per requested recall
distance, placed at exactly `teaching_position + distance`. Every other
position holds a filler `Observe` carrying a prose span drawn from the
configured corpus, so a taught fact has to survive real text rather than a
run of synthetic pairs. Because pairs interleave, the events between one
pair's teaching and its probe can include other pairs' teachings and
probes, so a probe has to survive other learning rather than being
answered right after its own private block.

The scheduler places each pair by drawing from the still-free positions
that would keep every one of its probes on a free position too, so a
placement is never rejected after the fact. When no free position remains
for a pair, it raises rather than shifting a probe off its requested
distance.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

from eval.stream import corpus, vocab
from eval.stream.config import StreamConfig
from eval.stream.events import Observe, Probe
from eval.stream.generator import derive
from eval.stream.render import carries_text, render_event, resolve_token_counter
from eval.stream.truth import ProbeTruth, StreamItem

# Word count of each filler prose span. Fixed rather than configurable,
# since nothing in the plan asks for it to vary. Spans come off one
# forward walk, so this sets how often the prose is interrupted, not
# which prose a stream reads.
FILLER_SPAN_WORDS = 15


def _validate(num_pairs: int, distances: list[int], max_distance: int, filler_density: float) -> int:
    for distance in distances:
        if distance < 1:
            raise ValueError(f"recall distance must be at least 1, got {distance}")
        if distance > max_distance:
            raise ValueError(f"recall distance {distance} exceeds max_distance {max_distance}")
    if filler_density < 0:
        raise ValueError(f"filler_density must be >= 0, got {filler_density}")

    non_filler = num_pairs * (1 + len(distances))
    filler = round(filler_density * non_filler)
    total_length = non_filler + filler

    reach = max(distances) if distances else 0
    if total_length <= reach:
        raise ValueError(
            f"timeline of {total_length} positions is too short to hold a "
            f"recall distance of {reach}; raise filler_density or num_pairs"
        )
    return total_length


def _valid_candidates(total_length: int, distances: list[int], occupied: set[int]) -> list[int]:
    reach = max(distances) if distances else 0
    candidates = []
    for position in range(total_length - reach):
        if position in occupied:
            continue
        if any(position + distance in occupied for distance in distances):
            continue
        candidates.append(position)
    return candidates


def _schedule(num_pairs: int, distances: list[int], total_length: int, order_source) -> dict[int, int]:
    occupied: set[int] = set()
    teaching_positions: dict[int, int] = {}

    for pair_index in range(num_pairs):
        candidates = _valid_candidates(total_length, distances, occupied)
        if not candidates:
            raise ValueError(
                f"could not schedule pair {pair_index} of {num_pairs} within a "
                f"timeline of {total_length} positions; raise filler_density or "
                "shrink num_pairs/recall_distances"
            )
        candidate = order_source.choice(candidates)
        occupied.add(candidate)
        occupied.update(candidate + distance for distance in distances)
        teaching_positions[pair_index] = candidate
    return teaching_positions


def _fill_token_distances(
    events_by_position: dict[int, StreamItem], token_counter
) -> None:
    """Replace each probe in place with one carrying its measured token distance.

    The distance covers the rendered text of every position strictly
    between a teaching and its probe. Only text a subject actually reads
    counts, so an event that renders to nothing is skipped. A probe sitting
    right after its teaching therefore measures zero.
    """
    for position, item in list(events_by_position.items()):
        if not isinstance(item.event, Probe):
            continue
        span = range(item.event.teaching_position + 1, item.event.position)
        texts = [
            render_event(events_by_position[p].event)
            for p in span
            if carries_text(events_by_position[p].event)
        ]
        events_by_position[position] = StreamItem(
            event=dataclasses.replace(
                item.event, token_distance=token_counter("\n".join(texts)) if texts else 0
            ),
            truth=item.truth,
        )


class AssocGenerator:
    """Teach key -> value pairs once each, probe recall at controlled distances."""

    name = "assoc"
    version = "6"

    def chance_rate(self, config: StreamConfig) -> float:
        """Return the accuracy a random answerer reaches on this generator's probes.

        Every probe answer is a value drawn uniformly from the full
        vocabulary, so chance is one over the vocabulary size regardless
        of `config`.
        """
        return vocab.chance_rate(len(vocab.VOCAB))

    def generate(self, config: StreamConfig, seed: int) -> Iterator[StreamItem]:
        params = config.params
        num_pairs = params["num_pairs"]
        distances = list(params["recall_distances"])
        filler_density = float(params["filler_density"])
        max_distance = params["max_distance"]
        if "corpus" not in params:
            raise ValueError(
                "assoc generator config is missing a 'corpus' param; a config "
                "must name the corpus snapshot to draw filler text from"
            )
        corpus_name = params["corpus"]
        token_counter_name = params.get("token_counter")
        # `config.params` must stay JSON-serializable so `stream_hash` can
        # hash it, so a config names its counter rather than carrying the
        # callable itself; resolve the name to a counter here.
        token_counter = (
            resolve_token_counter(token_counter_name)
            if token_counter_name is not None
            else None
        )

        total_length = _validate(num_pairs, distances, max_distance, filler_density)

        keys_source = derive(seed, "keys")
        values_source = derive(seed, "values")
        order_source = derive(seed, "order")
        filler_source = derive(seed, "filler")

        # Only the num_pairs taught keys need to be distinct, so a probe
        # query never matches more than one taught key. Drawing one
        # distinct key per stream position instead would cap a stream at
        # the vocabulary size, far below what a long recall distance
        # needs. Values draw from the same vocabulary but may repeat,
        # since only a key needs to disambiguate a probe.
        keys = vocab.sample(keys_source, num_pairs)
        values = [values_source.choice(vocab.VOCAB) for _ in range(num_pairs)]

        teaching_positions = _schedule(num_pairs, distances, total_length, order_source)

        # One forward walk per stream, so consecutive filler events
        # continue one document instead of jumping between unrelated
        # parts of the snapshot.
        filler_walk = corpus.load_corpus(corpus.resolve(corpus_name)).walk(filler_source)

        events_by_position: dict[int, StreamItem] = {}
        for pair_index, teach_position in teaching_positions.items():
            key = keys[pair_index]
            value = values[pair_index]
            events_by_position[teach_position] = StreamItem(
                event=Observe(
                    position=teach_position,
                    payload={"key": key, "value": value},
                ),
                truth=None,
            )
            for distance in distances:
                probe_position = teach_position + distance
                events_by_position[probe_position] = StreamItem(
                    event=Probe(
                        position=probe_position,
                        probe_id=f"assoc-{pair_index}-{distance}-{probe_position}",
                        task_id="assoc",
                        query=key,
                        teaching_position=teach_position,
                    ),
                    truth=ProbeTruth(answer=value),
                )

        if token_counter is None:
            for position in range(total_length):
                if position in events_by_position:
                    yield events_by_position[position]
                    continue
                yield StreamItem(
                    event=Observe(
                        position=position,
                        payload={"text": filler_walk.next_span(FILLER_SPAN_WORDS)},
                    ),
                    truth=None,
                )
            return

        # A configured counter means the caller wants a real token_distance,
        # which needs the rendered text of every position between a
        # teaching and its probe. That forces materializing filler events
        # up front instead of generating them lazily in the yield loop
        # below, so each probe can look back at the events it needs.
        for position in range(total_length):
            if position not in events_by_position:
                events_by_position[position] = StreamItem(
                    event=Observe(
                        position=position,
                        payload={"text": filler_walk.next_span(FILLER_SPAN_WORDS)},
                    ),
                    truth=None,
                )

        _fill_token_distances(events_by_position, token_counter)

        for position in range(total_length):
            yield events_by_position[position]
