"""A moving read position in a corpus, handing out consecutive prose spans.

Filler prose used to draw each span from an independent random start. The
text then jumped topic every span: a bakery, then arctic terns, then
mountaineering, glued end to end. That is a run of unrelated fragments,
which is the thing prose filler was brought in to replace. A walk reads
forward instead, so consecutive filler events in one stream continue one
document.
"""

from __future__ import annotations


class CorpusWalk:
    """A read position that advances by the length of each span it returns."""

    def __init__(self, words: tuple[str, ...], start: int) -> None:
        self._words = words
        self._position = start

    def next_span(self, target_words: int) -> str:
        """Return the next `target_words` words, continuing from the last span.

        A request that would run past the end of the corpus wraps back to
        word zero and reads forward from there, rather than returning a
        span shorter than asked for. A `target_words` larger than the
        whole corpus is clipped to the corpus length.
        """
        if target_words <= 0:
            raise ValueError(f"target_words must be positive, got {target_words}")
        count = min(target_words, len(self._words))
        if self._position + count > len(self._words):
            self._position = 0
        start = self._position
        self._position += count
        return " ".join(self._words[start : start + count])
