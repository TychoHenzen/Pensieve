import hashlib
import random
from pathlib import Path

import pytest

from eval.stream.corpus import load_corpus, resolve

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_PATH = FIXTURE_DIR / "corpus_fixture.jsonl"


# covers: eval/corpus::resolve locates a snapshot under an overridable directory::env var overrides the default directory
def test_resolve_finds_fixture_under_pensive_corpus_dir(monkeypatch):
    monkeypatch.setenv("PENSIVE_CORPUS_DIR", str(FIXTURE_DIR))
    path = resolve("corpus_fixture")
    assert path == FIXTURE_PATH


# covers: eval/corpus::resolve raises FileNotFoundError naming the fetch script::missing snapshot
def test_resolve_raises_and_names_fetch_script_when_snapshot_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("PENSIVE_CORPUS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="fetch_corpus.py"):
        resolve("missing-corpus")


# covers: eval/corpus::corpus_id is the SHA-256 of the raw file bytes::identity is content-based
def test_corpus_id_is_sha256_of_file_content():
    corpus = load_corpus(FIXTURE_PATH)
    expected = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    assert corpus.corpus_id == expected


# covers: eval/corpus::corpus_id is the SHA-256 of the raw file bytes::identity is content-based
def test_corpus_id_changes_when_content_changes(tmp_path):
    first_path = tmp_path / "a.jsonl"
    first_path.write_text('{"text": "one two three four five"}\n', encoding="utf-8")
    second_path = tmp_path / "b.jsonl"
    second_path.write_text('{"text": "one two three four six"}\n', encoding="utf-8")

    first = load_corpus(first_path)
    second = load_corpus(second_path)

    assert first.corpus_id != second.corpus_id


# covers: eval/corpus::Corpus.walk is reproducible from a seed and diverges across seeds::same seed, same walk
def test_walk_is_reproducible_from_the_same_seed():
    corpus = load_corpus(FIXTURE_PATH)
    first = corpus.walk(random.Random(0)).next_span(50)
    second = corpus.walk(random.Random(0)).next_span(50)
    assert first == second


# covers: eval/corpus::Corpus.walk is reproducible from a seed and diverges across seeds::same seed, same walk
def test_walk_diverges_across_seeds():
    corpus = load_corpus(FIXTURE_PATH)
    first = corpus.walk(random.Random(0)).next_span(50)
    second = corpus.walk(random.Random(1)).next_span(50)
    assert first != second


# covers: eval/corpus::next_span returns exactly the requested word count::exact word count
def test_span_has_the_requested_word_count():
    corpus = load_corpus(FIXTURE_PATH)
    text = corpus.walk(random.Random(0)).next_span(50)
    assert len(text.split()) == 50


# covers: eval/corpus::consecutive spans continue one document::forward continuation
def test_consecutive_spans_continue_one_document():
    corpus = load_corpus(FIXTURE_PATH)
    walk = corpus.walk(random.Random(0))
    joined = f"{walk.next_span(10)} {walk.next_span(10)} {walk.next_span(10)}"
    assert joined in " ".join(corpus.words)


# covers: eval/corpus::a walk past the end wraps and stays full length::wrap at the boundary
def test_a_walk_past_the_end_wraps_to_the_start_and_stays_full_length():
    corpus = load_corpus(FIXTURE_PATH)
    walk = corpus.walk(random.Random(0))
    span_words = 40
    spans = [walk.next_span(span_words) for _ in range(len(corpus.words) // span_words + 2)]
    assert all(len(span.split()) == span_words for span in spans)
    assert spans[-1] in " ".join(corpus.words)


# covers: eval/corpus::a request larger than the whole corpus is clipped::oversized request
def test_span_clips_a_request_larger_than_the_corpus():
    corpus = load_corpus(FIXTURE_PATH)
    text = corpus.walk(random.Random(0)).next_span(len(corpus.words) * 2)
    assert len(text.split()) == len(corpus.words)


# covers: eval/corpus::a non-positive word count raises ValueError::zero request
def test_span_rejects_a_non_positive_word_count():
    corpus = load_corpus(FIXTURE_PATH)
    with pytest.raises(ValueError, match="target_words must be positive"):
        corpus.walk(random.Random(0)).next_span(0)


# covers: eval/corpus::load_corpus rejects a wordless snapshot::snapshot with no words
def test_load_corpus_rejects_wordless_snapshot(tmp_path):
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text('{"text": ""}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no words"):
        load_corpus(empty_path)
