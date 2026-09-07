import hashlib
import random
import re
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
    with pytest.raises(FileNotFoundError, match=re.escape("fetch_corpus.py")):
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


# covers: eval/corpus::resolve locates a snapshot under an overridable directory::default directory is data/corpus
def test_resolve_default_directory_is_data_corpus(monkeypatch, tmp_path):
    corpus_dir = tmp_path / "data" / "corpus"
    corpus_dir.mkdir(parents=True)
    fixture = corpus_dir / "test.jsonl"
    fixture.write_text('{"text": "hello world"}\n', encoding="utf-8")
    monkeypatch.delenv("PENSIVE_CORPUS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    path = resolve("test")
    assert path.resolve() == fixture.resolve()


# covers: eval/corpus::resolve raises FileNotFoundError naming the fetch script::error message names the missing corpus
def test_resolve_error_names_missing_corpus(monkeypatch, tmp_path):
    monkeypatch.setenv("PENSIVE_CORPUS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve("my-corpus")
    message = str(excinfo.value)
    assert "fetch_corpus.py" in message
    assert "my-corpus" in message


# covers: eval/corpus::corpus_id is the SHA-256 of the raw file bytes::corpus_id equals SHA-256 hex digest
def test_corpus_id_equals_sha256_hex_digest():
    corpus = load_corpus(FIXTURE_PATH)
    expected = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    assert corpus.corpus_id == expected


# covers: eval/corpus::corpus_id is the SHA-256 of the raw file bytes::different content produces different corpus_id
def test_corpus_id_differs_for_different_content(tmp_path):
    a = tmp_path / "a.jsonl"
    a.write_text('{"text": "alpha beta gamma delta epsilon"}\n', encoding="utf-8")
    b = tmp_path / "b.jsonl"
    b.write_text('{"text": "zeta eta theta iota kappa"}\n', encoding="utf-8")
    assert load_corpus(a).corpus_id != load_corpus(b).corpus_id


# covers: eval/corpus::corpus_id is the SHA-256 of the raw file bytes::corpus_id is a 64-character hex string
def test_corpus_id_is_64_hex_characters():
    corpus = load_corpus(FIXTURE_PATH)
    assert len(corpus.corpus_id) == 64
    assert all(c in "0123456789abcdef" for c in corpus.corpus_id)


# covers: eval/corpus::Corpus.walk is reproducible from a seed and diverges across seeds::different seed, different walk
def test_walk_different_seed_different_walk():
    corpus = load_corpus(FIXTURE_PATH)
    a = corpus.walk(random.Random(0)).next_span(50)
    b = corpus.walk(random.Random(1)).next_span(50)
    assert a != b


# covers: eval/corpus::next_span returns exactly the requested word count::single word request
def test_span_single_word():
    corpus = load_corpus(FIXTURE_PATH)
    text = corpus.walk(random.Random(0)).next_span(1)
    assert len(text.split()) == 1


# covers: eval/corpus::consecutive spans continue one document::two consecutive spans form a continuous excerpt
def test_two_consecutive_spans_form_continuous_excerpt():
    corpus = load_corpus(FIXTURE_PATH)
    walk = corpus.walk(random.Random(0))
    a = walk.next_span(5)
    b = walk.next_span(5)
    joined = f"{a} {b}"
    assert joined in " ".join(corpus.words)


# covers: eval/corpus::a walk past the end wraps and stays full length::wrap produces valid corpus text
def test_wrap_produces_valid_corpus_text():
    corpus = load_corpus(FIXTURE_PATH)
    walk = corpus.walk(random.Random(0))
    total = len(corpus.words)
    for _ in range(total // 10 + 3):
        walk.next_span(10)
    wrapped = walk.next_span(5)
    for word in wrapped.split():
        assert word in corpus.words


# covers: eval/corpus::a request larger than the whole corpus is clipped::request exactly at corpus length
def test_span_request_exactly_at_corpus_length():
    corpus = load_corpus(FIXTURE_PATH)
    text = corpus.walk(random.Random(0)).next_span(len(corpus.words))
    assert len(text.split()) == len(corpus.words)


# covers: eval/corpus::a non-positive word count raises ValueError::negative request
def test_span_rejects_negative_word_count():
    corpus = load_corpus(FIXTURE_PATH)
    with pytest.raises(ValueError, match="target_words must be positive"):
        corpus.walk(random.Random(0)).next_span(-5)


# covers: eval/corpus::a non-positive word count raises ValueError::error message matches exact text
def test_span_error_message_matches_exact_text():
    corpus = load_corpus(FIXTURE_PATH)
    with pytest.raises(ValueError) as excinfo:
        corpus.walk(random.Random(0)).next_span(0)
    assert "target_words must be positive" in str(excinfo.value)


# covers: eval/corpus::load_corpus rejects a wordless snapshot::snapshot with no words
def test_load_corpus_rejects_wordless_snapshot(tmp_path):
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text('{"text": ""}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no words"):
        load_corpus(empty_path)
