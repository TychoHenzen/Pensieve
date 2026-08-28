## Purpose

Loads prose corpora from JSONL snapshots and provides a wrapping walk that supplies continuous filler text to stream generators.

## Requirements

### Requirement: resolve locates a snapshot under an overridable directory

`resolve(name)` MUST join `name` with `.jsonl` under a directory that defaults to `data/corpus/` and is overridden by the `PENSIVE_CORPUS_DIR` environment variable. [REQUIRED]

#### Scenario: env var overrides the default directory

- **GIVEN** `PENSIVE_CORPUS_DIR` is set to a directory containing `corpus_fixture.jsonl`
- **WHEN** `resolve("corpus_fixture")` is called
- **THEN** it returns the path to that file

#### Scenario: default directory is data/corpus

- **GIVEN** `PENSIVE_CORPUS_DIR` is not set
- **WHEN** `resolve("test")` is called
- **THEN** the returned path is under `data/corpus/`

### Requirement: resolve raises FileNotFoundError naming the fetch script

`resolve` MUST raise `FileNotFoundError` with a message containing `fetch_corpus.py` when the snapshot is absent. [REQUIRED]

#### Scenario: missing snapshot

- **GIVEN** the corpus directory holds no matching file
- **WHEN** `resolve("missing-corpus")` is called
- **THEN** `FileNotFoundError` is raised whose message matches `fetch_corpus.py`

#### Scenario: error message names the missing corpus

- **GIVEN** the corpus directory holds no matching file
- **WHEN** `resolve("my-corpus")` is called
- **THEN** the error message contains both `fetch_corpus.py` and `my-corpus`

### Requirement: corpus_id is the SHA-256 of the raw file bytes

`load_corpus` MUST set `Corpus.corpus_id` to the hex SHA-256 digest of the file's raw bytes. Two files with different content MUST produce different `corpus_id` values. [REQUIRED]

#### Scenario: identity is content-based

- **GIVEN** two files with different byte content
- **WHEN** each is loaded with `load_corpus`
- **THEN** their `corpus_id` values differ, and each equals the SHA-256 hex digest of its own bytes

#### Scenario: corpus_id equals SHA-256 hex digest

- **GIVEN** a file with known bytes
- **WHEN** `load_corpus` is called
- **THEN** `corpus_id` equals the SHA-256 hex digest of those bytes

#### Scenario: different content produces different corpus_id

- **GIVEN** two files with different byte content
- **WHEN** each is loaded with `load_corpus`
- **THEN** their `corpus_id` values differ

#### Scenario: corpus_id is a 64-character hex string

- **GIVEN** any valid corpus file
- **WHEN** `load_corpus` is called
- **THEN** `corpus_id` is 64 characters long and all hex

### Requirement: Corpus.walk is reproducible from a seed and diverges across seeds

`Corpus.walk(source)` MUST produce a walk whose output is identical for two `random.Random` sources built from the same seed, and differs across different seeds. [REQUIRED]

#### Scenario: same seed, same walk

- **GIVEN** a loaded corpus
- **WHEN** `walk(random.Random(0))` is called twice and `next_span(50)` is read from each
- **THEN** the two spans are identical

#### Scenario: different seed, different walk

- **GIVEN** a loaded corpus
- **WHEN** `walk(random.Random(0))` and `walk(random.Random(1))` each produce `next_span(50)`
- **THEN** the two spans differ

### Requirement: next_span returns exactly the requested word count

`next_span(target_words)` MUST return a string whose whitespace-split word count equals `target_words`. [REQUIRED]

#### Scenario: exact word count

- **GIVEN** a corpus walk
- **WHEN** `next_span(50)` is called
- **THEN** the returned string splits into exactly 50 words

#### Scenario: single word request

- **GIVEN** a corpus walk
- **WHEN** `next_span(1)` is called
- **THEN** the returned string splits into exactly 1 word

### Requirement: consecutive spans continue one document

Two or more consecutive calls to `next_span` on the same walk MUST return text that, joined with single spaces, appears verbatim as a substring in the corpus's full word sequence. [REQUIRED]

#### Scenario: forward continuation

- **GIVEN** a fresh walk
- **WHEN** `next_span(10)` is called three times and the results joined with spaces
- **THEN** that joined string is a substring of the corpus text

#### Scenario: two consecutive spans form a continuous excerpt

- **GIVEN** a fresh walk
- **WHEN** `next_span(5)` is called twice
- **THEN** the two spans joined with a space are a substring of the corpus text

### Requirement: a walk past the end wraps and stays full length

When advancing past the corpus length, `next_span` MUST wrap the read position to word zero and return a full-length span. [REQUIRED]

#### Scenario: wrap at the boundary

- **GIVEN** a walk near the end of the corpus
- **WHEN** enough `next_span` calls are made to run past the corpus length
- **THEN** every returned span still has exactly the requested word count

#### Scenario: wrap produces valid corpus text

- **GIVEN** a walk that wraps past the end
- **WHEN** the wrapped span is inspected
- **THEN** it contains words from the corpus

### Requirement: a request larger than the whole corpus is clipped

`next_span(target_words)` MUST clip the returned span to the corpus length when `target_words` exceeds it. [REQUIRED]

#### Scenario: oversized request

- **GIVEN** a corpus with N words
- **WHEN** `next_span(2*N)` is called
- **THEN** the returned string has exactly N words

#### Scenario: request exactly at corpus length

- **GIVEN** a corpus with N words
- **WHEN** `next_span(N)` is called
- **THEN** the returned string has exactly N words

### Requirement: a non-positive word count raises ValueError

`next_span` MUST raise `ValueError` with a message matching `"target_words must be positive"` when `target_words <= 0`. [REQUIRED]

#### Scenario: zero request

- **GIVEN** any walk
- **WHEN** `next_span(0)` is called
- **THEN** `ValueError` matching `"target_words must be positive"` is raised

#### Scenario: negative request

- **GIVEN** any walk
- **WHEN** `next_span(-5)` is called
- **THEN** `ValueError` matching `"target_words must be positive"` is raised

#### Scenario: error message matches exact text

- **GIVEN** any walk
- **WHEN** `next_span(0)` is called
- **THEN** the error message contains `"target_words must be positive"`

### Requirement: load_corpus rejects a wordless snapshot

`load_corpus` MUST raise `ValueError` when the file contains no words, so a valid-JSON-but-empty snapshot fails loudly. No test exercises this path. [OBSERVED]

#### Scenario: snapshot with no words

- **GIVEN** a `.jsonl` file whose lines contain only empty `text` fields
- **WHEN** `load_corpus` is called on it
- **THEN** `ValueError` is raised naming the path

## Usage census

| Call site | Consumes |
|---|---|
| `eval/stream/generators/assoc.py` | `resolve(corpus_name)`, `load_corpus(path)`, `Corpus.walk(source)` |
| `scripts/show_stream.py` | `resolve`, `CORPUS_DIR_ENV`, `FETCH_SCRIPT`, `load_corpus`, `.corpus_id` |
| `tests/stream/test_corpus.py` | `resolve`, `load_corpus`, `Corpus.corpus_id`, `.words`, `.walk` |
| `tests/stream/test_assoc.py` | `load_corpus(resolve("corpus_fixture"))`, `.words` |
| `tests/stream/test_replay.py` | `load_corpus(fixture_path)`, `.corpus_id` |

## Leak list

None.

## Banned vocabulary

`CorpusWalk` (class name), `corpus_walk` (module name), `_words`, `_position`
