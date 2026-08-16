## Purpose

Converts stream items to plain dicts and canonical JSON, and computes a content-addressing hash for identifying a stream by its generation parameters.

## Requirements

### Requirement: items_to_canonical_json is process-stable and hash-seed-independent

`items_to_canonical_json` MUST produce byte-identical output for the same `StreamConfig` and seed across a fresh subprocess and across different `PYTHONHASHSEED` values, and MUST differ when the seed differs. [REQUIRED]

#### Scenario: same seed, same output across processes

- **GIVEN** a config and seed
- **WHEN** `items_to_canonical_json` is called in the parent process and in a fresh subprocess with a different `PYTHONHASHSEED`
- **THEN** both produce the identical string

#### Scenario: different seed diverges

- **GIVEN** two different seeds with the same config
- **WHEN** `items_to_canonical_json` is called for each
- **THEN** the two strings differ

#### Scenario: same seed in same process repeats

- **GIVEN** a config and seed
- **WHEN** `items_to_canonical_json` is called twice in the same process
- **THEN** both produce identical output

### Requirement: stream_hash is a reproducible function of exactly five inputs

`stream_hash(config, seed, generator_version, render_version, corpus_id)` MUST return a 64-character lowercase hex string that is identical for identical inputs (insensitive to params key insertion order) and different when any one of the five inputs changes. [REQUIRED]

#### Scenario: identical inputs repeat

- **GIVEN** the same five arguments
- **WHEN** `stream_hash` is called twice
- **THEN** both calls return the same string

#### Scenario: any one input changing changes the hash

- **GIVEN** a baseline call to `stream_hash`
- **WHEN** exactly one of the five inputs is changed
- **THEN** the digest differs from the baseline

#### Scenario: digest format

- **GIVEN** any valid call to `stream_hash`
- **WHEN** the result is inspected
- **THEN** it is 64 characters long and every character is in `0123456789abcdef`

### Requirement: stream_hash reproduces a pinned digest

For `StreamConfig(generator="assoc", params={"pairs": 10, "distance": 100})`, `seed=0`, `generator_version="1.0"`, `render_version="1"`, `corpus_id="corpus-a"`, `stream_hash` MUST return exactly `e60856050c6024f1fdb90d15563c7102393b80c47ff549892f06cdaf41c03f3b`. [REQUIRED]

#### Scenario: golden value reproduces

- **GIVEN** the exact fixed inputs above
- **WHEN** `stream_hash` is called
- **THEN** it returns `e60856050c6024f1fdb90d15563c7102393b80c47ff549892f06cdaf41c03f3b`

#### Scenario: pinned digest is 64 hex characters

- **GIVEN** the pinned digest
- **WHEN** its format is checked
- **THEN** it is 64 characters of lowercase hex

### Requirement: canonical_json sorts keys and uses compact separators

`canonical_json` MUST serialize with sorted object keys and separators `(",", ":")` (no extra whitespace). [REQUIRED]

#### Scenario: key order does not affect output

- **GIVEN** two dicts with the same key-value pairs in different insertion order
- **WHEN** `canonical_json` is called on each
- **THEN** the outputs are identical

#### Scenario: compact separators used

- **GIVEN** `{"a": 1, "b": 2}`
- **WHEN** `canonical_json` is called
- **THEN** the output contains `","` and `":"` with no spaces around them

### Requirement: canonical_json handles frozen mapping proxies and tuples

`canonical_json` MUST successfully encode `MappingProxyType`-frozen params (including nested frozen mappings and tuples produced by `StreamConfig`) without raising. [REQUIRED]

#### Scenario: nested frozen structure encodes

- **GIVEN** a `StreamConfig` whose params contain a list of dicts (frozen into a tuple of mapping proxies)
- **WHEN** `canonical_json` is called via `stream_hash`
- **THEN** both calls succeed and return equal digests

#### Scenario: tuple encoded as array

- **GIVEN** a value containing a tuple `(1, 2, 3)`
- **WHEN** `canonical_json` is called
- **THEN** it succeeds and the tuple is encoded as a JSON array

### Requirement: canonical_json raises TypeError on sets

`canonical_json` MUST raise `TypeError` when given a `set`, because sets have no stable iteration order. [REQUIRED]

#### Scenario: a set cannot be canonicalized

- **GIVEN** the value `{1, 2, 3}`
- **WHEN** `canonical_json` is called on it
- **THEN** `TypeError` is raised

#### Scenario: nested set also rejected

- **GIVEN** the value `{"a": {1, 2}}`
- **WHEN** `canonical_json` is called on it
- **THEN** `TypeError` is raised

### Requirement: canonical_json forbids NaN and Infinity

`canonical_json` SHOULD pass `allow_nan=False` so non-finite floats raise rather than producing non-standard JSON. No test exercises this path. [OBSERVED]

#### Scenario: non-finite float rejected

- **GIVEN** a value containing `float("nan")`
- **WHEN** `canonical_json` is called on it
- **THEN** `ValueError` is raised

### Requirement: items_to_plain output shape is not externally pinned

`items_to_plain` SHOULD represent each `StreamItem` with `"event"` and `"truth"` top-level keys, with the event's class name under `"type"`. No test asserts on these field names directly - `test_replay.py` only compares the module's own output against itself. [OBSERVED]

#### Scenario: plain shape is self-consistent

- **GIVEN** a `StreamItem` sequence
- **WHEN** `items_to_plain` converts it
- **THEN** the result is only checked for round-trip stability, not against a fixed literal shape

## Usage census

| Call site | Consumes |
|---|---|
| `eval/stream/serialize.py:items_to_canonical_json` | calls `canonical_json` from `hashing.py` |
| `tests/stream/test_hashing.py` | `stream_hash`, `canonical_json`, pinned digest, `TypeError` on set |
| `tests/stream/test_assoc.py` | `stream_hash` with different token counter names |
| `tests/stream/test_replay.py` | `items_to_canonical_json`, `stream_hash` cross-process |
| `scripts/show_stream.py` | `stream_hash` |

## Leak list

None.

## Banned vocabulary

`_plain`, `_CanonicalEncoder`
