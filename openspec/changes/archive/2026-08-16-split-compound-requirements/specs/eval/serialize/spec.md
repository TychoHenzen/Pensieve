## MODIFIED Requirements

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
