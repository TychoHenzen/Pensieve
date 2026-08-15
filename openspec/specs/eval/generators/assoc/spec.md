## Purpose

Generates association-recall evaluation streams: teaches key-value pairs and probes recall at controlled token distances, filling gaps with continuous corpus prose.

## Requirements

### Requirement: Generator identity and version

`AssocGenerator` MUST report `name = "assoc"` and `version = "6"`. [REQUIRED]

#### Scenario: version is pinned

- **GIVEN** an `AssocGenerator` instance
- **WHEN** `.name` and `.version` are read
- **THEN** they equal `"assoc"` and `"6"`

### Requirement: Each pair is taught exactly once

Given `num_pairs`, the stream MUST emit one teaching `Observe` per pair carrying that pair's key and value in a payload with exactly the keys `{"key", "value"}`. [REQUIRED]

#### Scenario: teaching count matches pair count

- **GIVEN** `num_pairs=10`
- **WHEN** the stream is generated
- **THEN** exactly 10 teaching `Observe` events are emitted, each with a unique key

### Requirement: Probes at exactly the requested distances

The stream MUST emit exactly one `Probe` per pair per requested recall distance, positioned at exactly `teaching_position + distance`. Each probe's truth answer MUST equal the taught value. [REQUIRED]

#### Scenario: probe distance matches configuration

- **GIVEN** `recall_distances=[1, 2, 5]` and any valid config
- **WHEN** the stream is generated
- **THEN** the set of `(probe.position - probe.teaching_position)` values equals `{1, 2, 5}`

#### Scenario: probe truth matches the taught value

- **GIVEN** a generated stream
- **WHEN** a probe's `teaching_position` is followed to its teaching event
- **THEN** `probe.truth.answer` equals that teaching's `payload["value"]`

### Requirement: Positions are contiguous from zero

Positions across the whole stream MUST be exactly `range(len(stream))`, in order, with no gaps or repeats. [REQUIRED]

#### Scenario: no gaps in positions

- **GIVEN** a generated stream of length N
- **WHEN** positions are collected
- **THEN** they equal `list(range(N))`

### Requirement: Keys are distinct vocabulary words

Taught keys MUST be pairwise distinct. Every key, value, probe query, and truth answer MUST be a member of `vocab.VOCAB`. [REQUIRED]

#### Scenario: unique keys from vocabulary

- **GIVEN** a generated stream
- **WHEN** all taught keys are collected
- **THEN** they are pairwise distinct and each is in `VOCAB`

### Requirement: Filler occupies non-teaching, non-probe positions with continuous corpus prose

Filler `Observe` events MUST carry `payload={"text": ...}` drawn from the configured corpus. Consecutive filler spans, joined in stream order, MUST form a single verbatim excerpt of that corpus. [REQUIRED]

#### Scenario: filler continues one document

- **GIVEN** a generated stream
- **WHEN** all filler spans are joined in order
- **THEN** the joined text is a substring of the corpus text

### Requirement: Total stream length is a fixed function of config

Stream length MUST equal `num_pairs * (1 + len(recall_distances)) + round(filler_density * num_pairs * (1 + len(recall_distances)))`. [REQUIRED]

#### Scenario: pinned stream length

- **GIVEN** `num_pairs=10`, `recall_distances=[1, 10, 100]`, `filler_density=5.0`
- **WHEN** the stream is generated
- **THEN** it yields exactly 240 items

### Requirement: Unschedulable configurations raise ValueError

`generate` MUST raise `ValueError` when the timeline cannot fit every pair's probes at their requested distances. [REQUIRED]

#### Scenario: impossible schedule rejected

- **GIVEN** a config where timeline length is too short for the requested distances
- **WHEN** `generate` is called
- **THEN** `ValueError` is raised

### Requirement: Config validation

`generate` MUST raise `ValueError` for: distance `< 1`, distance exceeding `max_distance`, negative `filler_density`, or missing `corpus` param. [REQUIRED]

#### Scenario: distance below 1

- **GIVEN** `recall_distances=[0]`
- **WHEN** `generate` is called
- **THEN** `ValueError("recall distance must be at least 1, got 0")` is raised

### Requirement: token_distance absent without counter, computed with counter

When `config.params` has no `token_counter`, every probe's `token_distance` MUST be `None`. When configured, `token_distance` MUST be `0` with no intervening text and MUST rise monotonically with distance. [REQUIRED]

#### Scenario: no counter configured

- **GIVEN** a config without `token_counter`
- **WHEN** the stream is generated
- **THEN** every probe's `token_distance` is `None`

#### Scenario: counter configured, adjacent probe

- **GIVEN** a config with `token_counter="regex-whitespace-v1"` and a probe at distance 1 from its teaching with no filler between
- **WHEN** the stream is generated
- **THEN** that probe's `token_distance` is `0`

### Requirement: Unknown token_counter name raises ValueError

An unrecognized `token_counter` name MUST raise `ValueError` listing known counter names. [REQUIRED]

#### Scenario: bad counter name

- **GIVEN** `token_counter="not-a-real-counter"`
- **WHEN** `generate` is called
- **THEN** `ValueError` mentioning `"regex-whitespace-v1"` is raised

### Requirement: No payload or query text carries a role prefix

No text in any payload or query MUST start with `"key-"`, `"value-"`, or `"filler-"`. [REQUIRED]

#### Scenario: no role prefix leak

- **GIVEN** a generated stream
- **WHEN** all rendered text is inspected
- **THEN** none starts with `"key-"`, `"value-"`, or `"filler-"`

### Requirement: chance_rate equals 1/len(VOCAB)

`chance_rate(config)` MUST return `1/len(vocab.VOCAB)` regardless of config params. [REQUIRED]

#### Scenario: chance rate

- **GIVEN** any valid assoc config
- **WHEN** `chance_rate(config)` is called
- **THEN** it returns `1 / len(VOCAB)`

### Requirement: Deterministic in (config, seed) and diverges on different seeds

The same `(config, seed)` MUST reproduce an identical stream. A different `seed` MUST produce a different stream. [REQUIRED]

#### Scenario: replay determinism

- **GIVEN** a config and `seed=42`
- **WHEN** `generate` is called twice with the same arguments
- **THEN** both streams are identical

### Requirement: Interleaving puts other pairs' events between a teaching and its probe

The scheduler SHOULD allow other pairs' teachings and probes to fall between one pair's teaching and its own probe. [OBSERVED - direct test exists but no non-test caller depends on it]

#### Scenario: interleaving occurs

- **GIVEN** a multi-pair stream with sufficient distance
- **WHEN** events between a pair's teaching and its probe are inspected
- **THEN** some belong to other pairs

## Usage census

| Call site | Consumes |
|---|---|
| `eval/stream/registry.py` | imports `AssocGenerator`, registers as `"assoc"` |
| `tests/stream/test_assoc.py` | all generator behavior |
| `tests/stream/test_chance.py` | `chance_rate` measurement |
| `tests/stream/test_replay.py` | cross-process determinism |

## Leak list

None.

## Banned vocabulary

`_validate`, `_valid_candidates`, `_schedule`, `_fill_token_distances`, `FILLER_SPAN_WORDS`, `teaching_positions`, `occupied`, `keys_source`, `values_source`, `order_source`, `filler_source`, `events_by_position`
