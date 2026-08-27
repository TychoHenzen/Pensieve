## Purpose

Generates association-recall evaluation streams: teaches key-value pairs and probes recall at controlled token distances, filling gaps with continuous corpus prose.

## Requirements

### Requirement: Generator identity and version

`AssocGenerator` MUST report `name = "assoc"` and `version = "6"`. [REQUIRED]

#### Scenario: version is pinned

- **GIVEN** an `AssocGenerator` instance
- **WHEN** `.name` and `.version` are read
- **THEN** they equal `"assoc"` and `"6"`

#### Scenario: name is assoc

- **GIVEN** an `AssocGenerator` instance
- **WHEN** `.name` is read
- **THEN** it equals `"assoc"`

#### Scenario: version is 6

- **GIVEN** an `AssocGenerator` instance
- **WHEN** `.version` is read
- **THEN** it equals `"6"`

### Requirement: Each pair is taught exactly once

Given `num_pairs`, the stream MUST emit one teaching `Observe` per pair carrying that pair's key and value in a payload with exactly the keys `{"key", "value"}`. [REQUIRED]

#### Scenario: teaching count matches pair count

- **GIVEN** `num_pairs=10`
- **WHEN** the stream is generated
- **THEN** exactly 10 teaching `Observe` events are emitted, each with a unique key

#### Scenario: teaching payload has exactly key and value

- **GIVEN** a generated stream
- **WHEN** any teaching `Observe` payload is inspected
- **THEN** its key set is exactly `{"key", "value"}`

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

#### Scenario: one probe per pair per distance

- **GIVEN** `num_pairs=5`, `recall_distances=[1, 10]`
- **WHEN** the stream is generated
- **THEN** exactly 10 probes are emitted

### Requirement: Positions are contiguous from zero

Positions across the whole stream MUST be exactly `range(len(stream))`, in order, with no gaps or repeats. [REQUIRED]

#### Scenario: no gaps in positions

- **GIVEN** a generated stream of length N
- **WHEN** positions are collected
- **THEN** they equal `list(range(N))`

#### Scenario: no repeated positions

- **GIVEN** a generated stream
- **WHEN** positions are collected
- **THEN** every position is unique

### Requirement: Keys are distinct vocabulary words

Taught keys MUST be pairwise distinct. Every key, value, probe query, and truth answer MUST be a member of `vocab.VOCAB`. [REQUIRED]

#### Scenario: unique keys from vocabulary

- **GIVEN** a generated stream
- **WHEN** all taught keys are collected
- **THEN** they are pairwise distinct and each is in `VOCAB`

#### Scenario: keys are pairwise distinct

- **GIVEN** a generated stream
- **WHEN** all taught keys are collected
- **THEN** they are pairwise distinct

#### Scenario: keys are in VOCAB

- **GIVEN** a generated stream
- **WHEN** all taught keys are collected
- **THEN** each is in `VOCAB`

#### Scenario: values are in VOCAB

- **GIVEN** a generated stream
- **WHEN** all taught values are collected
- **THEN** each is in `VOCAB`

### Requirement: Filler occupies non-teaching, non-probe positions with continuous corpus prose

Filler `Observe` events MUST carry `payload={"text": ...}` drawn from the configured corpus. Consecutive filler spans, joined in stream order, MUST form a single verbatim excerpt of that corpus. [REQUIRED]

#### Scenario: filler payload has text key

- **GIVEN** a generated stream
- **WHEN** any filler `Observe` payload is inspected
- **THEN** its key set is `{"text"}`

#### Scenario: filler continues one document

- **GIVEN** a generated stream
- **WHEN** all filler spans are joined in order
- **THEN** the joined text is a substring of the corpus text

#### Scenario: filler drawn from configured corpus

- **GIVEN** two streams with different corpora
- **WHEN** filler text is compared
- **THEN** it differs (each matches its own corpus)

### Requirement: Total stream length is a fixed function of config

Stream length MUST equal `num_pairs * (1 + len(recall_distances)) + round(filler_density * num_pairs * (1 + len(recall_distances)))`. [REQUIRED]

#### Scenario: pinned stream length

- **GIVEN** `num_pairs=10`, `recall_distances=[1, 10, 100]`, `filler_density=5.0`
- **WHEN** the stream is generated
- **THEN** it yields exactly 240 items

#### Scenario: length formula holds for different configs

- **GIVEN** `num_pairs=5`, `recall_distances=[1]`, `filler_density=1.0`
- **WHEN** the stream is generated
- **THEN** it yields exactly `5 * 2 + round(1.0 * 5 * 2) = 20` items

### Requirement: Unschedulable configurations raise ValueError

`generate` MUST raise `ValueError` when the timeline cannot fit every pair's probes at their requested distances. [REQUIRED]

#### Scenario: impossible schedule rejected

- **GIVEN** a config where timeline length is too short for the requested distances
- **WHEN** `generate` is called
- **THEN** `ValueError` is raised

#### Scenario: error message explains the conflict

- **GIVEN** an unschedulable config
- **WHEN** `generate` raises `ValueError`
- **THEN** the message indicates a scheduling failure

### Requirement: Config validation

`generate` MUST raise `ValueError` for: distance `< 1`, distance exceeding `max_distance`, negative `filler_density`, or missing `corpus` param. [REQUIRED]

#### Scenario: distance below 1

- **GIVEN** `recall_distances=[0]`
- **WHEN** `generate` is called
- **THEN** `ValueError("recall distance must be at least 1, got 0")` is raised

#### Scenario: negative filler_density

- **GIVEN** `filler_density=-1.0`
- **WHEN** `generate` is called
- **THEN** `ValueError` is raised

#### Scenario: missing corpus param

- **GIVEN** a config with no `corpus` in params
- **WHEN** `generate` is called
- **THEN** `ValueError` or `KeyError` is raised

#### Scenario: distance exceeding max_distance

- **GIVEN** a distance larger than the allowed maximum
- **WHEN** `generate` is called
- **THEN** `ValueError` is raised

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

#### Scenario: counter configured, token distance rises with event distance

- **GIVEN** a config with `token_counter` and probes at increasing distances
- **WHEN** the stream is generated
- **THEN** `token_distance` values rise monotonically

#### Scenario: counter configured, all probes have integer token_distance

- **GIVEN** a config with `token_counter`
- **WHEN** the stream is generated
- **THEN** every probe's `token_distance` is an `int`, not `None`

### Requirement: Unknown token_counter name raises ValueError

An unrecognized `token_counter` name MUST raise `ValueError` listing known counter names. [REQUIRED]

#### Scenario: bad counter name

- **GIVEN** `token_counter="not-a-real-counter"`
- **WHEN** `generate` is called
- **THEN** `ValueError` mentioning `"regex-whitespace-v1"` is raised

#### Scenario: error lists known counters

- **GIVEN** an unrecognized `token_counter`
- **WHEN** `generate` raises `ValueError`
- **THEN** the message contains at least one known counter name

### Requirement: No payload or query text carries a role prefix

No text in any payload or query MUST start with `"key-"`, `"value-"`, or `"filler-"`. [REQUIRED]

#### Scenario: no role prefix leak

- **GIVEN** a generated stream
- **WHEN** all rendered text is inspected
- **THEN** none starts with `"key-"`, `"value-"`, or `"filler-"`

#### Scenario: no key- prefix

- **GIVEN** a generated stream
- **WHEN** all rendered text is inspected
- **THEN** none starts with `"key-"`

#### Scenario: no value- prefix

- **GIVEN** a generated stream
- **WHEN** all rendered text is inspected
- **THEN** none starts with `"value-"`

### Requirement: chance_rate equals 1/len(VOCAB)

`chance_rate(config)` MUST return `1/len(vocab.VOCAB)` regardless of config params. [REQUIRED]

#### Scenario: chance rate

- **GIVEN** any valid assoc config
- **WHEN** `chance_rate(config)` is called
- **THEN** it returns `1 / len(VOCAB)`

#### Scenario: chance rate independent of config params

- **GIVEN** two configs with different `num_pairs`
- **WHEN** `chance_rate` is called on each
- **THEN** both return the same value

### Requirement: Deterministic in (config, seed) and diverges on different seeds

The same `(config, seed)` MUST reproduce an identical stream. A different `seed` MUST produce a different stream. [REQUIRED]

#### Scenario: replay determinism

- **GIVEN** a config and `seed=42`
- **WHEN** `generate` is called twice with the same arguments
- **THEN** both streams are identical

#### Scenario: different seeds diverge

- **GIVEN** a config with `seed=42` and `seed=99`
- **WHEN** `generate` is called for each
- **THEN** the two streams differ

#### Scenario: determinism covers event fields

- **GIVEN** a config and `seed=42`
- **WHEN** `generate` is called twice
- **THEN** every event field (payload, query, position, truth) is identical

### Requirement: Interleaving puts other pairs' events between a teaching and its probe

The scheduler MUST allow other pairs' teachings and probes to fall between one pair's teaching and its own probe. [OBSERVED - direct test exists but no non-test caller depends on it]

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
