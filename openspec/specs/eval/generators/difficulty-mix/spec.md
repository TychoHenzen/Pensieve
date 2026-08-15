## Purpose

Generates arithmetic-chain probes of mixed difficulty for measuring whether a system allocates more compute to harder items, interleaved with idle filler.

## Requirements

### Requirement: Generator identity and version

`DifficultyMixGenerator` MUST report `name = "difficulty-mix"`. `.version` MUST be a non-empty string. [REQUIRED]

#### Scenario: name and version present

- **GIVEN** a `DifficultyMixGenerator` instance
- **WHEN** `.name` and `.version` are read
- **THEN** `name` equals `"difficulty-mix"` and `version` is a non-empty string

### Requirement: Every configured difficulty level appears at least once

Given `difficulty_levels`, every listed level MUST appear as `truth.difficulty` on at least one probe. [REQUIRED]

#### Scenario: all levels represented

- **GIVEN** `difficulty_levels=[1, 2, 3]`
- **WHEN** the stream is generated
- **THEN** the set of `truth.difficulty` values across all probes is a superset of `{1, 2, 3}`

### Requirement: Truth answer matches independently evaluated chain

Each probe's `truth.answer` MUST equal the result of independently reparsing the `query` text as a start value plus a sequence of add/subtract steps, taken modulo `CHAIN_MODULUS` (1000). [REQUIRED]

#### Scenario: chain replay agrees

- **GIVEN** a probe with `query = "Start at 42. Add 7. Subtract 3. What is the result modulo 1000?"`
- **WHEN** the chain is evaluated independently: `(42 + 7 - 3) % 1000 = 46`
- **THEN** `truth.answer == 46`

### Requirement: Difficulty equals the number of arithmetic steps

`truth.difficulty` MUST equal the number of add/subtract steps parseable from the query. [REQUIRED]

#### Scenario: difficulty matches step count

- **GIVEN** a probe whose query contains 3 steps
- **WHEN** `truth.difficulty` is read
- **THEN** it equals `3`

### Requirement: Answer space is bounded by CHAIN_MODULUS

Every probe's `truth.answer` MUST be in `[0, CHAIN_MODULUS)`. [REQUIRED]

#### Scenario: answer is bounded

- **GIVEN** any probe in a difficulty-mix stream
- **WHEN** `truth.answer` is read
- **THEN** `0 <= truth.answer < 1000`

### Requirement: Rendered query never leaks difficulty or level labels

No text produced by the generator for text-carrying events SHALL contain the case-insensitive substrings `"difficulty"` or `"level"`. [REQUIRED]

#### Scenario: no difficulty leak

- **GIVEN** a difficulty-mix stream
- **WHEN** every text-carrying event is rendered
- **THEN** no rendered string contains `"difficulty"` or `"level"` (case-insensitive)

### Requirement: Rendered query follows a parseable grammar

The query MUST start with `f"Start at {start}."`, followed by one segment per step matching `^(Add|Subtract) (\d+)\.$`, and end with `f" What is the result modulo {CHAIN_MODULUS}?"`. [REQUIRED]

#### Scenario: query grammar

- **GIVEN** a probe in a difficulty-mix stream
- **WHEN** its query is parsed
- **THEN** it matches the specified grammar exactly

### Requirement: Non-probe positions are Idle events with no truth

Every position that is not a probe MUST be an `Idle` event with `truth=None`. [REQUIRED]

#### Scenario: idle filler

- **GIVEN** a generated stream
- **WHEN** non-probe positions are inspected
- **THEN** every one is an `Idle` event with `truth is None`

### Requirement: Probe count equals num_items exactly

The total number of probes in the stream MUST equal `num_items`. [REQUIRED]

#### Scenario: exact probe count

- **GIVEN** `num_items=20`
- **WHEN** the stream is generated
- **THEN** exactly 20 probes are emitted

### Requirement: Positions are contiguous from zero

Positions MUST be `range(len(stream))` with no gaps. [REQUIRED]

#### Scenario: contiguous positions

- **GIVEN** a generated stream of length N
- **WHEN** positions are collected
- **THEN** they equal `list(range(N))`

### Requirement: Validates num_items, difficulty_levels, and probe_rate

`generate` MUST raise `ValueError` when `num_items <= 0`, `difficulty_levels` is empty, any level `< 1`, `num_items < len(difficulty_levels)`, or `probe_rate` outside `(0, 1]`. [REQUIRED]

#### Scenario: empty difficulty_levels rejected

- **GIVEN** `difficulty_levels=[]`
- **WHEN** `generate` is called
- **THEN** `ValueError("difficulty_levels must not be empty")` is raised

#### Scenario: probe_rate out of range

- **GIVEN** `probe_rate=0`
- **WHEN** `generate` is called
- **THEN** `ValueError` is raised

### Requirement: chance_rate equals 1/CHAIN_MODULUS

`chance_rate(config)` MUST return `1/CHAIN_MODULUS` regardless of config params. [REQUIRED]

#### Scenario: chance rate

- **GIVEN** any valid difficulty-mix config
- **WHEN** `chance_rate(config)` is called
- **THEN** it returns `0.001`

### Requirement: Deterministic in (config, seed) and diverges on different seeds

The same `(config, seed)` MUST reproduce an identical stream. A different `seed` MUST produce a different stream. [REQUIRED]

#### Scenario: replay determinism

- **GIVEN** a config and `seed=42`
- **WHEN** `generate` is called twice
- **THEN** both streams are identical

## Usage census

| Call site | Consumes |
|---|---|
| `eval/stream/registry.py` | imports `DifficultyMixGenerator`, registers as `"difficulty-mix"` |
| `tests/stream/test_difficulty_mix.py` | all generator behavior, imports `CHAIN_MODULUS` |
| `tests/stream/test_chance.py` | `chance_rate` measurement |

## Leak list

None.

## Banned vocabulary

`_assign_difficulties`, `_build_chain`, `_render_query`, `_OPERAND_MIN`, `_OPERAND_MAX`, `difficulty_by_position`, `probe_positions`, `chain_source`, `idle_source`, `difficulty_source`
