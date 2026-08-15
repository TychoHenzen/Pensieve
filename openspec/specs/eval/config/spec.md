## Purpose

Holds stream generation parameters as a frozen, hashable configuration object.

## Requirements

### Requirement: StreamConfig immutable params

`StreamConfig.params` MUST be a read-only mapping: assigning into it, or into any dict nested inside it, MUST raise `TypeError`. Mutation of the caller's original dict after construction MUST NOT be visible through `config.params`. [REQUIRED]

#### Scenario: top-level params assignment rejected

- **GIVEN** a constructed `StreamConfig`
- **WHEN** `config.params["key"] = value` is attempted
- **THEN** `TypeError` is raised

#### Scenario: nested params assignment rejected

- **GIVEN** a `StreamConfig` whose params contain a nested dict
- **WHEN** an item of that nested mapping is assigned
- **THEN** `TypeError` is raised

#### Scenario: caller's dict independence

- **GIVEN** a `StreamConfig` built from a caller-owned dict
- **WHEN** the caller mutates that original dict afterward
- **THEN** `config.params` and any hash computed from it are unaffected

### Requirement: StreamConfig params are canonically hashable

Everything reachable from `config.params` MUST remain encodable by `canonical_json` to a value stable under key-insertion-order changes. A fixed config/seed/version/corpus combination MUST reproduce the pinned digest `e60856050c6024f1fdb90d15563c7102393b80c47ff549892f06cdaf41c03f3b`. [REQUIRED]

#### Scenario: key order does not affect the hash

- **GIVEN** two configs built with the same params in different insertion order
- **WHEN** each is hashed via `stream_hash`
- **THEN** the resulting digests are equal

### Requirement: StreamConfig is frozen

`StreamConfig` MUST be a frozen dataclass: attribute assignment after construction MUST raise `dataclasses.FrozenInstanceError`. [REQUIRED]

#### Scenario: frozen attribute

- **GIVEN** a constructed `StreamConfig`
- **WHEN** `.generator` is assigned a new value
- **THEN** `dataclasses.FrozenInstanceError` is raised

## Usage census

| Call site | Consumes |
|---|---|
| `eval/stream/generator.py` | protocol parameter type |
| `eval/stream/registry.py` | `config.generator` string for lookup |
| `eval/stream/hashing.py` | `.generator`, `.params` for digest |
| `eval/stream/generators/assoc.py` | `config.params[...]` for all generator params |
| `eval/stream/generators/split_classify.py` | `config.params[...]` for all generator params |
| `eval/stream/generators/difficulty_mix.py` | `config.params[...]` for all generator params |

## Leak list

None.

## Banned vocabulary

`_freeze`
