## Purpose

Provides a fixed pool of common English words for generating unambiguous key-value associations in evaluation streams.

## Requirements

### Requirement: VOCAB pool size, uniqueness, order, and shape

`VOCAB` MUST contain at least 256 entries, MUST contain no duplicates, MUST be presented in sorted order, and every entry MUST be lowercase alphabetic. `VOCAB` MUST be a tuple, not a list. [REQUIRED]

#### Scenario: vocabulary size and shape

- **GIVEN** the `VOCAB` tuple
- **WHEN** it is inspected
- **THEN** it has at least 256 entries, no duplicates, sorted order, and every word is lowercase alphabetic

### Requirement: sample determinism, distinctness, and bounds checking

`sample(source, n)` MUST be reproducible for a fixed `random.Random` state, MUST diverge across different seeds, MUST return `n` pairwise-distinct words, and MUST raise `ValueError` when `n` exceeds `len(VOCAB)` or is negative. [REQUIRED]

#### Scenario: reproducible draw

- **GIVEN** two `random.Random(0)` sources
- **WHEN** `sample` is called on each with the same `n`
- **THEN** the two results are equal

#### Scenario: oversized draw rejected

- **GIVEN** `n = len(VOCAB) + 1`
- **WHEN** `sample` is called
- **THEN** `ValueError` is raised

#### Scenario: negative draw rejected

- **GIVEN** `n = -1`
- **WHEN** `sample` is called
- **THEN** `ValueError("n must be non-negative, got -1")` is raised

### Requirement: chance_rate formula and validation

`chance_rate(n_choices)` MUST return `1.0 / n_choices` and MUST raise `ValueError` for `n_choices <= 0`. [REQUIRED]

#### Scenario: chance rate for the full vocabulary

- **GIVEN** `n_choices = len(VOCAB)`
- **WHEN** `chance_rate` is called
- **THEN** it returns `1 / len(VOCAB)`

#### Scenario: non-positive input rejected

- **GIVEN** `n_choices = 0`
- **WHEN** `chance_rate` is called
- **THEN** `ValueError` is raised

### Requirement: VOCAB_VERSION tracks vocabulary changes

`VOCAB_VERSION` SHOULD change whenever `VOCAB` changes, so a stored run can identify which word set produced it. No importer of `VOCAB_VERSION` was found in the codebase. [OBSERVED]

#### Scenario: vocabulary content changes

- **GIVEN** a future edit that adds or removes words from `VOCAB`
- **WHEN** the module is released
- **THEN** `VOCAB_VERSION` is expected to change

## Usage census

| Call site | Consumes |
|---|---|
| `eval/stream/generators/assoc.py` | `VOCAB`, `sample`, `chance_rate` |
| `tests/stream/test_vocab.py` | `VOCAB` shape, `sample` determinism, `chance_rate` formula |
| `tests/stream/test_chance.py` | `VOCAB` for answer-set size |

## Leak list

None.

## Banned vocabulary

None. All public identifiers are boundary.
