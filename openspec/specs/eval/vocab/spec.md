## Purpose

Provides a fixed pool of common English words for generating unambiguous key-value associations in evaluation streams.

## Requirements

### Requirement: VOCAB pool size, uniqueness, order, and shape

`VOCAB` MUST contain at least 256 entries, MUST contain no duplicates, MUST be presented in sorted order, and every entry MUST be lowercase alphabetic. `VOCAB` MUST be a tuple, not a list. [REQUIRED]

#### Scenario: vocabulary size and shape

- **GIVEN** the `VOCAB` tuple
- **WHEN** it is inspected
- **THEN** it has at least 256 entries, no duplicates, sorted order, and every word is lowercase alphabetic

#### Scenario: vocabulary has at least 256 entries

- **GIVEN** the `VOCAB` tuple
- **WHEN** its length is checked
- **THEN** it is at least 256

#### Scenario: vocabulary has no duplicates

- **GIVEN** the `VOCAB` tuple
- **WHEN** it is converted to a set
- **THEN** the set has the same length as the tuple

#### Scenario: vocabulary is sorted

- **GIVEN** the `VOCAB` tuple
- **WHEN** it is compared to `tuple(sorted(VOCAB))`
- **THEN** they are equal

#### Scenario: every entry is lowercase alphabetic

- **GIVEN** the `VOCAB` tuple
- **WHEN** every entry is checked against `str.isalpha()` and `str.islower()`
- **THEN** all return `True`

#### Scenario: VOCAB is a tuple

- **GIVEN** the `VOCAB` object
- **WHEN** its type is checked
- **THEN** it is `tuple`

#### Scenario: no entry is an empty string

- **GIVEN** the `VOCAB` tuple
- **WHEN** every entry is checked
- **THEN** none is empty

### Requirement: sample determinism, distinctness, and bounds checking

`sample(source, n)` MUST be reproducible for a fixed `random.Random` state, MUST diverge across different seeds, MUST return `n` pairwise-distinct words, and MUST raise `ValueError` when `n` exceeds `len(VOCAB)` or is negative. [REQUIRED]

#### Scenario: reproducible draw

- **GIVEN** two `random.Random(0)` sources
- **WHEN** `sample` is called on each with the same `n`
- **THEN** the two results are equal

#### Scenario: different seeds diverge

- **GIVEN** `random.Random(0)` and `random.Random(1)`
- **WHEN** `sample` is called with the same `n`
- **THEN** the two results differ

#### Scenario: n distinct words returned

- **GIVEN** `random.Random(0)` and `n=10`
- **WHEN** `sample` is called
- **THEN** the result has 10 pairwise-distinct entries

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

#### Scenario: negative input rejected

- **GIVEN** `n_choices = -1`
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
