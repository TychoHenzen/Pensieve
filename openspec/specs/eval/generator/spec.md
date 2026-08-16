## Purpose

Defines the protocol every stream generator satisfies and the registry that resolves a config's generator name to a concrete implementation.

## Requirements

### Requirement: Independent named random sources

`derive(seed, name)` MUST return a `random.Random` whose draw sequence depends only on `(seed, name)`, is stable across processes and `PYTHONHASHSEED` values, and is unaffected by draws taken from a source built with a different `name` at the same `seed`. [REQUIRED]

#### Scenario: same seed and name repeat

- **GIVEN** a fixed `seed` and `name`
- **WHEN** `derive(seed, name)` is called twice and each result's `.random()` is drawn 5 times
- **THEN** both draw sequences are identical

#### Scenario: different names diverge

- **GIVEN** the same `seed` but different `name` values
- **WHEN** `derive` is called for each
- **THEN** the first draws differ

#### Scenario: known seed produces a known first draw

- **GIVEN** `seed=0`, `name="keys"`
- **WHEN** `derive(0, "keys").random()` is called once
- **THEN** the result equals `0.9233467507638282`

### Requirement: StreamGenerator protocol shape

Every stream generator MUST expose a `name: str` property, a `version: str` property, and a `generate(config: StreamConfig, seed: int) -> Iterator[StreamItem]` method. The protocol MUST be runtime-checkable via `isinstance`. [REQUIRED]

#### Scenario: protocol conformance

- **GIVEN** a class with `name`, `version`, and `generate` matching the protocol
- **WHEN** `isinstance(instance, StreamGenerator)` is checked
- **THEN** it returns `True`

#### Scenario: protocol is runtime-checkable

- **GIVEN** the `StreamGenerator` protocol
- **WHEN** `isinstance` is called with a non-conforming object
- **THEN** it returns `False` without raising

#### Scenario: protocol requires generate method

- **GIVEN** a class with `name` and `version` but no `generate`
- **WHEN** `isinstance(instance, StreamGenerator)` is checked
- **THEN** it returns `False`

### Requirement: Every generator exposes chance_rate

Each registered generator MUST expose `chance_rate(self, config: StreamConfig) -> float` giving the accuracy a uniform-random answerer achieves over its legal answer set. This method is not declared in the `StreamGenerator` Protocol text but is called by `scripts/show_stream.py` and tested by `tests/stream/test_chance.py`. [REQUIRED]

#### Scenario: chance_rate callable on every registered generator

- **GIVEN** any generator class resolved from `REGISTRY`
- **WHEN** `generator.chance_rate(config)` is called with an appropriate config
- **THEN** it returns a float in `(0, 1]`

#### Scenario: chance_rate returns a float

- **GIVEN** any registered generator
- **WHEN** `chance_rate(config)` is called
- **THEN** `isinstance(result, float)` is `True`

### Requirement: Registry resolves a config's generator name

`build(config, seed)` MUST look up `config.generator` in `REGISTRY` and return that generator's stream. An unregistered name MUST raise `ValueError` naming the unknown value and every registered name. [REQUIRED]

#### Scenario: unknown generator name rejected

- **GIVEN** `StreamConfig(generator="does-not-exist")`
- **WHEN** `build(config, seed=0)` is called
- **THEN** `ValueError` is raised whose message contains `"does-not-exist"` and every name in `REGISTRY`

#### Scenario: same config and seed replay identically

- **GIVEN** an assoc config and `seed=0`
- **WHEN** `build` is called twice
- **THEN** the two resulting sequences are equal element-for-element

#### Scenario: error message names all registered generators

- **GIVEN** `StreamConfig(generator="bogus")`
- **WHEN** `build` raises `ValueError`
- **THEN** the message contains `"assoc"`, `"split-classify"`, and `"difficulty-mix"`

### Requirement: Three generators are registered under fixed names

`REGISTRY` MUST contain the keys `"assoc"`, `"split-classify"`, `"difficulty-mix"`. [REQUIRED]

#### Scenario: registry contents

- **GIVEN** the `REGISTRY` dict
- **WHEN** its keys are inspected
- **THEN** `{"assoc", "split-classify", "difficulty-mix"}` is a subset of the key set

#### Scenario: assoc is registered

- **GIVEN** the `REGISTRY` dict
- **WHEN** `"assoc"` is looked up
- **THEN** it is present

#### Scenario: split-classify is registered

- **GIVEN** the `REGISTRY` dict
- **WHEN** `"split-classify"` is looked up
- **THEN** it is present

#### Scenario: difficulty-mix is registered

- **GIVEN** the `REGISTRY` dict
- **WHEN** `"difficulty-mix"` is looked up
- **THEN** it is present

### Requirement: Cross-process and cross-hash-seed replay determinism

The same `StreamConfig` and seed MUST regenerate an identical event/truth sequence and an identical `stream_hash` in the same process, in a fresh subprocess, and under different `PYTHONHASHSEED` values. [REQUIRED]

#### Scenario: subprocess replay matches

- **GIVEN** a config and seed used to generate a stream in the current process
- **WHEN** the identical config and seed generate a stream in a fresh subprocess with a different `PYTHONHASHSEED`
- **THEN** the canonical serialized sequence and stream hash are identical

#### Scenario: same-process replay matches

- **GIVEN** a config and seed
- **WHEN** `build` is called twice in the same process
- **THEN** the event/truth sequences are identical

## Usage census

| Call site | Consumes |
|---|---|
| `eval/stream/registry.py` | `StreamGenerator` for type annotation, `derive` not imported directly |
| `eval/stream/generators/assoc.py` | `derive(seed, "keys")`, `derive(seed, "values")`, etc. |
| `eval/stream/generators/split_classify.py` | `derive(seed, "example")`, `derive(seed, "probe")` |
| `eval/stream/generators/difficulty_mix.py` | `derive(seed, ...)` for each random source |
| `scripts/show_stream.py` | `REGISTRY[name]()`, `.version`, `.chance_rate(config)` |
| `tests/stream/test_generator.py` | `derive` pinned values, protocol `isinstance` check |
| `tests/stream/test_registry.py` | `build`, `REGISTRY` membership |

## Leak list

None.

## Banned vocabulary

None. All public identifiers are boundary.
