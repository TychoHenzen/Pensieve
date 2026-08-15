## Purpose

Generates sequential classification tasks for measuring forward and backward transfer: teaches labeled examples task by task, then probes recall of all tasks seen so far.

## Requirements

### Requirement: Generator identity and version

`SplitClassifyGenerator` MUST report `name = "split-classify"` and `version = "2"`. [REQUIRED]

#### Scenario: version is pinned

- **GIVEN** a `SplitClassifyGenerator` instance
- **WHEN** `.name` and `.version` are read
- **THEN** they equal `"split-classify"` and `"2"`

### Requirement: Sequential task blocks with examples then probes

`generate` MUST teach `examples_per_task` examples per task in `num_tasks` sequential tasks. After each task's examples, it MUST emit `probes_per_task` probes for every task index from 0 through the current task inclusive. [REQUIRED]

#### Scenario: probes after task j cover every task up to j

- **GIVEN** `num_tasks=4`, `probes_per_task=2`
- **WHEN** the stream is generated
- **THEN** the probe task-id set in the block following task `j`'s examples equals `{"task0", ..., f"task{j}"}`

#### Scenario: probe count per checkpoint

- **GIVEN** `num_tasks=3`, `probes_per_task=2`
- **WHEN** the stream is generated
- **THEN** the probe block after task 2 contains `2 * 3 = 6` probes

### Requirement: Boundaries between tasks but not after the last

A `Boundary` with `kind=BoundaryKind.TASK_SWITCH` MUST separate consecutive task blocks. No `Boundary` MUST appear after the final task. [REQUIRED]

#### Scenario: boundaries land between tasks

- **GIVEN** `num_tasks=3`
- **WHEN** the stream is generated
- **THEN** exactly 2 `Boundary` events appear, each with `kind=BoundaryKind.TASK_SWITCH`

#### Scenario: no boundary after the final task

- **GIVEN** any valid config
- **WHEN** the stream is generated
- **THEN** the last event is not a `Boundary`

### Requirement: Boundary hidden_from_subject reflects config

`hidden_from_subject` on every emitted `Boundary` MUST equal the config's `hidden_from_subject` param (default `False`). [REQUIRED]

#### Scenario: hidden boundaries

- **GIVEN** `hidden_from_subject=True` in config
- **WHEN** the stream is generated
- **THEN** every `Boundary` has `hidden_from_subject=True`

### Requirement: Observe payload has fixed shape

Every `Observe.payload` MUST have exactly the keys `{"features", "label", "source"}`, with `features` a non-empty list, `label` a string, and `source` always `None`. [REQUIRED]

#### Scenario: payload shape

- **GIVEN** a generated stream
- **WHEN** any `Observe` payload is inspected
- **THEN** its key set is `{"features", "label", "source"}` and `source is None`

### Requirement: Probe truth answer is a valid label for the probed task

`truth.answer` for a probe of task index `t` MUST start with `f"task{t}-class"`. [REQUIRED]

#### Scenario: answer matches task

- **GIVEN** a probe with `task_id="task2"`
- **WHEN** `truth.answer` is read
- **THEN** it starts with `"task2-class"`

### Requirement: Probe query and example use the same feature notation

A probe's `query` MUST start with `"features=("` and the rendering of a taught example MUST agree on feature notation. Neither MUST contain 3 or more consecutive digits after a decimal point. [REQUIRED]

#### Scenario: feature precision alignment

- **GIVEN** a probe query and the rendered example with the same features
- **WHEN** both are inspected
- **THEN** neither contains a float with 3 or more decimal digits

### Requirement: Positions are contiguous from zero

Positions MUST be `range(len(stream))` with no gaps. [REQUIRED]

#### Scenario: contiguous positions

- **GIVEN** a generated stream of length N
- **WHEN** positions are collected
- **THEN** they equal `list(range(N))`

### Requirement: Validates all four required params as >= 1

`generate` MUST raise `ValueError` when `num_tasks`, `classes_per_task`, `examples_per_task`, or `probes_per_task` is less than 1. [REQUIRED]

#### Scenario: zero tasks rejected

- **GIVEN** `num_tasks=0`
- **WHEN** `generate` is called
- **THEN** `ValueError("num_tasks must be at least 1, got 0")` is raised

### Requirement: chance_rate equals 1/classes_per_task, computed from that single param alone

`chance_rate(config)` MUST return `1/config.params["classes_per_task"]` and MUST work even when no other param is present in the config. [REQUIRED]

#### Scenario: chance rate with minimal config

- **GIVEN** `StreamConfig(generator="split-classify", params={"classes_per_task": 5})`
- **WHEN** `chance_rate(config)` is called
- **THEN** it returns `0.2` without raising

### Requirement: Deterministic in (config, seed) and diverges on different seeds

The same `(config, seed)` MUST reproduce an identical stream. A different `seed` MUST produce a different stream. [REQUIRED]

#### Scenario: replay determinism

- **GIVEN** a config and `seed=42`
- **WHEN** `generate` is called twice
- **THEN** both streams are identical

### Requirement: Class clusters stay separable

Feature draws for different classes SHOULD remain well-separated relative to noise at any `classes_per_task`. No test measures separability directly. [OBSERVED]

#### Scenario: cluster geometry

- **GIVEN** any valid `classes_per_task`
- **WHEN** examples are drawn
- **THEN** class centers are spaced farther apart than the noise standard deviation

## Usage census

| Call site | Consumes |
|---|---|
| `eval/stream/registry.py` | imports `SplitClassifyGenerator`, registers as `"split-classify"` |
| `tests/stream/test_split_classify.py` | all generator behavior |
| `tests/stream/test_chance.py` | `chance_rate` measurement |

## Leak list

None.

## Banned vocabulary

`_class_label`, `_class_center`, `_draw_features`, `FEATURE_SEPARATION`, `FEATURE_NOISE_STD`, `example_source`, `probe_source`
