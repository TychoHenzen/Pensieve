## Purpose

Converts stream events into subject-facing text and provides pluggable token counting for measuring inter-event distances.

## Requirements

### Requirement: RENDER_VERSION is a non-empty string

`RENDER_VERSION` MUST be a non-empty `str`. Its literal value (`"2"`) is not pinned by any test. [REQUIRED]

#### Scenario: version is present

- **GIVEN** the module is imported
- **WHEN** `RENDER_VERSION` is read
- **THEN** it is a non-empty string

### Requirement: fact payload renders as "[fact] key = value"

`render_event` on an `Observe` with payload `{"key": k, "value": v}` MUST return exactly `f"[fact] {k} = {v}"`. [REQUIRED]

#### Scenario: rendering a fact

- **GIVEN** an `Observe` with payload `{"key": "beam", "value": "duty"}`
- **WHEN** `render_event` is called
- **THEN** it returns `"[fact] beam = duty"`

### Requirement: prose payload renders bare with no marker

`render_event` on an `Observe` with payload `{"text": t}` MUST return exactly that text, unmodified, with no prefix. [REQUIRED]

#### Scenario: rendering prose

- **GIVEN** an `Observe` with payload `{"text": "the mill wheel turned all morning"}`
- **WHEN** `render_event` is called
- **THEN** it returns `"the mill wheel turned all morning"` exactly

### Requirement: example payload renders with 2-decimal features and hides source

`render_event` on an `Observe` with payload `{"features": [...], "label": l, "source": s}` MUST return `f"[example] features=({...}) label={l}"` with each feature formatted to exactly 2 decimal places. The `source` value MUST NOT appear anywhere in the output. [REQUIRED]

#### Scenario: rendering an example

- **GIVEN** an `Observe` with payload `{"features": [5.383941741180422, -0.5668909031452558], "label": "task0-class0", "source": "mnist-train-17"}`
- **WHEN** `render_event` is called
- **THEN** it returns `"[example] features=(5.38, -0.57) label=task0-class0"`
- **AND** `"mnist-train-17"` does not appear in the output

### Requirement: unknown Observe payload shape raises ValueError

`render_event` MUST raise `ValueError` with a message containing `"known payload shapes"` when the payload is a non-matching mapping or not a mapping at all. [REQUIRED]

#### Scenario: unrecognized shape

- **GIVEN** an `Observe` with payload `{"mystery": 1}`
- **WHEN** `render_event` is called
- **THEN** `ValueError` matching `"known payload shapes"` is raised

#### Scenario: non-mapping payload

- **GIVEN** an `Observe` with payload `["a", "b"]`
- **WHEN** `render_event` is called
- **THEN** `ValueError` matching `"known payload shapes"` is raised

### Requirement: Idle carries no text and render_event refuses it

`carries_text` MUST return `False` for `Idle` and `True` for every other event kind. `render_event` MUST raise `ValueError` matching `"idle"` when given an `Idle`. [REQUIRED]

#### Scenario: Idle is not text

- **GIVEN** an `Idle` event
- **WHEN** `carries_text` is called
- **THEN** it returns `False`
- **AND** `render_event` on the same event raises `ValueError` matching `"idle"`

### Requirement: Probe renders its query and hides harness fields

`render_event` on a `Probe` MUST return `f"[probe] {event.query}"`. It MUST NOT include the probe's answer, `probe_id`, `task_id`, `teaching_position`, or `hostile` flag in the output. Repeated calls on the same `Probe` MUST return the same string. [REQUIRED]

#### Scenario: probe text hides harness fields

- **GIVEN** a `Probe` with a known answer, `probe_id`, `task_id`, `teaching_position=7`, `hostile=True`
- **WHEN** `render_event` is called twice
- **THEN** both outputs are equal, and none contains the answer, probe id, task id, `"7"`, `"True"`, or `"hostile"`

### Requirement: Observe and Boundary rendering omits hostile and is stable

`render_event` on `Observe` and `Boundary` MUST NOT emit `"True"` or `"hostile"`, and repeated calls MUST return the same string. [REQUIRED]

#### Scenario: hostile flag never leaks

- **GIVEN** an `Observe` or `Boundary` with `hostile=True`
- **WHEN** `render_event` is called twice
- **THEN** both outputs are equal and contain neither `"True"` nor `"hostile"`

### Requirement: format_features renders exactly 2 decimal places

`format_features` output MUST use exactly 2 decimal digits per value. No run of 3 or more digits after a decimal point is allowed. [REQUIRED]

#### Scenario: probe and example use the same feature notation

- **GIVEN** a probe query built from `format_features` and an example rendered with the same features
- **WHEN** both are inspected
- **THEN** neither string contains 3 or more consecutive digits after a decimal point

### Requirement: rendered text never contains difficulty or level labels

No text produced by `render_event` for a `difficulty-mix` stream's carries-text events SHALL contain the case-insensitive substrings `"difficulty"` or `"level"`. [REQUIRED]

#### Scenario: no difficulty leak in rendered text

- **GIVEN** a `difficulty-mix` stream
- **WHEN** every text-carrying event is rendered
- **THEN** no rendered string contains `"difficulty"` or `"level"` (case-insensitive)

### Requirement: rendered_subject_view drops Idle and hidden Boundary

`rendered_subject_view` MUST omit `Idle` events and `Boundary` events with `hidden_from_subject=True`. It MUST produce the same sequence across repeated calls and MUST NOT leak any probe answer, identifiers, teaching position, budget, or hostile flag. [REQUIRED]

#### Scenario: four items in, three lines out

- **GIVEN** a sequence of `[Observe, Probe-with-truth, Idle, Boundary]`
- **WHEN** `rendered_subject_view` is called twice
- **THEN** both calls yield the same 3 strings, none containing harness-only fields

#### Scenario: hidden boundary is omitted

- **GIVEN** a single `Boundary` item with `hidden_from_subject=True`
- **WHEN** `rendered_subject_view` is called
- **THEN** it yields nothing

### Requirement: default_token_counter is deterministic and reports its name

`default_token_counter(text)` MUST return the same positive integer for the same text across calls. `.name` MUST be a non-empty string. [REQUIRED]

#### Scenario: repeated counting is stable

- **GIVEN** fixed text
- **WHEN** `default_token_counter` is called twice
- **THEN** both calls return the same positive integer

### Requirement: default token counter name is "regex-whitespace-v1"

`default_token_counter.name` MUST equal `"regex-whitespace-v1"`. Configs may reference it by this exact string via `resolve_token_counter`. [REQUIRED]

#### Scenario: known counter name resolves

- **GIVEN** `token_counter="regex-whitespace-v1"` in a config
- **WHEN** the stream is generated
- **THEN** no error is raised and probes carry computed `token_distance`

### Requirement: resolve_token_counter raises ValueError naming known counters

`resolve_token_counter(name)` MUST raise `ValueError` whose message contains `"regex-whitespace-v1"` when `name` is not registered. [REQUIRED]

#### Scenario: unknown counter name

- **GIVEN** `name="not-a-real-counter"`
- **WHEN** `resolve_token_counter` is called
- **THEN** `ValueError` mentioning `"regex-whitespace-v1"` is raised

### Requirement: TokenCounter protocol allows injection

Any object with a `name: str` attribute and `__call__(text: str) -> int` MUST be usable in place of `default_token_counter`. [REQUIRED]

#### Scenario: swap in a custom counter

- **GIVEN** an object with `.name = "fixed-for-test"` and `__call__` returning `1`
- **WHEN** it is used where a `TokenCounter` is expected
- **THEN** it reports `1` regardless of text

### Requirement: token distance is measured over rendered text between teaching and probe

A probe's `token_distance` MUST be derived by rendering every text-carrying event between its teaching position and its probe position and passing that text into the configured token counter. It MUST be `0` when no intervening text-carrying events exist and MUST rise monotonically as event distance grows. [REQUIRED]

#### Scenario: adjacent teaching and probe measure zero

- **GIVEN** a probe placed immediately after its teaching with a counter configured
- **WHEN** the stream is generated
- **THEN** `probe.token_distance == 0`

#### Scenario: token distance grows with event distance

- **GIVEN** probes at increasing distances from the same teaching
- **WHEN** the stream is generated
- **THEN** `token_distance` values are non-decreasing and strictly greater for the farthest probe

## Usage census

| Call site | Consumes |
|---|---|
| `eval/stream/generators/assoc.py` | `render_event`, `carries_text`, `resolve_token_counter` |
| `eval/stream/generators/split_classify.py` | `format_features` |
| `scripts/show_stream.py` | `RENDER_VERSION`, `carries_text`, `default_token_counter`, `render_event` |
| `tests/stream/test_render.py` | all public API |
| `tests/stream/test_difficulty_mix.py` | rendered text substring checks |
| `tests/stream/test_split_classify.py` | feature format precision |
| `tests/stream/test_assoc.py` | `default_token_counter.name`, `token_distance` behavior |

## Leak list

None.

## Banned vocabulary

`_RegexTokenCounter`, `_WORD_PATTERN`, `_FEATURE_DECIMALS`, `_PAYLOAD_RENDERERS`, `_render_fact`, `_render_prose`, `_render_example`, `_render_observe`
