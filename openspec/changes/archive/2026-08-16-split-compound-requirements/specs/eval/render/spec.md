## MODIFIED Requirements

### Requirement: RENDER_VERSION is a non-empty string

`RENDER_VERSION` MUST be a non-empty `str`. Its literal value (`"2"`) is not pinned by any test. [REQUIRED]

#### Scenario: version is present

- **GIVEN** the module is imported
- **WHEN** `RENDER_VERSION` is read
- **THEN** it is a non-empty string

#### Scenario: version is a str

- **GIVEN** the module is imported
- **WHEN** `type(RENDER_VERSION)` is checked
- **THEN** it is `str`

### Requirement: fact payload renders as "[fact] key = value"

`render_event` on an `Observe` with payload `{"key": k, "value": v}` MUST return exactly `f"[fact] {k} = {v}"`. [REQUIRED]

#### Scenario: rendering a fact

- **GIVEN** an `Observe` with payload `{"key": "beam", "value": "duty"}`
- **WHEN** `render_event` is called
- **THEN** it returns `"[fact] beam = duty"`

#### Scenario: fact format is exact

- **GIVEN** an `Observe` with payload `{"key": "alpha", "value": "beta"}`
- **WHEN** `render_event` is called
- **THEN** the result starts with `"[fact] "` and contains `" = "`

### Requirement: prose payload renders bare with no marker

`render_event` on an `Observe` with payload `{"text": t}` MUST return exactly that text, unmodified, with no prefix. [REQUIRED]

#### Scenario: rendering prose

- **GIVEN** an `Observe` with payload `{"text": "the mill wheel turned all morning"}`
- **WHEN** `render_event` is called
- **THEN** it returns `"the mill wheel turned all morning"` exactly

#### Scenario: prose has no prefix

- **GIVEN** an `Observe` with payload `{"text": "hello world"}`
- **WHEN** `render_event` is called
- **THEN** the result does not start with `"["` or any marker

### Requirement: example payload renders with 2-decimal features and hides source

`render_event` on an `Observe` with payload `{"features": [...], "label": l, "source": s}` MUST return `f"[example] features=({...}) label={l}"` with each feature formatted to exactly 2 decimal places. The `source` value MUST NOT appear anywhere in the output. [REQUIRED]

#### Scenario: rendering an example

- **GIVEN** an `Observe` with payload `{"features": [5.383941741180422, -0.5668909031452558], "label": "task0-class0", "source": "mnist-train-17"}`
- **WHEN** `render_event` is called
- **THEN** it returns `"[example] features=(5.38, -0.57) label=task0-class0"`
- **AND** `"mnist-train-17"` does not appear in the output

#### Scenario: rendering an example with 2-decimal features

- **GIVEN** an `Observe` with payload `{"features": [5.383941741180422, -0.5668909031452558], "label": "task0-class0", "source": "mnist-train-17"}`
- **WHEN** `render_event` is called
- **THEN** it returns `"[example] features=(5.38, -0.57) label=task0-class0"`

#### Scenario: source value hidden

- **GIVEN** an `Observe` with payload `{"features": [1.0], "label": "a", "source": "secret-source"}`
- **WHEN** `render_event` is called
- **THEN** `"secret-source"` does not appear in the output

#### Scenario: features use exactly 2 decimal places

- **GIVEN** an `Observe` with payload `{"features": [1.0, 2.12345], "label": "x", "source": null}`
- **WHEN** `render_event` is called
- **THEN** the output contains `"1.00"` and `"2.12"` with no more than 2 digits after the decimal

### Requirement: Idle carries no text and render_event refuses it

`carries_text` MUST return `False` for `Idle` and `True` for every other event kind. `render_event` MUST raise `ValueError` matching `"idle"` when given an `Idle`. [REQUIRED]

#### Scenario: Idle is not text

- **GIVEN** an `Idle` event
- **WHEN** `carries_text` is called
- **THEN** it returns `False`
- **AND** `render_event` on the same event raises `ValueError` matching `"idle"`

#### Scenario: carries_text returns False for Idle

- **GIVEN** an `Idle` event
- **WHEN** `carries_text` is called
- **THEN** it returns `False`

#### Scenario: render_event raises on Idle

- **GIVEN** an `Idle` event
- **WHEN** `render_event` is called
- **THEN** `ValueError` matching `"idle"` is raised

#### Scenario: carries_text returns True for Observe

- **GIVEN** an `Observe` event
- **WHEN** `carries_text` is called
- **THEN** it returns `True`

### Requirement: Probe renders its query and hides harness fields

`render_event` on a `Probe` MUST return `f"[probe] {event.query}"`. It MUST NOT include the probe's answer, `probe_id`, `task_id`, `teaching_position`, or `hostile` flag in the output. Repeated calls on the same `Probe` MUST return the same string. [REQUIRED]

#### Scenario: probe text hides harness fields

- **GIVEN** a `Probe` with a known answer, `probe_id`, `task_id`, `teaching_position=7`, `hostile=True`
- **WHEN** `render_event` is called twice
- **THEN** both outputs are equal, and none contains the answer, probe id, task id, `"7"`, `"True"`, or `"hostile"`

#### Scenario: probe renders query

- **GIVEN** a `Probe` with `query="what is beam?"`
- **WHEN** `render_event` is called
- **THEN** it returns `"[probe] what is beam?"`

#### Scenario: probe hides answer

- **GIVEN** a `Probe` with a known answer `"duty"`
- **WHEN** `render_event` is called
- **THEN** `"duty"` does not appear in the output

#### Scenario: probe hides probe_id and task_id

- **GIVEN** a `Probe` with `probe_id="p-99"`, `task_id="t-7"`
- **WHEN** `render_event` is called
- **THEN** neither `"p-99"` nor `"t-7"` appears in the output

#### Scenario: probe rendering is idempotent

- **GIVEN** a `Probe`
- **WHEN** `render_event` is called twice
- **THEN** both outputs are identical

### Requirement: Observe and Boundary rendering omits hostile and is stable

`render_event` on `Observe` and `Boundary` MUST NOT emit `"True"` or `"hostile"`, and repeated calls MUST return the same string. [REQUIRED]

#### Scenario: hostile flag never leaks

- **GIVEN** an `Observe` or `Boundary` with `hostile=True`
- **WHEN** `render_event` is called twice
- **THEN** both outputs are equal and contain neither `"True"` nor `"hostile"`

#### Scenario: Observe hostile flag never leaks

- **GIVEN** an `Observe` with `hostile=True`
- **WHEN** `render_event` is called
- **THEN** the output contains neither `"True"` nor `"hostile"`

#### Scenario: Boundary hostile flag never leaks

- **GIVEN** a `Boundary` with `hostile=True`
- **WHEN** `render_event` is called
- **THEN** the output contains neither `"True"` nor `"hostile"`

#### Scenario: Observe rendering is idempotent

- **GIVEN** an `Observe`
- **WHEN** `render_event` is called twice
- **THEN** both outputs are identical

### Requirement: format_features renders exactly 2 decimal places

`format_features` output MUST use exactly 2 decimal digits per value. No run of 3 or more digits after a decimal point is allowed. [REQUIRED]

#### Scenario: probe and example use the same feature notation

- **GIVEN** a probe query built from `format_features` and an example rendered with the same features
- **WHEN** both are inspected
- **THEN** neither string contains 3 or more consecutive digits after a decimal point

#### Scenario: 2-decimal precision enforced

- **GIVEN** features `[1.23456, 7.891]`
- **WHEN** `format_features` is called
- **THEN** the output contains `"1.23"` and `"7.89"` with no more than 2 digits after the decimal

#### Scenario: whole numbers get trailing zeros

- **GIVEN** features `[5.0, 10.0]`
- **WHEN** `format_features` is called
- **THEN** the output contains `"5.00"` and `"10.00"`

### Requirement: rendered text never contains difficulty or level labels

No text produced by `render_event` for a `difficulty-mix` stream's carries-text events SHALL contain the case-insensitive substrings `"difficulty"` or `"level"`. [REQUIRED]

#### Scenario: no difficulty leak in rendered text

- **GIVEN** a `difficulty-mix` stream
- **WHEN** every text-carrying event is rendered
- **THEN** no rendered string contains `"difficulty"` or `"level"` (case-insensitive)

#### Scenario: no level leak in rendered text

- **GIVEN** a `difficulty-mix` stream
- **WHEN** every text-carrying event is rendered
- **THEN** no rendered string contains `"level"` (case-insensitive)

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

#### Scenario: Idle events omitted

- **GIVEN** a sequence containing an `Idle` event
- **WHEN** `rendered_subject_view` is called
- **THEN** the Idle is not in the output

#### Scenario: hidden boundary omitted

- **GIVEN** a single `Boundary` item with `hidden_from_subject=True`
- **WHEN** `rendered_subject_view` is called
- **THEN** it yields nothing

#### Scenario: repeated calls produce same sequence

- **GIVEN** a sequence of `[Observe, Probe-with-truth, Idle, Boundary]`
- **WHEN** `rendered_subject_view` is called twice
- **THEN** both calls yield the same strings

#### Scenario: no harness fields leak

- **GIVEN** a sequence containing a Probe with truth, teaching_position, and budget
- **WHEN** `rendered_subject_view` is called
- **THEN** no output string contains the answer, probe_id, task_id, teaching_position, budget, or hostile values

### Requirement: default_token_counter is deterministic and reports its name

`default_token_counter(text)` MUST return the same positive integer for the same text across calls. `.name` MUST be a non-empty string. [REQUIRED]

#### Scenario: repeated counting is stable

- **GIVEN** fixed text
- **WHEN** `default_token_counter` is called twice
- **THEN** both calls return the same positive integer

#### Scenario: returns a positive integer

- **GIVEN** non-empty text
- **WHEN** `default_token_counter` is called
- **THEN** it returns a value > 0

#### Scenario: name is a non-empty string

- **GIVEN** `default_token_counter`
- **WHEN** `.name` is read
- **THEN** it is a non-empty string

### Requirement: default token counter name is "regex-whitespace-v1"

`default_token_counter.name` MUST equal `"regex-whitespace-v1"`. Configs may reference it by this exact string via `resolve_token_counter`. [REQUIRED]

#### Scenario: name equals regex-whitespace-v1

- **GIVEN** `default_token_counter`
- **WHEN** `.name` is read
- **THEN** it equals `"regex-whitespace-v1"`

#### Scenario: known counter name resolves

- **GIVEN** `token_counter="regex-whitespace-v1"` in a config
- **WHEN** the stream is generated
- **THEN** no error is raised and probes carry computed `token_distance`

#### Scenario: resolve_token_counter returns the default counter

- **GIVEN** `name="regex-whitespace-v1"`
- **WHEN** `resolve_token_counter` is called
- **THEN** the returned counter's `.name` equals `"regex-whitespace-v1"`

### Requirement: resolve_token_counter raises ValueError naming known counters

`resolve_token_counter(name)` MUST raise `ValueError` whose message contains `"regex-whitespace-v1"` when `name` is not registered. [REQUIRED]

#### Scenario: unknown counter name

- **GIVEN** `name="not-a-real-counter"`
- **WHEN** `resolve_token_counter` is called
- **THEN** `ValueError` mentioning `"regex-whitespace-v1"` is raised

#### Scenario: error message names the registered counter

- **GIVEN** `name="bogus"`
- **WHEN** `resolve_token_counter` raises
- **THEN** the message contains `"regex-whitespace-v1"`

### Requirement: TokenCounter protocol allows injection

Any object with a `name: str` attribute and `__call__(text: str) -> int` MUST be usable in place of `default_token_counter`. [REQUIRED]

#### Scenario: swap in a custom counter

- **GIVEN** an object with `.name = "fixed-for-test"` and `__call__` returning `1`
- **WHEN** it is used where a `TokenCounter` is expected
- **THEN** it reports `1` regardless of text

#### Scenario: custom counter name is preserved

- **GIVEN** an object with `.name = "my-counter"` and a valid `__call__`
- **WHEN** `.name` is read
- **THEN** it equals `"my-counter"`

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

#### Scenario: token distance derived from rendered text

- **GIVEN** a probe with intervening text-carrying events
- **WHEN** the stream is generated
- **THEN** `token_distance` equals the token counter applied to the concatenated rendered text of intervening events

#### Scenario: non-text events do not contribute to token distance

- **GIVEN** a probe separated from teaching by Idle events only
- **WHEN** the stream is generated
- **THEN** `token_distance == 0`
