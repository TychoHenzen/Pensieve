## Purpose

Defines the event type hierarchy and truth envelope that every stream generator emits and every harness consumer reads.

## Requirements

### Requirement: Event immutability

Every event type (Observe, Probe, Idle, Boundary) MUST be frozen: assigning to any field after construction MUST raise `dataclasses.FrozenInstanceError`. [REQUIRED]

#### Scenario: attribute assignment on a constructed event

- **GIVEN** a constructed `Observe`
- **WHEN** a field is assigned a new value
- **THEN** `dataclasses.FrozenInstanceError` is raised

#### Scenario: attribute assignment on a constructed Observe

- **GIVEN** a constructed `Observe`
- **WHEN** a field is assigned a new value
- **THEN** `dataclasses.FrozenInstanceError` is raised

#### Scenario: attribute assignment on a constructed Probe

- **GIVEN** a constructed `Probe`
- **WHEN** a field is assigned a new value
- **THEN** `dataclasses.FrozenInstanceError` is raised

#### Scenario: attribute assignment on a constructed Idle

- **GIVEN** a constructed `Idle`
- **WHEN** a field is assigned a new value
- **THEN** `dataclasses.FrozenInstanceError` is raised

#### Scenario: attribute assignment on a constructed Boundary

- **GIVEN** a constructed `Boundary`
- **WHEN** a field is assigned a new value
- **THEN** `dataclasses.FrozenInstanceError` is raised

### Requirement: Event required fields

`Probe` MUST require `probe_id`, `task_id`, and `query`. `Idle` MUST require `budget`. `Observe` MUST require `payload`. `Boundary` MUST require `kind`. Omitting any of these MUST raise `TypeError`. [REQUIRED]

#### Scenario: omitted required field

- **GIVEN** a call to construct a `Probe` without `probe_id`
- **WHEN** construction is attempted
- **THEN** `TypeError` is raised

#### Scenario: Probe without probe_id

- **GIVEN** a call to construct a `Probe` without `probe_id`
- **WHEN** construction is attempted
- **THEN** `TypeError` is raised

#### Scenario: Probe without task_id

- **GIVEN** a call to construct a `Probe` without `task_id`
- **WHEN** construction is attempted
- **THEN** `TypeError` is raised

#### Scenario: Probe without query

- **GIVEN** a call to construct a `Probe` without `query`
- **WHEN** construction is attempted
- **THEN** `TypeError` is raised

#### Scenario: Idle without budget

- **GIVEN** a call to construct an `Idle` without `budget`
- **WHEN** construction is attempted
- **THEN** `TypeError` is raised

#### Scenario: Observe without payload

- **GIVEN** a call to construct an `Observe` without `payload`
- **WHEN** construction is attempted
- **THEN** `TypeError` is raised

#### Scenario: Boundary without kind

- **GIVEN** a call to construct a `Boundary` without `kind`
- **WHEN** construction is attempted
- **THEN** `TypeError` is raised

### Requirement: Event optional-field defaults

`narration` MUST default to `None`, `hostile` MUST default to `False`, and `Boundary.hidden_from_subject` MUST default to `False`. [REQUIRED]

#### Scenario: default construction

- **GIVEN** an `Observe` constructed without `narration` or `hostile`
- **WHEN** the fields are read
- **THEN** `narration is None` and `hostile is False`

#### Scenario: narration defaults to None

- **GIVEN** an `Observe` constructed without `narration`
- **WHEN** `.narration` is read
- **THEN** it is `None`

#### Scenario: hostile defaults to False

- **GIVEN** an `Observe` constructed without `hostile`
- **WHEN** `.hostile` is read
- **THEN** it is `False`

#### Scenario: hidden_from_subject defaults to False

- **GIVEN** a `Boundary` constructed without `hidden_from_subject`
- **WHEN** `.hidden_from_subject` is read
- **THEN** it is `False`

#### Scenario: narration defaults to None on Probe

- **GIVEN** a `Probe` constructed without `narration`
- **WHEN** `.narration` is read
- **THEN** it is `None`

### Requirement: BoundaryKind string values

`BoundaryKind` members MUST serialize to the literal strings `"session_end"`, `"task_switch"`, `"distribution_shift"`. These values appear in rendered subject-facing text and in hashed serialized sequences. [REQUIRED]

#### Scenario: rendered boundary text

- **GIVEN** a `Boundary` event with `kind=BoundaryKind.TASK_SWITCH`
- **WHEN** the event is rendered or serialized
- **THEN** the output contains the literal string `"task_switch"`

#### Scenario: session_end string value

- **GIVEN** `BoundaryKind.SESSION_END`
- **WHEN** `.value` is read
- **THEN** it equals `"session_end"`

#### Scenario: task_switch string value

- **GIVEN** `BoundaryKind.TASK_SWITCH`
- **WHEN** `.value` is read
- **THEN** it equals `"task_switch"`

#### Scenario: distribution_shift string value

- **GIVEN** `BoundaryKind.DISTRIBUTION_SHIFT`
- **WHEN** `.value` is read
- **THEN** it equals `"distribution_shift"`

### Requirement: Event type union

`Event` MUST be a type alias for `Union[Observe, Probe, Idle, Boundary]`. [REQUIRED]

#### Scenario: isinstance dispatch

- **GIVEN** any value produced by a stream generator as an event
- **WHEN** it is checked with `isinstance` against each of the four types
- **THEN** exactly one check returns `True`

#### Scenario: isinstance dispatch covers all four types

- **GIVEN** any value produced by a stream generator as an event
- **WHEN** it is checked with `isinstance` against each of the four types
- **THEN** exactly one check returns `True`

#### Scenario: Event alias accepts all four types

- **GIVEN** one instance of each of `Observe`, `Probe`, `Idle`, `Boundary`
- **WHEN** each is checked against the `Event` union
- **THEN** every check passes

### Requirement: StreamItem truth pairing invariant

`StreamItem` MUST reject a `Probe` event paired with `truth=None`, raising `ValueError("a Probe StreamItem must carry a ProbeTruth")`. It MUST reject any non-`Probe` event paired with a non-`None` truth, raising `ValueError("only a Probe StreamItem may carry a ProbeTruth")`. [REQUIRED]

#### Scenario: probe without truth

- **GIVEN** a `Probe` event
- **WHEN** `StreamItem(event=probe, truth=None)` is constructed
- **THEN** `ValueError("a Probe StreamItem must carry a ProbeTruth")` is raised

#### Scenario: non-probe with truth

- **GIVEN** an `Observe` event and a `ProbeTruth`
- **WHEN** `StreamItem(event=observe, truth=truth)` is constructed
- **THEN** `ValueError("only a Probe StreamItem may carry a ProbeTruth")` is raised

#### Scenario: Idle with truth rejected

- **GIVEN** an `Idle` event and a `ProbeTruth`
- **WHEN** `StreamItem(event=idle, truth=truth)` is constructed
- **THEN** `ValueError("only a Probe StreamItem may carry a ProbeTruth")` is raised

#### Scenario: Boundary with truth rejected

- **GIVEN** a `Boundary` event and a `ProbeTruth`
- **WHEN** `StreamItem(event=boundary, truth=truth)` is constructed
- **THEN** `ValueError("only a Probe StreamItem may carry a ProbeTruth")` is raised

#### Scenario: Probe with truth accepted

- **GIVEN** a `Probe` event and a valid `ProbeTruth`
- **WHEN** `StreamItem(event=probe, truth=truth)` is constructed
- **THEN** no error is raised and `.truth` equals the provided `ProbeTruth`

### Requirement: ProbeTruth immutability and optional difficulty

`ProbeTruth` MUST be frozen. `difficulty` MUST default to `None`. [REQUIRED]

#### Scenario: difficulty is optional

- **GIVEN** a `ProbeTruth(answer="x")` with no `difficulty` argument
- **WHEN** `.difficulty` is read
- **THEN** it is `None`

#### Scenario: ProbeTruth is frozen

- **GIVEN** a constructed `ProbeTruth`
- **WHEN** `.answer` is assigned a new value
- **THEN** `dataclasses.FrozenInstanceError` is raised

#### Scenario: difficulty defaults to None

- **GIVEN** a `ProbeTruth(answer="x")` with no `difficulty` argument
- **WHEN** `.difficulty` is read
- **THEN** it is `None`

#### Scenario: difficulty can be set explicitly

- **GIVEN** a `ProbeTruth(answer="x", difficulty=3)`
- **WHEN** `.difficulty` is read
- **THEN** it is `3`

### Requirement: subject_view hides truth and hidden boundaries

`subject_view` MUST yield only bare `Event` values, MUST omit any `Boundary` whose `hidden_from_subject` is `True`, and MUST preserve the relative order and `.position` values of everything it yields. The probe answer string MUST NOT appear in the `repr()` of subject-view output. [REQUIRED]

#### Scenario: hidden boundary omitted

- **GIVEN** a `StreamItem` sequence containing a `Boundary` with `hidden_from_subject=True`
- **WHEN** `subject_view` is applied
- **THEN** that event does not appear in the output

#### Scenario: probe answer never leaks

- **GIVEN** a `StreamItem` wrapping a `Probe` with `ProbeTruth(answer="secret-42")`
- **WHEN** `subject_view` output is converted to `repr()`
- **THEN** the string `"secret-42"` does not appear

#### Scenario: yields bare Events not StreamItems

- **GIVEN** a `StreamItem` sequence
- **WHEN** `subject_view` is applied
- **THEN** every yielded value is an `Event`, not a `StreamItem`

#### Scenario: preserves position order

- **GIVEN** a `StreamItem` sequence with positions `[0, 1, 2, 3]`
- **WHEN** `subject_view` is applied
- **THEN** the positions of yielded events are in strictly increasing order

#### Scenario: visible boundary included

- **GIVEN** a `Boundary` with `hidden_from_subject=False`
- **WHEN** `subject_view` is applied
- **THEN** that event appears in the output

### Requirement: harness_view yields everything unfiltered

`harness_view` MUST yield every `StreamItem` unchanged, including hidden boundaries and truth. [REQUIRED]

#### Scenario: hidden boundary included for harness

- **GIVEN** a `StreamItem` wrapping a `Boundary` with `hidden_from_subject=True`
- **WHEN** `harness_view` is applied
- **THEN** that exact `StreamItem` is yielded

#### Scenario: truth preserved in harness view

- **GIVEN** a `StreamItem` wrapping a `Probe` with a `ProbeTruth`
- **WHEN** `harness_view` is applied
- **THEN** the yielded `StreamItem` has the same `truth` value

### Requirement: narration and hostile flags exist for future stages

`narration` and `hostile` MUST exist on every event as forward-looking hooks for a Stage 0 narration decoder and Stage 2 attack streams. No current generator or caller sets either to a non-default value. [OBSERVED]

#### Scenario: no current consumer

- **GIVEN** any generator in the codebase
- **WHEN** it constructs an event
- **THEN** `narration` and `hostile` are left at their defaults

## Usage census

| Call site | Consumes |
|---|---|
| `eval/stream/truth.py` | `Boundary`, `Event`, `Probe` types, `isinstance` dispatch, `.hidden_from_subject` |
| `eval/stream/render.py` | all four event classes, `.kind.value`, `.payload`, `.query` |
| `eval/stream/serialize.py` | `type(item.event).__name__`, all dataclass field names via `dataclasses.fields` |
| `eval/stream/generators/assoc.py` | constructs `Observe`, `Probe`; uses `dataclasses.replace` on `Probe` |
| `eval/stream/generators/split_classify.py` | constructs `Boundary`, `BoundaryKind`, `Observe`, `Probe` |
| `eval/stream/generators/difficulty_mix.py` | constructs `Idle`, `Probe` |
| `tests/stream/test_events.py` | required-field `TypeError`, frozen-instance error, defaults |
| `tests/stream/test_truth.py` | construction validation, immutability, view filtering |

## Leak list

None. No copies in build output.

## Banned vocabulary

`_EventBase`
