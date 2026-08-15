## Purpose

Defines the event type hierarchy and truth envelope that every stream generator emits and every harness consumer reads.

## Requirements

### Requirement: Event immutability

Every event type (Observe, Probe, Idle, Boundary) MUST be frozen: assigning to any field after construction MUST raise `dataclasses.FrozenInstanceError`. [REQUIRED]

#### Scenario: attribute assignment on a constructed event

- **GIVEN** a constructed `Observe`
- **WHEN** a field is assigned a new value
- **THEN** `dataclasses.FrozenInstanceError` is raised

### Requirement: Event required fields

`Probe` MUST require `probe_id`, `task_id`, and `query`. `Idle` MUST require `budget`. `Observe` MUST require `payload`. `Boundary` MUST require `kind`. Omitting any of these MUST raise `TypeError`. [REQUIRED]

#### Scenario: omitted required field

- **GIVEN** a call to construct a `Probe` without `probe_id`
- **WHEN** construction is attempted
- **THEN** `TypeError` is raised

### Requirement: Event optional-field defaults

`narration` MUST default to `None`, `hostile` MUST default to `False`, and `Boundary.hidden_from_subject` MUST default to `False`. [REQUIRED]

#### Scenario: default construction

- **GIVEN** an `Observe` constructed without `narration` or `hostile`
- **WHEN** the fields are read
- **THEN** `narration is None` and `hostile is False`

### Requirement: BoundaryKind string values

`BoundaryKind` members MUST serialize to the literal strings `"session_end"`, `"task_switch"`, `"distribution_shift"`. These values appear in rendered subject-facing text and in hashed serialized sequences. [REQUIRED]

#### Scenario: rendered boundary text

- **GIVEN** a `Boundary` event with `kind=BoundaryKind.TASK_SWITCH`
- **WHEN** the event is rendered or serialized
- **THEN** the output contains the literal string `"task_switch"`

### Requirement: Event type union

`Event` MUST be a type alias for `Union[Observe, Probe, Idle, Boundary]`. [REQUIRED]

#### Scenario: isinstance dispatch

- **GIVEN** any value produced by a stream generator as an event
- **WHEN** it is checked with `isinstance` against each of the four types
- **THEN** exactly one check returns `True`

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

### Requirement: ProbeTruth immutability and optional difficulty

`ProbeTruth` MUST be frozen. `difficulty` MUST default to `None`. [REQUIRED]

#### Scenario: difficulty is optional

- **GIVEN** a `ProbeTruth(answer="x")` with no `difficulty` argument
- **WHEN** `.difficulty` is read
- **THEN** it is `None`

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

### Requirement: harness_view yields everything unfiltered

`harness_view` MUST yield every `StreamItem` unchanged, including hidden boundaries and truth. [REQUIRED]

#### Scenario: hidden boundary included for harness

- **GIVEN** a `StreamItem` wrapping a `Boundary` with `hidden_from_subject=True`
- **WHEN** `harness_view` is applied
- **THEN** that exact `StreamItem` is yielded

### Requirement: narration and hostile flags exist for future stages

`narration` and `hostile` SHOULD exist on every event as forward-looking hooks for a Stage 0 narration decoder and Stage 2 attack streams. No current generator or caller sets either to a non-default value. [OBSERVED]

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
