# codecs/narration Specification

## Purpose
Turns workspace slot state into human-readable text for inspection, writing to the narration field the harness already reserves on each event in the run record.
## Requirements
### Requirement: Narration decoder produces text from workspace state

The narration decoder MUST accept the current workspace slot vectors and produce a text description of the workspace state. The output is diagnostic, not a subject response.

#### Scenario: workspace state narrated

- **WHEN** the narration decoder is called with workspace slot vectors
- **THEN** it produces a non-empty text string describing the slot state

#### Scenario: different states produce different narrations

- **WHEN** the narration decoder is called with two distinct workspace states
- **THEN** the two narration strings differ

### Requirement: Narration writes to the event narration field

The narration decoder's output MUST be written to the `narration` field on the run record event. The harness already defines this field as `str | None` on each event.

#### Scenario: narration field populated

- **WHEN** an event is processed with narration enabled
- **THEN** the event's `narration` field in the run record contains the narration decoder's output

#### Scenario: narration field absent when disabled

- **WHEN** an event is processed with narration disabled
- **THEN** the event's `narration` field is `None`

### Requirement: Narration is optional and configurable

Narration MUST be off by default (no performance cost when unused). It MUST be toggleable via run config without changing the subject code.

#### Scenario: off by default

- **WHEN** a run starts with default config
- **THEN** no narration decoder runs and no narration fields are populated

#### Scenario: enabled via config

- **WHEN** the run config sets narration to enabled
- **THEN** the narration decoder runs on each event and populates the narration field

