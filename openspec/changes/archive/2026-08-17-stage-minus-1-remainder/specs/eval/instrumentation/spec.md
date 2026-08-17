## Purpose

Defines the cost-counter interface, the extensible instrumentation schema for later-stage signals, and the difficulty-correlation test that validates compute tracking.

## ADDED Requirements

### Requirement: CostCounters data structure

`CostCounters` MUST be a frozen dataclass with three fields: `steps` (int), `flops` (int), and `wall_seconds` (float). All three MUST default to 0. [REQUIRED]

#### Scenario: default construction

- **WHEN** `CostCounters()` is constructed with no arguments
- **THEN** `steps == 0`, `flops == 0`, `wall_seconds == 0.0`

### Requirement: Instrumentation schema extensibility

The run record MUST include an optional-fields schema for four future signals: `halting_steps`, `memory_write_magnitude`, `channel_bandwidth`, and `consolidation_gain`. Each field MUST default to `None`. Current subjects leave them empty. [REQUIRED]

#### Scenario: fields present but empty in a Stage -1 run

- **GIVEN** a completed run with a Stage -1 subject
- **WHEN** the instrumentation fields are inspected
- **THEN** all four are `None`

### Requirement: Difficulty-correlation test

A test MUST run the `difficulty-mix` generator with a variable-compute oracle and verify that compute per input correlates with the difficulty label. The oracle MUST spend more steps on longer arithmetic chains. [REQUIRED]

#### Scenario: positive correlation

- **GIVEN** a difficulty-mix stream and a variable-compute oracle
- **WHEN** inner-loop steps are plotted against difficulty labels
- **THEN** the Spearman rank correlation is positive and statistically significant
