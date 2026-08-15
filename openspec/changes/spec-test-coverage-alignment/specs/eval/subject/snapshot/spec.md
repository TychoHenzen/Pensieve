## Purpose

Defines the snapshot/restore contract that all stateful subject implementations must satisfy, enabling probe isolation and state rollback.

## ADDED Requirements

### Requirement: Snapshot captures full state

`snapshot()` MUST return an opaque state object that captures the subject's entire internal state at the moment of the call. [REQUIRED]

#### Scenario: restore erases intermediate observations on PerfectMemoryOracle

- **GIVEN** a PerfectMemoryOracle that has observed some events
- **WHEN** `snapshot()` is taken, then noise events are observed, then `restore(state)` is called
- **THEN** subsequent answers match a clean oracle that never saw the noise events

#### Scenario: restore erases intermediate observations on ForgetfulOracle

- **GIVEN** a ForgetfulOracle that has observed some events
- **WHEN** `snapshot()` is taken, then noise events are observed, then `restore(state)` is called
- **THEN** subsequent answers match a clean oracle that never saw the noise events
