# eval/subject-protocol Specification

## Purpose
Defines the interface every subject implements, the probe isolation mechanism that prevents measurement from teaching, and the oracle subjects that validate the harness against known-correct metric values.
## Requirements
### Requirement: Subject interface methods

A subject MUST implement five methods: `observe(event)` returning an optional emission, `answer(probe)` returning an answer string, `idle(budget)` returning nothing, `snapshot()` returning an opaque state object, and `restore(state)` accepting a state object and resetting the subject to that exact state. [REQUIRED]

#### Scenario: interface completeness

- **WHEN** a class inherits from the subject protocol
- **THEN** it cannot be instantiated unless it implements all five methods

### Requirement: Cost counters

A subject MUST implement `cost()` returning a `CostCounters` object with three cumulative fields: `steps` (inner-loop step count), `flops` (estimated floating-point operations), and `wall_seconds` (elapsed wall time). Each field MUST be monotonically non-decreasing across calls within a run. [REQUIRED]

#### Scenario: monotonicity

- **GIVEN** a subject that has processed some events
- **WHEN** `cost()` is called twice with processing between them
- **THEN** every field in the second result is greater than or equal to the corresponding field in the first

### Requirement: Probe isolation via snapshot-restore

The harness MUST isolate every probe by calling `snapshot()` before `answer(probe)`, then calling `restore(state)` with the saved snapshot afterward. The subject's state after restore MUST be identical to its state before the probe. [REQUIRED]

#### Scenario: state unchanged after isolated probe

- **GIVEN** a subject that has processed N observe events
- **WHEN** the harness snapshots, calls `answer` on a probe, then restores
- **THEN** the subject produces the same outputs on subsequent events as one that never saw the probe

### Requirement: Isolation integrity test

The harness MUST ship an integrity test that feeds a stream through a subject twice: once with probes isolated via snapshot-restore, and once with probe isolation disabled. The test MUST compare outputs on post-probe events. A subject that learns from probes (the cheater oracle) MUST produce different outputs in the two runs, proving isolation caught it. A subject that does not learn from probes MUST produce identical outputs in both runs. [REQUIRED]

#### Scenario: cheater detected

- **WHEN** the cheater oracle runs through the integrity test
- **THEN** the isolated and non-isolated runs produce different post-probe outputs

#### Scenario: honest subject passes

- **WHEN** the perfect-memory oracle runs through the integrity test
- **THEN** the isolated and non-isolated runs produce identical post-probe outputs

### Requirement: Snapshot-restore round-trip fidelity

A test MUST verify that snapshot-restore is exact: snapshot a subject, feed noise events, restore, then check that the subject produces the same outputs as a run that never saw the noise. "Same outputs" means identical answer strings on the same probes. [REQUIRED]

#### Scenario: noise erased by restore

- **GIVEN** a subject that has observed events A, B, C
- **WHEN** state is snapshotted, events X, Y, Z (noise) are fed, state is restored
- **THEN** the subject answers probes identically to one that only saw A, B, C

### Requirement: Read-only flag as optional cheaper isolation

A subject MAY implement a `read_only` flag that prevents learning during `answer`. A subject MUST NOT use the read-only path until a side-by-side check against snapshot-restore on a short stream shows identical answers. Snapshot-restore remains the reference. [REQUIRED]

#### Scenario: read-only equivalence check

- **GIVEN** a subject that supports `read_only`
- **WHEN** the same stream runs with read-only isolation and with snapshot-restore isolation
- **THEN** both runs produce identical probe answers

### Requirement: Perfect-memory oracle

An oracle subject that stores every observed key-value pair and answers every probe correctly for any fact it has seen. [REQUIRED]

#### Scenario: recalls a fact at any distance

- **GIVEN** a fact taught at position 5
- **WHEN** a probe for that fact arrives at position 500
- **THEN** the oracle answers correctly

#### Scenario: does not know untaught facts

- **GIVEN** a probe for a fact never observed
- **WHEN** the oracle answers
- **THEN** the answer is wrong (or a distinguished "unknown" value)

### Requirement: Forgetful oracle

An oracle that remembers only the most recent item. It answers correctly if the probe asks about the last thing it saw. It answers wrong otherwise. [REQUIRED]

#### Scenario: correct on immediate recall

- **GIVEN** an observe event followed immediately by a probe for it
- **WHEN** the oracle answers
- **THEN** the answer is correct

#### Scenario: wrong after intervening events

- **GIVEN** an observe event, then two more observe events, then a probe for the first
- **WHEN** the oracle answers
- **THEN** the answer is wrong

### Requirement: Chance oracle

An oracle that answers randomly from the vocabulary. Over enough probes, its accuracy MUST land within sampling error of the stream's stated chance rate. [REQUIRED]

#### Scenario: accuracy near chance

- **GIVEN** a stream with chance rate 1/50 and at least 1000 probes
- **WHEN** the chance oracle answers all probes
- **THEN** its accuracy falls within a 99% binomial confidence interval of 1/50

### Requirement: Task-wiper oracle

An oracle that learns the current task perfectly and destroys the previous task's knowledge on every task switch. After a `Boundary(TASK_SWITCH)`, it forgets the task that just ended. [REQUIRED]

#### Scenario: current task correct, previous task at chance

- **GIVEN** a split-classify stream with at least 2 tasks
- **WHEN** probes arrive after task 1's examples
- **THEN** task 1 probes are correct and task 0 probes are at chance accuracy

### Requirement: Cheater oracle

An oracle that learns from probes. It reads the truth answer during `answer()` and stores it. It exists solely to test that probe isolation catches it. [REQUIRED]

#### Scenario: learns from probes when unprotected

- **GIVEN** probe isolation is disabled
- **WHEN** the cheater sees a probe, then later sees the same probe again
- **THEN** the cheater answers the second probe correctly (having learned from the first)

#### Scenario: cannot learn from probes when isolated

- **GIVEN** probe isolation via snapshot-restore is active
- **WHEN** the cheater sees a probe, then later sees the same probe again
- **THEN** the cheater does not benefit from the first probe (restore erased the learning)

