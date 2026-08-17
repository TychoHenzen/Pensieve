# eval/metrics Specification

## Purpose
Defines the probe log schema and the metric set that every run produces, plus oracle tests that validate each metric against paper-derived expected values.
## Requirements
### Requirement: Probe log schema

Every run MUST produce a probe log: a table with one row per probe scored. Each row MUST contain `position` (stream position), `probe_id`, `task_id`, `teaching_position` (position of the event that taught the answer), `correct` (boolean), and the `CostCounters` snapshot at that moment. [REQUIRED]

#### Scenario: row completeness

- **GIVEN** a completed run
- **WHEN** any row of the probe log is inspected
- **THEN** it has all six fields and none are null

### Requirement: Task accuracy

Task accuracy MUST be computed per task, per probe class, and pooled. Per-task accuracy is the fraction of correct probes for that task across all measurement points. [REQUIRED]

#### Scenario: perfect-memory oracle scores 1.0 on every task

- **GIVEN** the perfect-memory oracle on an assoc stream
- **WHEN** task accuracy is computed
- **THEN** every task's accuracy is 1.0

### Requirement: Retention matrix

The retention matrix `R[i][j]` MUST store the accuracy on task `i` probes measured in the probe block after task `j` finished, for all `j >= i`. [REQUIRED]

#### Scenario: perfect-memory retention is all ones

- **GIVEN** the perfect-memory oracle on a split-classify stream
- **WHEN** the retention matrix is computed
- **THEN** every entry is 1.0

#### Scenario: task-wiper shows diagonal ones and below-diagonal at chance

- **GIVEN** the task-wiper oracle on a split-classify stream with 3 tasks
- **WHEN** the retention matrix is computed
- **THEN** `R[i][i]` is 1.0 for all i, and `R[i][j]` for j > i is near chance

### Requirement: Backward transfer

Backward transfer for task `i` MUST be computed as the change in task `i` accuracy caused by learning later tasks, derived from the retention matrix. A negative value means forgetting. [REQUIRED]

#### Scenario: task-wiper backward transfer is strongly negative

- **GIVEN** the task-wiper oracle on a split-classify stream
- **WHEN** backward transfer is computed for task 0
- **THEN** the value is strongly negative (accuracy dropped from 1.0 to chance)

#### Scenario: perfect-memory backward transfer is zero

- **GIVEN** the perfect-memory oracle
- **WHEN** backward transfer is computed
- **THEN** it is 0.0 for every task

### Requirement: Forward transfer

Forward transfer for task `i` MUST be the accuracy on task `i` probes before task `i` was taught, compared against the chance baseline for that task. [REQUIRED]

#### Scenario: chance oracle forward transfer is zero

- **GIVEN** the chance oracle on any stream
- **WHEN** forward transfer is computed
- **THEN** it is approximately 0.0 (at chance both before and after teaching)

### Requirement: Compute per input

Three separate numbers MUST be reported: inner-loop steps, estimated FLOPs, and wall time. Each MUST be derived from `CostCounters` differences between consecutive probes. [REQUIRED]

#### Scenario: three distinct counters

- **GIVEN** a completed run
- **WHEN** compute-per-input is reported
- **THEN** three values are present and independently computed

### Requirement: Time to first use

Time to first use MUST be the gap in stream positions between teaching a fact and the first probe the subject answers correctly. A fact never correctly recalled MUST receive a configurable cut-off value. The report MUST show the median and the share of facts that hit the cut-off. [REQUIRED]

#### Scenario: perfect-memory oracle first use equals distance to next probe

- **GIVEN** the perfect-memory oracle on an assoc stream
- **WHEN** time to first use is computed
- **THEN** it equals the distance from teaching position to the first probe position for each fact

#### Scenario: cut-off share is zero for perfect memory

- **GIVEN** the perfect-memory oracle
- **WHEN** the cut-off share is computed
- **THEN** it is 0.0

### Requirement: Oracle metric validation

Every metric MUST have an oracle test: a run of a known oracle subject whose expected metric value is derived on paper before the code runs. A mismatch between computed and expected values MUST fail the test. [REQUIRED]

#### Scenario: forgetful oracle retention drops to chance beyond window

- **GIVEN** the forgetful oracle on a stream with probes at distances > 1
- **WHEN** retention is computed for probes beyond the 1-event window
- **THEN** accuracy is at the chance rate within sampling error

