# eval/baselines Specification

## Purpose
Defines the five baseline continual-learning methods, the parameter-matching runner, and the real data binding for Split-MNIST, so every Stage -1 measurement has a comparison floor and ceiling.
## Requirements
### Requirement: Five baseline implementations

The harness MUST ship five baselines, each implementing the subject protocol: naive sequential (SGD on the stream as it arrives), joint training (all tasks at once, offline), frozen (no learning), EWC (elastic weight consolidation), and generative replay. [REQUIRED]

#### Scenario: each baseline is a valid subject

- **GIVEN** any of the five baselines
- **WHEN** it is checked against the subject protocol
- **THEN** it implements all required methods including snapshot-restore

### Requirement: Parameter matching

The baseline runner MUST count parameters of the subject and of each baseline. It MUST refuse to report a comparison when the parameter counts differ by more than a stated tolerance. The report MUST also record depth, width, and optimizer for each model. [REQUIRED]

#### Scenario: mismatched parameters rejected

- **GIVEN** a subject with 100k parameters and a baseline with 500k parameters, tolerance 10%
- **WHEN** a comparison is requested
- **THEN** the runner raises an error or returns a rejection

#### Scenario: report includes architecture metadata

- **GIVEN** a completed comparison
- **WHEN** the report is inspected
- **THEN** it includes depth, width, optimizer, and parameter count for both models

### Requirement: Naive sequential baseline

The naive baseline MUST train with plain SGD on each example as it arrives in stream order. It serves as the forgetting floor. [REQUIRED]

#### Scenario: catastrophic forgetting visible

- **GIVEN** a split-classify stream with 5 tasks
- **WHEN** the naive baseline runs
- **THEN** accuracy on early tasks drops sharply after later tasks are learned

### Requirement: Joint training baseline

The joint baseline MUST train on all tasks simultaneously in an offline pass. It serves as the retention ceiling. [REQUIRED]

#### Scenario: high accuracy on all tasks

- **GIVEN** a split-classify stream
- **WHEN** the joint baseline runs
- **THEN** accuracy is high on all tasks because it sees all data

### Requirement: Frozen baseline

The frozen baseline MUST perform no learning at all. It serves as the no-adaptation floor. [REQUIRED]

#### Scenario: accuracy at initialization level

- **GIVEN** a split-classify stream
- **WHEN** the frozen baseline runs
- **THEN** accuracy is at or near chance on all tasks

### Requirement: EWC baseline

The EWC baseline MUST implement elastic weight consolidation (Kirkpatrick et al., 2017), slowing learning on weights important to earlier tasks by adding a quadratic penalty from the Fisher information matrix. [REQUIRED]

#### Scenario: EWC collapses in class-incremental setting

- **GIVEN** a class-incremental split-classify stream
- **WHEN** EWC runs
- **THEN** its accuracy is near chance (this is the expected EWC failure mode for class-incremental)

### Requirement: Generative replay baseline

The generative replay baseline MUST train a generator alongside the solver, replaying generated examples from past tasks while learning new ones, following van de Ven et al. 2020. [REQUIRED]

#### Scenario: replay retains old task accuracy

- **GIVEN** a class-incremental split-classify stream
- **WHEN** generative replay runs
- **THEN** accuracy on old tasks stays substantially above chance

### Requirement: Split-MNIST data binding

The `split-classify` generator MUST accept a config option to bind real MNIST digit data through the existing `source` field, replacing synthetic features with actual pixel features. The stream identity (hash) MUST change when the data source changes. [REQUIRED]

#### Scenario: real MNIST features in payload

- **GIVEN** `split-classify` configured with MNIST binding
- **WHEN** an `Observe` payload is inspected
- **THEN** `features` is a 784-element list (28x28 pixels) and `source` names the MNIST item

#### Scenario: hash changes with data source

- **GIVEN** two split-classify streams, one synthetic and one MNIST-bound, same seed
- **WHEN** their stream hashes are compared
- **THEN** the hashes differ

