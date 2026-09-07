## MODIFIED Requirements

### Requirement: Shared Stage 0 dataset contract

Standalone gradient, standalone Eggroll, and alternating training MUST load the same complete seed-0 selection of 570 normalized Calc-ASDiv_A training records. These records MUST be shuffled source positions 520 through 1,089 from the generator contract. Every epoch MUST replay that persisted order without reshuffling and MUST consume every record exactly once. A gradient optimizer call MUST consume the next one record. An EGGROLL optimizer call MUST consume the next deterministic fitness batch, defaulting to eight records, and every candidate MUST use the arithmetic mean of its per-record fitnesses. An EGGROLL batch MUST stop before crossing an epoch boundary, an alternating observation-window boundary, or a structured logging boundary. A final partial batch MUST remain valid. Held-out evaluation MUST use all 128 records from shuffled source positions 1,090 through 1,217, selected once and stored in the checkpoint. It MUST snapshot and restore wrapper mode, gradients, EMA state, workspace, optimizers, all RNGs, and schedule state so each is bitwise unchanged by evaluation. Training, fitness batching, and phase evaluation MUST NOT load or receive any of the first 520 shuffled source rows assigned to the test partition.

#### Scenario: Training modes share examples

- **WHEN** each Stage 0 training command uses the same selection arguments
- **THEN** each receives the same ordered normalized training examples and item identifiers

#### Scenario: Eggroll aggregates a deterministic fitness batch

- **WHEN** an EGGROLL optimizer call begins with at least the configured fitness-batch size remaining before every active boundary
- **THEN** every candidate is evaluated on the same next ordered records, its fitness is their arithmetic mean, and the dataset cursor advances by that record count after one update

#### Scenario: Eggroll caps a boundary batch

- **WHEN** fewer records remain before an epoch, observation-window, or logging boundary than the configured EGGROLL fitness-batch size
- **THEN** EGGROLL uses exactly the remaining positive number of records and does not skip, repeat, or read across that boundary

#### Scenario: Held-out split isolation

- **WHEN** phase or epoch evaluation runs during training
- **THEN** it uses only the fixed validation selection and no test item

## ADDED Requirements

### Requirement: Training methods own separate optimizer parameter scopes

Gradient training MUST use Adam over all ten allowed Stage 0 trainable parameter paths. EGGROLL MUST use momentum-free SGD over exactly `encoder.projection.weight`, `encoder.slot_queries`, and `latent_loop.projection.weight`. Alternating training MUST share the same underlying model tensors while retaining those method-specific optimizer types, parameter groups, and states. An inactive method MUST NOT change its optimizer state.

#### Scenario: Construct method-specific optimizers

- **WHEN** standalone or alternating Stage 0 trainers construct their optimizers
- **THEN** gradient owns Adam state for all allowed paths while EGGROLL owns SGD state only for the three declared matrix paths

#### Scenario: Eggroll leaves non-matrix trainables unchanged

- **WHEN** one or more EGGROLL updates complete without an intervening gradient update
- **THEN** every allowed non-matrix trainable tensor remains bitwise equal to its value before those EGGROLL updates

### Requirement: Stabilized EGGROLL identity is resumable and incompatible changes are rejected

Every Stage 0 run configuration containing EGGROLL MUST record optimizer type, momentum, learning rate, ordered optimizer parameter paths, population, sigma, rank, evaluation batch size, fitness-batch size, prompt-alignment weight, and stability-report identity. Schedule progress MUST count consumed training examples and MUST separately record EGGROLL optimizer-call count. Resume MUST reject an Adam-based EGGROLL optimizer, a changed EGGROLL parameter scope, a changed fitness-batch size, or a schedule that cannot account exactly for consumed examples before model construction or state application.

#### Scenario: Save stabilized Eggroll identity

- **WHEN** standalone or alternating EGGROLL training saves a checkpoint
- **THEN** its metadata identifies momentum-free SGD, the three ordered matrix paths, fitness-batch size, consumed-example position, EGGROLL optimizer-call count, and the compatible passing stability report

#### Scenario: Reject legacy Adam Eggroll checkpoint

- **WHEN** a checkpoint declares Adam for EGGROLL or contains EGGROLL optimizer state for a non-matrix path
- **THEN** resume rejects it before applying tensors, optimizer state, RNG state, cursor movement, or metrics
