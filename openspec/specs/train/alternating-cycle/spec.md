# train/alternating-cycle Specification

## Purpose
Defines a reproducible multi-epoch experiment that alternates Eggroll and gradient updates on one continuous dataset traversal while exposing comparable representation and task metrics.
## Requirements
### Requirement: Fixed-budget optimizer phases
The training command SHALL start in the Eggroll phase and SHALL alternate between Eggroll and gradient phases after a fixed number of training examples. The phase budget SHALL default to 500 examples and SHALL be configurable as a positive integer.

#### Scenario: Default phase transition
- **WHEN** a new run reaches 500 completed Eggroll training steps
- **THEN** step 501 uses the gradient update method on the next dataset example

#### Scenario: Repeated phase transition
- **WHEN** the run completes 500 gradient training steps after an Eggroll phase
- **THEN** the next dataset example starts a new Eggroll phase

#### Scenario: Invalid phase budget
- **WHEN** the user supplies a phase budget below one
- **THEN** the command rejects the configuration before loading models or data

### Requirement: Complete multi-epoch dataset traversal
The training command SHALL process every selected training example exactly once per configured epoch. Phase transitions SHALL preserve the dataset cursor and SHALL NOT restart, skip, or repeat examples. The epoch count SHALL default to five and SHALL be configurable as a positive integer.

#### Scenario: Phase boundary within an epoch
- **WHEN** a phase budget expires before the current epoch ends
- **THEN** the next phase continues with the next example in that epoch

#### Scenario: Epoch boundary within a phase
- **WHEN** an epoch ends before the current 500-step phase budget is exhausted
- **THEN** the next epoch starts in the same phase with its remaining phase budget

#### Scenario: Full default run
- **WHEN** the command runs with five epochs over a dataset containing 7,473 examples
- **THEN** it performs exactly 37,365 training steps without changing the phase schedule at epoch boundaries

#### Scenario: Invalid epoch count
- **WHEN** the user supplies an epoch count below one
- **THEN** the command rejects the configuration before loading models or data

### Requirement: Continuous model and optimizer state
Both update methods SHALL operate on one shared set of trainable model parameters. Each method SHALL retain its own optimizer state across inactive phases, and switching phases SHALL NOT reinitialize model parameters or either optimizer.

#### Scenario: Switch from Eggroll to gradient training
- **WHEN** an Eggroll phase ends
- **THEN** the gradient phase receives the parameter values produced by the final Eggroll step and restores its prior optimizer state

#### Scenario: Return to Eggroll
- **WHEN** a gradient phase ends
- **THEN** the next Eggroll phase receives the parameter values produced by the final gradient step and restores its prior optimizer state

### Requirement: Comparable phase measurements
Every training step SHALL report language-model loss and one shared post-latent-loop slot variance definition. The reported variance SHALL use the same tensor, axes, estimator settings, and aggregation in both update methods.

#### Scenario: Compare phase metrics
- **WHEN** adjacent Eggroll and gradient steps emit metrics
- **THEN** their loss and variance fields have the same meanings and units

#### Scenario: Record experiment position
- **WHEN** a training step emits a progress record
- **THEN** the record identifies the update method, cycle number, global step, epoch number, example position, and phase-step position

### Requirement: Phase and epoch evaluation
The experiment SHALL evaluate the unperturbed shared model without parameter updates at every completed phase and epoch. Evaluation SHALL use a fixed held-out problem set selected once for the run and SHALL report mean language-model loss, shared slot variance, and answer exact match.

#### Scenario: Completed phase evaluation
- **WHEN** a 500-step phase completes
- **THEN** the command evaluates the current unperturbed model on the fixed held-out problems and labels the result with the phase that produced it

#### Scenario: Partial final phase evaluation
- **WHEN** training ends before the current phase reaches 500 steps
- **THEN** the command evaluates the final model and labels the result as a partial phase

#### Scenario: Evaluation isolation
- **WHEN** held-out evaluation runs
- **THEN** model parameters, optimizer states, dataset position, and phase position remain unchanged

### Requirement: Resumable experiment checkpoints
The command SHALL save a checkpoint at every phase and epoch boundary. A checkpoint SHALL contain the shared model parameters, both optimizer states, active phase, completed phase steps, global step, epoch position, dataset position, run configuration, and random-number-generator state needed to continue the same schedule.

#### Scenario: Resume within an epoch
- **WHEN** the command resumes from a phase-boundary checkpoint created within an epoch
- **THEN** it continues with the next unprocessed example and the phase dictated by the saved schedule

#### Scenario: Resume at an epoch boundary
- **WHEN** the command resumes from an epoch-boundary checkpoint
- **THEN** it starts the next epoch without repeating the completed epoch and preserves any unfinished phase budget

#### Scenario: Reject incompatible resume configuration
- **WHEN** resume arguments change a schedule-defining setting stored in the checkpoint
- **THEN** the command rejects the resume before applying another training update and identifies the incompatible setting

### Requirement: Standalone trainer compatibility
The existing standalone Eggroll and gradient training commands SHALL remain available with their current single-method behavior.

#### Scenario: Run standalone Eggroll training
- **WHEN** the existing Eggroll command is invoked
- **THEN** it trains only with Eggroll and does not invoke the alternating scheduler

#### Scenario: Run standalone gradient training
- **WHEN** the existing gradient command is invoked
- **THEN** it trains only with gradient updates and does not invoke the alternating scheduler

### Requirement: Throttled progress output
The command SHALL filter only structured JSON progress records identified by `record_type`. It SHALL emit `training` records at a configurable positive `--log-every` interval that defaults to 50 global steps. It SHALL emit `evaluation` and `checkpoint` records at epoch boundaries and SHALL emit neither record type at phase-only boundaries. This output policy SHALL NOT change when training, evaluation, or checkpoint operations execute.

#### Scenario: Default training progress interval
- **WHEN** the command runs without a `--log-every` override
- **THEN** it emits exactly one `training` record at each positive global-step multiple of 50 and emits no `training` record at other steps

#### Scenario: Configured training progress interval
- **WHEN** the command runs with `--log-every N` for a positive integer N accepted by the CLI parser
- **THEN** it emits exactly one `training` record at each positive global-step multiple of N and emits no `training` record at other steps

#### Scenario: Resumed training progress interval
- **WHEN** the command resumes from a valid checkpoint with a restored nonzero global step
- **THEN** the current invocation's `--log-every` value controls subsequent `training` records using the restored global-step count

#### Scenario: Final step outside the progress interval
- **WHEN** normal training completion occurs at a global step that is not a multiple of `--log-every`
- **THEN** the command emits no additional `training` record for that final step

#### Scenario: Invalid training progress interval
- **WHEN** the user supplies `--log-every` with zero, a negative integer, or a non-integer value
- **THEN** the command exits unsuccessfully with error text containing `--log-every` before model construction, dataset access, checkpoint loading, or structured progress output

#### Scenario: Epoch boundary progress records
- **WHEN** an epoch boundary completes without another evaluation boundary at the same global step
- **THEN** the command emits exactly one `evaluation` record containing the epoch and global step and one `checkpoint` record whose non-empty `paths` array names files saved by that boundary

#### Scenario: Phase-only boundary progress records
- **WHEN** a phase boundary completes without an epoch boundary
- **THEN** evaluation and checkpoint operations run but the command emits no `evaluation` or `checkpoint` record for that boundary

#### Scenario: Coincident phase and epoch boundary progress records
- **WHEN** a phase boundary and epoch boundary complete at the same global step
- **THEN** evaluation and checkpoint operations each run once, one `evaluation` record contains both `epoch` and `phase` in its `boundaries` array, and one `checkpoint` record contains both saved paths

#### Scenario: Partial final phase at an epoch boundary
- **WHEN** normal training completion produces a partial-phase and epoch boundary at the same global step
- **THEN** evaluation runs once, one `evaluation` record contains both `epoch` and `partial_phase` in its `boundaries` array, and one `checkpoint` record contains the epoch checkpoint path

#### Scenario: Non-progress output remains independent
- **WHEN** progress filtering suppresses a structured JSON record
- **THEN** it does not intercept stderr or writes that are not structured records with `record_type` equal to `training`, `evaluation`, or `checkpoint`

