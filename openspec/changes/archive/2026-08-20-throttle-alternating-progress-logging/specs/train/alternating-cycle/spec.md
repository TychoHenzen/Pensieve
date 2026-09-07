## ADDED Requirements

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
