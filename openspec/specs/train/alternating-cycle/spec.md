# train/alternating-cycle Specification

## Purpose
Defines a reproducible multi-epoch experiment that alternates Eggroll and gradient updates on one continuous dataset traversal while exposing comparable representation and task metrics.
## Requirements
### Requirement: Average-variance hysteresis optimizer control
The training command SHALL start with Eggroll updates and SHALL select the active update method from the average shared post-loop slot variance over consecutive observation windows. The observation window SHALL default to 50 examples and SHALL be configurable as a positive integer through `--phase-steps`. The lower threshold SHALL default to `0.01`, and the upper threshold SHALL default to `0.02`. Both thresholds SHALL be configurable, finite, non-negative, and ordered so the lower threshold is less than the upper threshold. At a completed window, Eggroll SHALL switch to gradient when average variance is at least the upper threshold. Gradient SHALL switch to Eggroll when average variance is at most the lower threshold. The active method SHALL remain unchanged while its switching condition is false, including averages inside the hysteresis band.

#### Scenario: Eggroll restores variance
- **WHEN** a completed Eggroll window has average variance at or above the upper threshold
- **THEN** the next dataset example uses the gradient update method

#### Scenario: Gradient detects collapse
- **WHEN** a completed gradient window has average variance at or below the lower threshold
- **THEN** the next dataset example uses the Eggroll update method

#### Scenario: Hysteresis band retains the active method
- **WHEN** a completed window has average variance strictly between the thresholds
- **THEN** the next window uses the same update method

#### Scenario: Invalid variance controller
- **WHEN** the user supplies a window below one, a non-finite threshold, a negative lower threshold, or thresholds not ordered as lower less than upper
- **THEN** the command rejects the configuration before loading models or data

### Requirement: Complete multi-epoch dataset traversal
The training command SHALL process every selected training example exactly once per configured epoch. Observation-window boundaries and optimizer switches SHALL preserve the dataset cursor and SHALL NOT restart, skip, or repeat examples. The epoch count SHALL default to five and SHALL be configurable as a positive integer.

#### Scenario: Observation window ends within an epoch
- **WHEN** an observation window completes before the current epoch ends
- **THEN** the controller continues with the next example in that epoch using the selected method

#### Scenario: Epoch boundary within an observation window
- **WHEN** an epoch ends before the current observation window is complete
- **THEN** the next epoch preserves the active method, completed window steps, and accumulated variance sum

#### Scenario: Full default run
- **WHEN** the command runs with five epochs over the 570-example Calc-ASDiv_A training partition
- **THEN** it performs exactly 2,850 training steps without resetting the variance controller at epoch boundaries

#### Scenario: Invalid epoch count
- **WHEN** the user supplies an epoch count below one
- **THEN** the command rejects the configuration before loading models or data

### Requirement: Continuous model and optimizer state
Both update methods SHALL operate on one shared set of trainable model parameters. Each method SHALL retain its own optimizer state while inactive, and switching methods SHALL NOT reinitialize model parameters or either optimizer.

#### Scenario: Switch from Eggroll to gradient training
- **WHEN** the variance controller switches from Eggroll to gradient
- **THEN** gradient receives the parameter values produced by the final Eggroll step and restores its prior optimizer state

#### Scenario: Return to Eggroll
- **WHEN** the variance controller switches from gradient to Eggroll
- **THEN** Eggroll receives the parameter values produced by the final gradient step and restores its prior optimizer state

### Requirement: Comparable phase measurements
Every training step SHALL report language-model loss and one shared post-latent-loop slot variance definition. The reported variance SHALL use the same tensor, axes, estimator settings, and aggregation in both update methods.

Both update methods SHALL use the same bounded post-loop collapse penalty and configured variance weight. For each feature, the penalty is `max(0, 1 - sqrt(population_variance_across_slots + 1e-4))`, averaged across features. Gradient training SHALL minimize language-model loss plus the weighted penalty. Eggroll SHALL maximize the negative of that same objective. Candidate evaluation SHALL execute the exact latent-loop update equation without an extra normalization before a latent run.

#### Scenario: Compare phase metrics
- **WHEN** adjacent Eggroll and gradient steps emit metrics
- **THEN** their loss and variance fields have the same meanings and units

#### Scenario: Record experiment position
- **WHEN** a training step emits a progress record
- **THEN** the record identifies the update method, cycle number, global step, epoch number, example position, and phase-step position

### Requirement: Observation-window and epoch evaluation
The experiment SHALL evaluate the unperturbed shared model without parameter updates at every completed observation window and epoch. Evaluation SHALL use a fixed held-out problem set selected once for the run and SHALL report mean language-model loss, shared slot variance, and answer exact match.

#### Scenario: Completed observation-window evaluation
- **WHEN** an observation window completes
- **THEN** the command evaluates the current unperturbed model on the fixed held-out problems and labels the result with the method that produced it

#### Scenario: Partial final observation-window evaluation
- **WHEN** training ends before the current observation window completes
- **THEN** the command evaluates the final model and labels the result as a partial phase

#### Scenario: Evaluation isolation
- **WHEN** held-out evaluation runs
- **THEN** model parameters, optimizer states, dataset position, and variance-controller state remain unchanged

### Requirement: Resumable experiment checkpoints
The command SHALL save the version-2 JSON-plus-safetensors `.ckpt` container at every observation-window and epoch boundary. It SHALL contain shared trainable parameters, separate Eggroll and gradient optimizer states and parameter groups, active method, completed window steps, observation-window size, partial-window variance sum, global step, epoch, next dataset position, fixed per-epoch order identity, complete Stage 0 identity, fixed held-out identifiers, run configuration, and Python, NumPy, PyTorch CPU, every CUDA RNG state, and ordered CUDA device identities. The run configuration SHALL record the complete typed dataset selection, epoch target, observation-window size, lower and upper variance thresholds, slot count, latent-step count, both optimizer learning rates, Eggroll population size, sigma, rank, variance weight, evaluation batch size, AMP setting, held-out selection, and logging frequency. The two optimizers SHALL retain independent state over the same declared shared parameters; each step SHALL clear the active optimizer's gradients before its update. Resume SHALL use the safe loading contract and MUST reject a changed CUDA device count or ordering during metadata validation before model construction. Resume MAY change logging frequency and MAY increase the epoch target, but the epoch target MUST NOT be below completed progress. Every other run-configuration mismatch SHALL be rejected with its canonical field path before model construction or state application. A checkpoint inside an unfinished observation window MUST contain its partial-window variance sum.

#### Scenario: Resume within an epoch
- **WHEN** the command resumes from a compatible observation-window checkpoint created within an epoch
- **THEN** it continues with the next unprocessed example and the method dictated by the saved controller state

#### Scenario: Resume at an epoch boundary
- **WHEN** the command resumes from a compatible epoch-boundary checkpoint
- **THEN** it starts the next epoch without repeating the completed epoch and preserves any unfinished variance window

#### Scenario: Reject incompatible resume configuration
- **WHEN** resume arguments or loaded data change a schedule-defining setting, dataset identity, backbone identity, model shape, tap layer, or held-out item identity stored in the checkpoint
- **THEN** the command rejects the resume before applying another training update and identifies every incompatible or missing setting

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

### Requirement: Portable training graph export
The graph command SHALL read structured `training` and `evaluation` records from mixed training logs encoded as UTF-8 or UTF-16. It SHALL ignore human-readable status, stderr, and checkpoint records. It SHALL write a conventional CSV plus one self-contained HTML file that requires no Python plotting package, external viewer, or network resource. The graph SHALL plot language-model loss, shared slot variance on a logarithmic scale with both hysteresis thresholds, and held-out exact match with an optional baseline reference. Training samples SHALL identify Eggroll and gradient updates without connecting a line across an intervening optimizer phase.

#### Scenario: Convert a PowerShell training log
- **WHEN** `python -m train.plot_training` receives a mixed UTF-16 training log
- **THEN** it writes CSV and HTML outputs from every valid training and evaluation record

#### Scenario: Reject a log without metrics
- **WHEN** the input contains no valid training or evaluation record
- **THEN** the graph command fails without writing an empty or misleading graph
