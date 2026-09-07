## MODIFIED Requirements

### Requirement: Average-variance hysteresis optimizer control

The training command SHALL start with Eggroll updates and SHALL select the active update method from the average shared post-loop slot variance over consecutive observation windows of consumed examples. The observation window SHALL default to 50 examples and SHALL be configurable as a positive integer through `--phase-steps`. The lower threshold SHALL default to `0.01`, and the upper threshold SHALL default to `0.02`. Both thresholds SHALL be configurable, finite, non-negative, and ordered so the lower threshold is less than the upper threshold. An Eggroll fitness batch SHALL be capped so it cannot cross the remaining example count in the current observation window. At a completed window, Eggroll SHALL switch to gradient when average variance is at least the upper threshold. Gradient SHALL switch to Eggroll when average variance is at most the lower threshold. The active method SHALL remain unchanged while its switching condition is false, including averages inside the hysteresis band.

#### Scenario: Eggroll restores variance

- **WHEN** a completed Eggroll window has average variance at or above the upper threshold
- **THEN** the next dataset example uses the gradient update method

#### Scenario: Gradient detects collapse

- **WHEN** a completed gradient window has average variance at or below the lower threshold
- **THEN** the next dataset example uses the Eggroll update method

#### Scenario: Hysteresis band retains the active method

- **WHEN** a completed window has average variance strictly between the thresholds
- **THEN** the next window uses the same update method

#### Scenario: Eggroll batch ends at the observation boundary

- **WHEN** the configured Eggroll fitness-batch size exceeds the examples remaining in the current observation window
- **THEN** the current Eggroll update consumes only those remaining examples and the controller decides the next method before another record is consumed

#### Scenario: Invalid variance controller

- **WHEN** the user supplies a window below one, a non-finite threshold, a negative lower threshold, or thresholds not ordered as lower less than upper
- **THEN** the command rejects the configuration before loading models or data

### Requirement: Complete multi-epoch dataset traversal

The training command SHALL consume every selected training example exactly once per configured epoch. Eggroll fitness-batch boundaries, observation-window boundaries, optimizer switches, and logging boundaries SHALL preserve the dataset cursor and SHALL NOT restart, skip, or repeat examples. Gradient SHALL consume one example per optimizer call, while Eggroll SHALL consume one or more examples per optimizer call. The epoch count SHALL default to five and SHALL be configurable as a positive integer. Global progress SHALL count consumed examples rather than optimizer calls.

#### Scenario: Observation window ends within an epoch

- **WHEN** an observation window completes before the current epoch ends
- **THEN** the controller continues with the next example in that epoch using the selected method

#### Scenario: Epoch boundary within an observation window

- **WHEN** an epoch ends before the current observation window is complete
- **THEN** the next epoch preserves the active method, completed window example count, and accumulated variance sum

#### Scenario: Full default run

- **WHEN** the command runs with five epochs over the 570-example Calc-ASDiv_A training partition
- **THEN** it consumes exactly 2,850 ordered training examples while recording separate gradient and Eggroll optimizer-call counts

#### Scenario: Invalid epoch count

- **WHEN** the user supplies an epoch count below one
- **THEN** the command rejects the configuration before loading models or data

### Requirement: Continuous model and optimizer state

Both update methods SHALL operate on one shared set of trainable model parameters. Gradient SHALL retain Adam state over the complete allowed trainable registry. Eggroll SHALL retain momentum-free SGD state over only the declared matrix parameter scope. Switching methods SHALL NOT reinitialize model parameters or either optimizer. An inactive optimizer's parameter groups and state SHALL remain unchanged.

#### Scenario: Switch from Eggroll to gradient training

- **WHEN** the variance controller switches from Eggroll to gradient
- **THEN** gradient receives the shared parameter values produced by the final Eggroll step and restores its prior complete-registry Adam state

#### Scenario: Return to Eggroll

- **WHEN** the variance controller switches from gradient to Eggroll
- **THEN** Eggroll receives the shared matrix values produced by the final gradient step and restores its prior matrix-only SGD state

### Requirement: Comparable phase measurements

Every training update SHALL report mean language-model loss and one shared post-latent-loop slot variance definition over the examples consumed by that update. The reported variance SHALL use the same tensor, axes, estimator settings, and per-example aggregation in both update methods, followed by an arithmetic mean across an Eggroll fitness batch. Both update methods SHALL use the same bounded post-loop collapse penalty and configured variance weight per example. For each feature, the penalty is `max(0, 1 - sqrt(population_variance_across_slots + 1e-4))`, averaged across features. Gradient training SHALL minimize the one-example objective. Eggroll SHALL maximize the mean negative objective across its deterministic fitness batch. Candidate evaluation SHALL execute the exact latent-loop update equation without an extra normalization before a latent run.

#### Scenario: Compare phase metrics

- **WHEN** adjacent Eggroll and gradient updates emit metrics
- **THEN** their loss and variance fields have the same per-example meanings and units, and the Eggroll fields are arithmetic means across its consumed records

#### Scenario: Record experiment position

- **WHEN** a training update emits a progress record
- **THEN** the record identifies the update method, cycle number, consumed-example global position, optimizer-call count, epoch number, next example position, observation-window example count, and records consumed by that update

### Requirement: Resumable experiment checkpoints

The command SHALL save the version-2 JSON-plus-safetensors `.ckpt` container at every observation-window and epoch boundary. It SHALL contain shared trainable parameters, complete-registry gradient Adam state, matrix-only Eggroll SGD state, active method, completed window example count, observation-window size, partial-window variance sum and contributing-example count, consumed-example global position, method-specific optimizer-call counts, epoch, next dataset position, fixed per-epoch order identity, complete Stage 0 identity, fixed held-out identifiers, run configuration, and Python, NumPy, PyTorch CPU, every CUDA RNG state, and ordered CUDA device identities. The run configuration SHALL record the complete typed dataset selection, epoch target, observation-window size, lower and upper variance thresholds, slot count, latent-step count, both optimizer types and learning rates, Eggroll parameter scope, population size, sigma, rank, fitness-batch size, variance weight, evaluation batch size, AMP setting, held-out selection, compatible stability-report identity, and logging frequency. Resume SHALL use the safe loading contract and MUST reject a changed CUDA device count or ordering during metadata validation before model construction. Resume MAY change logging frequency and MAY increase the epoch target, but the epoch target MUST NOT be below completed progress. Every other run-configuration mismatch SHALL be rejected with its canonical field path before model construction or state application.

#### Scenario: Resume within an epoch

- **WHEN** the command resumes from a compatible observation-window checkpoint created within an epoch
- **THEN** it continues with the next unprocessed example and the method dictated by the saved controller state without repeating any member of the preceding Eggroll batch

#### Scenario: Resume at an epoch boundary

- **WHEN** the command resumes from a compatible epoch-boundary checkpoint
- **THEN** it starts the next epoch without repeating the completed epoch and preserves any unfinished variance window

#### Scenario: Reject incompatible resume configuration

- **WHEN** resume arguments or loaded data change a schedule-defining setting, dataset identity, backbone identity, model shape, tap layer, optimizer type, Eggroll parameter scope, fitness-batch size, stability-report identity, or held-out item identity stored in the checkpoint
- **THEN** the command rejects the resume before applying another training update and identifies every incompatible or missing setting
