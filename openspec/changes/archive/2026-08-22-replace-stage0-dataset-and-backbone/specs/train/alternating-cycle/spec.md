## MODIFIED Requirements

### Requirement: Complete multi-epoch dataset traversal
The training command SHALL process every selected training example exactly once per configured epoch. Phase transitions SHALL preserve the dataset cursor and SHALL NOT restart, skip, or repeat examples. The epoch count SHALL default to five and SHALL be configurable as a positive integer.

#### Scenario: Phase boundary within an epoch
- **WHEN** a phase budget expires before the current epoch ends
- **THEN** the next phase continues with the next example in that epoch

#### Scenario: Epoch boundary within a phase
- **WHEN** an epoch ends before the current 500-step phase budget is exhausted
- **THEN** the next epoch starts in the same phase with its remaining phase budget

#### Scenario: Full default run
- **WHEN** the command runs with five epochs over the 1,089-example filtered Calc-MAWPS training split
- **THEN** it performs exactly 5,445 training steps without changing the phase schedule at epoch boundaries

#### Scenario: Invalid epoch count
- **WHEN** the user supplies an epoch count below one
- **THEN** the command rejects the configuration before loading models or data

### Requirement: Resumable experiment checkpoints
The command SHALL save the version-2 JSON-plus-safetensors `.ckpt` container at every phase and epoch boundary. It SHALL contain shared trainable parameters, separate Eggroll and gradient optimizer states and parameter groups, active phase, completed phase steps, phase budget, global step, epoch, next dataset position, fixed per-epoch order identity, complete Stage 0 identity, fixed held-out identifiers, run configuration, and Python, NumPy, PyTorch CPU, every CUDA RNG state, and ordered CUDA device identities. The run configuration SHALL record the complete typed dataset selection, epoch target, phase budget, slot count, latent-step count, both optimizer learning rates, Eggroll population size, sigma, rank, variance weight, evaluation batch size, AMP setting, held-out selection, and logging frequency. The two optimizers SHALL retain independent state over the same declared shared parameters; each step SHALL clear the active optimizer's gradients before its update. Resume SHALL use the safe loading contract and MUST reject a changed CUDA device count or ordering during metadata validation before model construction. Resume MAY change logging frequency and MAY increase the epoch target, but the epoch target MUST NOT be below completed progress. Every other run-configuration mismatch SHALL be rejected with its canonical field path before model construction or state application.

#### Scenario: Resume within an epoch
- **WHEN** the command resumes from a compatible phase-boundary checkpoint created within an epoch
- **THEN** it continues with the next unprocessed example and the phase dictated by the saved schedule

#### Scenario: Resume at an epoch boundary
- **WHEN** the command resumes from a compatible epoch-boundary checkpoint
- **THEN** it starts the next epoch without repeating the completed epoch and preserves any unfinished phase budget

#### Scenario: Reject incompatible resume configuration
- **WHEN** resume arguments or loaded data change a schedule-defining setting, dataset identity, backbone identity, model shape, tap layer, or held-out item identity stored in the checkpoint
- **THEN** the command rejects the resume before applying another training update and identifies every incompatible or missing setting
