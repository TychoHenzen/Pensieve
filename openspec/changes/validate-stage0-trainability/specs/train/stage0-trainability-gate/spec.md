## Purpose

Defines a bounded, reproducible investigation that distinguishes an untrainable Stage 0 objective from optimizer-specific failure before any recipe recalibration or full training run.

## ADDED Requirements

### Requirement: Investigation uses compatible fresh-state evidence

The repository MUST provide a Stage 0 trainability command that accepts the retained failed EGGROLL stability report, validates its complete asset identity, implementation digest, evaluation selection, and EGGROLL configuration, and starts every probe from a separately constructed deterministic seed-0 state. It MUST use only the first training record for the overfit probe, the first 32 ordered training records for method comparison, and the same first 64 training-held-out records recorded by the stability report. It MUST NOT load a test record, resume a checkpoint, or mutate the supplied stability report.

#### Scenario: Compatible failed report starts investigation

- **WHEN** the command receives the retained failed stability report whose identity and source digest match the current investigation
- **THEN** it records the report digest and constructs each probe from an independent fresh state before applying any update

#### Scenario: Stale or passing report is rejected

- **WHEN** the supplied report is missing, passing, malformed, or differs in any guarded identity, implementation, selection, or EGGROLL configuration field
- **THEN** the command exits unsuccessfully before model construction or dataset access and names every incompatible canonical field path

#### Scenario: Test and checkpoint inputs are unavailable

- **WHEN** the investigation loads its records and initializes its probes
- **THEN** it receives no Stage 0 test record and has no checkpoint or resume input

### Requirement: One-record overfit probe falsifies the shared objective first

The command MUST run gradient-only one-record memorization attempts at learning rates `0.0001`, `0.001`, and `0.01`. Each attempt MUST start from the same fresh parameter and RNG state, repeatedly use the fixed first training record, and stop after at most 64 optimizer calls. It MUST evaluate the training record at 0, 1, 4, 16, and 64 calls unless it passes or encounters a non-finite value earlier. An attempt passes only when decoded exact accuracy and first-token accuracy are both `1.0`, every value is finite, and teacher-forced language-model loss is at most `0.5` times that attempt's baseline. These diagnostic learning rates MUST NOT become production defaults or recommendations.

#### Scenario: Objective demonstrates memorization

- **WHEN** any declared learning rate meets every overfit condition by its final completed checkpoint
- **THEN** the shared objective is marked trainable and the causal and equal-budget method probes may run

#### Scenario: Objective fails bounded memorization

- **WHEN** no declared learning rate meets every overfit condition within 64 optimizer calls
- **THEN** the result is classified `objective_untrainable`, later method probes do not run, and every completed attempt remains in the report

#### Scenario: One learning rate becomes non-finite

- **WHEN** an overfit attempt emits a non-finite objective, metric, parameter, gradient, or optimizer value
- **THEN** that attempt stops without modifying another attempt's initial state and the remaining declared learning rates continue

### Requirement: Causal probes compare predicted and actual update direction

For gradient and EGGROLL independently, the command MUST evaluate the method's complete declared training objective on the fixed first eight training records, construct one update from a fresh state, record the signed first-order predicted objective change, apply the update to that arm only, and evaluate the same objective and the fixed held-out records again. A causal probe passes only when the predicted objective change is negative, the observed post-update objective is below the pre-update objective, all values are finite, maximum relative matrix RMS change is at most `0.01`, and held-out separation retention is at least `0.95` times its fresh baseline.

#### Scenario: Method update has the predicted effect

- **WHEN** a method's proposed update satisfies every causal condition
- **THEN** the report marks that method's causal probe `passed` and records the predicted and observed deltas

#### Scenario: Update moves against its prediction

- **WHEN** a method predicts an objective decrease but its post-update objective does not decrease on the same support records
- **THEN** the report marks that method's causal probe `direction_mismatch` and makes that method ineligible for recipe recalibration

#### Scenario: Causal update damages held-out separation

- **WHEN** a method lowers its support objective but held-out separation falls below `0.95` times baseline or matrix movement exceeds `0.01`
- **THEN** the report marks that causal probe `unsafe_update` and preserves the conflicting objective and behavior metrics

### Requirement: Equal-budget arms isolate update-method behavior

After the overfit probe passes, the command MUST run no-update, gradient-only, and EGGROLL-only arms from identical fresh parameter and RNG states over the same first 32 ordered training records. The gradient arm MUST consume one record per optimizer call. The EGGROLL arm MUST consume deterministic fitness batches without crossing the 8- or 32-record checkpoint. The no-update arm MUST advance the same record cursor without changing model or optimizer state. Every arm MUST evaluate the same held-out records at 0, 8, and 32 consumed examples and record the stability metrics, complete training objective, maximum relative matrix RMS change, optimizer-call count, elapsed time, and ETA.

#### Scenario: Arms receive equal evidence

- **WHEN** all three arms complete the 32-example budget
- **THEN** their records contain the same ordered training item identifiers and held-out item identifiers at each checkpoint

#### Scenario: No-update control detects drift

- **WHEN** the no-update arm reaches its 8- and 32-example checkpoints
- **THEN** its model tensors, optimizer state, RNG state, and evaluation metrics equal its zero-example baseline

#### Scenario: Unsafe arm stops independently

- **WHEN** an updated arm emits a non-finite value, exceeds the `0.01` relative matrix RMS ceiling, or falls below `0.95` baseline separation at a checkpoint
- **THEN** that arm stops, records its exact failure, and does not prevent the other arms from completing

### Requirement: Method findings require decoded improvement

An updated method arm MUST be classified `viable` only when its causal probe passed and its final completed 32-example checkpoint has language-model loss below baseline, exact accuracy above baseline, first-token accuracy above baseline, separation retention at least `0.95` times baseline, and maximum relative matrix RMS change at most `0.01`. A finite arm that lowers language-model loss but fails decoded improvement or separation MUST be classified `loss_behavior_conflict`. Other failed arms MUST be classified with the specific statuses `direction_mismatch`, `unsafe_update`, `no_improvement`, or `non_finite`.

#### Scenario: Method shows bounded trainability

- **WHEN** a method meets every causal and 32-example viability condition
- **THEN** the report marks that method `viable` without claiming full-training acceptance

#### Scenario: Loss improves without useful behavior

- **WHEN** a method lowers language-model loss but exact or first-token accuracy does not improve or separation falls below its floor
- **THEN** the report marks that method `loss_behavior_conflict` and does not reinterpret the loss reduction as task progress

### Requirement: Overall classification follows a deterministic precedence

The final report MUST assign exactly one overall classification using this precedence: `inconclusive` for control drift, incomplete required evidence, or infrastructure failure; `objective_untrainable` when the overfit ladder completes without a pass; `bounded_trainability_observed` when at least one updated method is viable; `shared_loss_behavior_conflict` when neither method is viable and both finite methods lower language-model loss while failing decoded improvement or separation; otherwise `method_specific_failure`. It MUST include the independent status and failed conditions for every requested probe and arm.

#### Scenario: At least one method is viable

- **WHEN** one or both updated method arms are classified `viable` and no inconclusive condition exists
- **THEN** the overall classification is `bounded_trainability_observed`

#### Scenario: Both methods repeat the observed conflict

- **WHEN** both finite updated arms lower language-model loss but neither is viable because decoded behavior or separation fails
- **THEN** the overall classification is `shared_loss_behavior_conflict`

#### Scenario: Control drift invalidates interpretation

- **WHEN** the no-update control changes state or evaluation results
- **THEN** the overall classification is `inconclusive` regardless of updated-arm metrics

### Requirement: Reports preserve complete bounded evidence

The command MUST write append-only JSONL progress after every overfit checkpoint, causal probe, arm update, arm evaluation, early stop, and classification decision. It MUST write one canonical final JSON report without overwriting an existing path. The report MUST contain schema version, supplied stability-report digest, complete asset and runtime identity, implementation digest, fixed record identifiers, initial-state digest, every immutable probe configuration, all completed metrics and deltas, stop reasons, per-method status, overall classification, elapsed time, and observed ETA. Failure or interruption MUST leave readable partial JSONL evidence.

#### Scenario: Failed investigation retains evidence

- **WHEN** any probe fails an acceptance condition
- **THEN** the command exits nonzero after writing its final classification and retains all prior progress records

#### Scenario: Existing output is protected

- **WHEN** either requested output path already exists
- **THEN** the command rejects the invocation before model construction, dataset access, or file modification

### Requirement: Recalibration eligibility is method-specific and limited

The report MUST mark a method `recalibration_eligible` only when the shared overfit probe passed and that method's causal probe passed. A recalibration proposal or command MUST cite a compatible report and MUST limit its search to methods marked eligible. Trainability evidence MUST NOT authorize a full training run, satisfy the EGGROLL stability gate, satisfy the five-seed Stage 0 gate, weaken any existing threshold, select a replacement production default, or archive another OpenSpec change.

#### Scenario: Causally supported method may be recalibrated

- **WHEN** the shared overfit probe and a requested method's causal probe both passed under a compatible identity
- **THEN** that method is marked `recalibration_eligible` while full-training acceptance remains false

#### Scenario: Unsupported method remains blocked

- **WHEN** the objective overfit probe or the requested method's causal probe did not pass
- **THEN** that method is marked ineligible and the report names the failed prerequisite

#### Scenario: Trainability report is not a training gate

- **WHEN** a full gradient, EGGROLL, or alternating run receives only a trainability report
- **THEN** the run remains subject to its existing stability and five-seed acceptance requirements
