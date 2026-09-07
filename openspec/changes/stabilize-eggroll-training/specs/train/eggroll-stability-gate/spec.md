## Purpose

Defines a bounded, reproducible health gate that prevents unstable EGGROLL configurations from reaching alignment selection or full Stage 0 training.

## ADDED Requirements

### Requirement: Stability gate measures a fresh absolute baseline

The repository MUST provide an EGGROLL stability-gate command that creates a fresh deterministic Stage 0 state and evaluates the fixed first 64 records of the training-held-out partition before any update. It MUST record complete asset identity, implementation digest, EGGROLL configuration, parameter RMS values, language-model loss, exact accuracy, first-token accuracy, valid-answer rate, output diversity, output dominance, shared slot variance, student-to-teacher MSE, cross-problem student cosine, and separation retention. It MUST NOT read test records or a prior training checkpoint.

#### Scenario: Record the untrained baseline

- **WHEN** the stability gate starts with a valid configuration
- **THEN** it writes one immutable baseline record at zero consumed examples before applying an optimizer update

#### Scenario: Baseline evaluation is isolated

- **WHEN** the zero-example evaluation completes
- **THEN** model tensors, optimizer state, workspace, dataset cursor, and every RNG state equal their pre-evaluation values

### Requirement: Stability gate runs bounded development checkpoints

The stability gate MUST train from the measured fresh state over the first 256 ordered training records. It MUST evaluate the same held-out records after 8, 32, and 256 consumed examples. Each evaluation record MUST include the baseline metrics, current metrics, deltas, maximum per-update relative matrix RMS change, consumed-example count, EGGROLL optimizer-call count, elapsed time, and ETA. It MUST stop before 256 examples when language-model loss exceeds `1.05` times baseline or separation retention falls below `0.95` times baseline at a scheduled checkpoint.

#### Scenario: Emit every healthy development checkpoint

- **WHEN** no early-stop condition occurs
- **THEN** the report contains ordered evaluation records for 0, 8, 32, and 256 consumed examples

#### Scenario: Stop on absolute regression

- **WHEN** a scheduled checkpoint exceeds the loss ceiling or falls below the separation floor
- **THEN** the command stops further training, records the exact failed thresholds, exits nonzero, and retains the partial report

### Requirement: Passing health requires task improvement and bounded updates

A stability report MUST pass only when its final checkpoint has language-model loss below baseline, exact accuracy above baseline, first-token accuracy above baseline, separation retention at least `0.95` times baseline, and maximum per-update relative matrix RMS change at most `0.01`. The report MUST use status `passed` only when every condition holds. It MUST otherwise use status `failed` and list every failed condition.

#### Scenario: Configuration passes the absolute gate

- **WHEN** the 256-example checkpoint satisfies every health condition
- **THEN** the command exits zero and writes status `passed` with no failed condition

#### Scenario: Configuration lacks task improvement

- **WHEN** exact accuracy or first-token accuracy does not exceed its fresh baseline despite other metrics passing
- **THEN** the command exits nonzero and identifies the missing task improvement

### Requirement: Full EGGROLL workflows require a compatible passing report

Alignment-weight search, standalone EGGROLL training beyond 256 consumed examples, and alternating training MUST require a passing stability report whose asset identity, implementation digest, optimizer contract, parameter scope, population, sigma, rank, fitness-batch size, evaluation batch size, variance weight, and initialization seed match the requested run. A missing, failed, stale, or mismatched report MUST be rejected before model construction or dataset access.

#### Scenario: Start with a compatible passing report

- **WHEN** a guarded EGGROLL workflow receives a passing report with an exact identity and configuration match
- **THEN** it records that report identity and may proceed

#### Scenario: Reject a mismatched report

- **WHEN** any guarded field differs between the report and requested run
- **THEN** the workflow exits unsuccessfully before model construction and names every mismatched canonical field path

### Requirement: Alignment search rejects an unhealthy zero-weight control

Alignment-weight search MUST run or load the absolute stability result for its zero-alignment EGGROLL control before evaluating positive alignment weights. If that control is unhealthy, the EGGROLL method result MUST use status `method_unhealthy`, MUST contain no recommended weight, and MUST preserve the failed stability evidence. Gradient alignment search MAY continue independently.

#### Scenario: Eggroll zero control is unhealthy

- **WHEN** the zero-alignment EGGROLL control fails any absolute health condition
- **THEN** no positive EGGROLL weight trial runs and the final EGGROLL search result reports `method_unhealthy` with no recommendation

#### Scenario: Gradient search remains independent

- **WHEN** EGGROLL is unhealthy and the requested search includes both methods
- **THEN** the gradient search continues without using EGGROLL metrics, state, or eligibility
