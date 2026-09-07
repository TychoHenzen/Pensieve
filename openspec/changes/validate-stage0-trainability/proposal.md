## Why

The stabilized seed-0 EGGROLL gate lowered language-model loss by 2.25% while separation retention fell by 12.94% and exact and first-token accuracy remained zero. This repeated loss-versus-behavior conflict means default-recipe recalibration is not yet justified; the project first needs bounded evidence that the Stage 0 objective and at least one update method can produce useful decoded behavior.

## What Changes

- Add a deterministic trainability investigation that starts every arm from the same fresh Stage 0 state and uses fixed training and held-out records.
- Prove or falsify the shared objective with a one-record gradient overfit probe before comparing optimizers.
- Compare no-update, gradient-only, and EGGROLL-only arms after equal consumed-example budgets at 0, 8, and 32 examples.
- Record predicted update direction, actual post-update loss, decoded task metrics, separation retention, parameter movement, elapsed time, and complete run identity.
- Classify the result as an objective failure, a method-specific failure, a shared loss-versus-behavior conflict, bounded trainability, or inconclusive evidence.
- **BREAKING**: Require method-specific bounded trainability evidence before proposing or running default-recipe recalibration. A passing investigation permits recalibration only; it does not authorize full training or replace the existing stability and five-seed acceptance gates.
- Retain failed and partial reports as first-class evidence. Do not weaken the existing separation threshold or select replacement defaults in this change.

## Capabilities

### New Capabilities

- `train/stage0-trainability-gate`: Defines the bounded overfit, causal-direction, and equal-budget method comparison probes; their reports and classifications; and method-specific eligibility for later recalibration.

### Modified Capabilities

None.

## Impact

- Training diagnostics: a new command and reusable runners around the existing fresh-state construction, gradient trainer, EGGROLL trainer, and stability evaluation.
- Reporting: append-only JSONL progress and one canonical final JSON report with immutable arm identity and explicit stop reasons.
- Workflow: default-recipe recalibration remains blocked until the requested method has bounded trainability evidence. Full EGGROLL, gradient, and alternating training remain governed by their existing acceptance gates.
- Tests: deterministic arm isolation, one-record overfit behavior, equal consumed-example accounting, causal metric capture, classification, early stopping, report validation, and recalibration rejection.
- Operations: the investigation must remain development-sized and must never load test records, resume a checkpoint, start full training, archive `stabilize-eggroll-training`, or choose production defaults.
