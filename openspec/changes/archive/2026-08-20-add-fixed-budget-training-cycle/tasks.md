## 1. Shared Training State and Metrics

- [x] 1.1 Add tests for one canonical post-loop variance function on batched and unbatched slot tensors, then implement the shared metric with `unbiased=False` and identical slot-axis aggregation.
  <!-- covers: train/alternating-cycle :: Comparable phase measurements :: Compare phase metrics -->
  <!-- status: completed -->
- [x] 1.2 Add a shared model-state container with a named trainable-parameter registry, and test that both update engines reference the same tensor objects.
  <!-- status: completed -->
- [x] 1.3 Add separate gradient and Eggroll optimizer construction against the shared registry, and test that inactive optimizer state survives both phase directions without model reinitialization.
  <!-- covers: train/alternating-cycle :: Continuous model and optimizer state :: Switch from Eggroll to gradient training -->
  <!-- covers: train/alternating-cycle :: Continuous model and optimizer state :: Return to Eggroll -->
  <!-- status: completed -->
- [x] 1.4 Define common step and evaluation result records that separate language-model loss, total objective, regularizers, shared variance, and experiment position.
  <!-- covers: train/alternating-cycle :: Comparable phase measurements :: Record experiment position -->
  <!-- status: completed -->

## 2. Update Engine Refactor

- [x] 2.1 Refactor gradient training to accept shared model state, emit the common result, and preserve its current language-model plus VICReg objective.
  <!-- status: completed -->
- [x] 2.2 Refactor Eggroll training to accept shared model state, use the canonical variance in fitness and reporting, and emit the common result.
  <!-- status: completed -->
- [x] 2.3 Add regression tests proving the existing Eggroll command remains single-method after the refactor.
  <!-- covers: train/alternating-cycle :: Standalone trainer compatibility :: Run standalone Eggroll training -->
  <!-- status: completed -->
- [x] 2.4 Add regression tests proving the existing gradient command remains single-method after the refactor.
  <!-- covers: train/alternating-cycle :: Standalone trainer compatibility :: Run standalone gradient training -->
  <!-- status: completed -->

## 3. Fixed-Budget Multi-Epoch Scheduler

- [x] 3.1 Add scheduler configuration validation before model and dataset loading for positive phase budgets and epoch counts.
  <!-- covers: train/alternating-cycle :: Fixed-budget optimizer phases :: Invalid phase budget -->
  <!-- covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Invalid epoch count -->
  <!-- status: completed -->
- [x] 3.2 Add fake-engine tests for an Eggroll-first 500-step phase, the transition at step 501, and repeated alternation, then implement the fixed-budget scheduler.
  <!-- covers: train/alternating-cycle :: Fixed-budget optimizer phases :: Default phase transition -->
  <!-- covers: train/alternating-cycle :: Fixed-budget optimizer phases :: Repeated phase transition -->
  <!-- status: completed -->
- [x] 3.3 Add scheduler tests proving a phase boundary continues with the next example inside an epoch and does not reset the dataset cursor.
  <!-- covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Phase boundary within an epoch -->
  <!-- status: completed -->
- [x] 3.4 Add scheduler tests proving an epoch boundary preserves the active phase and remaining phase budget.
  <!-- covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Epoch boundary within a phase -->
  <!-- status: completed -->
- [x] 3.5 Add a count-only five-epoch test for 7,473 examples that produces exactly 37,365 unique epoch-position visits under the continuous phase schedule.
  <!-- covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Full default run -->
  <!-- status: completed -->

## 4. Held-Out Evaluation

- [x] 4.1 Add a deterministic GSM8K test-subset loader and an unperturbed evaluator for mean language-model loss, canonical variance, and generated-answer exact match.
  <!-- status: completed -->
- [x] 4.2 Add an isolation test proving evaluation leaves model parameters, both optimizer states, random states, dataset position, and phase position unchanged.
  <!-- covers: train/alternating-cycle :: Phase and epoch evaluation :: Evaluation isolation -->
  <!-- status: completed -->
- [x] 4.3 Wire evaluation to completed phase boundaries and label each result with the phase that produced it.
  <!-- covers: train/alternating-cycle :: Phase and epoch evaluation :: Completed phase evaluation -->
  <!-- status: completed -->
- [x] 4.4 Wire deduplicated epoch-boundary evaluation and final partial-phase evaluation with boundary labels.
  <!-- covers: train/alternating-cycle :: Phase and epoch evaluation :: Partial final phase evaluation -->
  <!-- status: completed -->

## 5. Checkpoint and Resume

- [x] 5.1 Add a versioned checkpoint payload containing named model state, both optimizers, schedule position, configuration, held-out selection, metrics, and Python and PyTorch random states.
  <!-- status: completed -->
- [x] 5.2 Add phase-boundary save and resume tests that continue with the next unprocessed example and saved phase.
  <!-- covers: train/alternating-cycle :: Resumable experiment checkpoints :: Resume within an epoch -->
  <!-- status: completed -->
- [x] 5.3 Add epoch-boundary save and resume tests that start the next epoch while preserving unfinished phase budget.
  <!-- covers: train/alternating-cycle :: Resumable experiment checkpoints :: Resume at an epoch boundary -->
  <!-- status: completed -->
- [x] 5.4 Add compatibility checks that reject schedule-defining resume overrides and identify each conflicting setting.
  <!-- covers: train/alternating-cycle :: Resumable experiment checkpoints :: Reject incompatible resume configuration -->
  <!-- status: completed -->
- [x] 5.5 Implement `latest` checkpoint discovery by highest stored global step and deduplicate coincident phase and epoch boundary payloads.
  <!-- status: completed -->

## 6. Alternating Experiment Command

- [x] 6.1 Add `train.run_alternating` with five epochs and 500 phase steps by default, distinct learning-rate flags, existing Eggroll controls, held-out size, logging, save directory, and resume options.
  <!-- status: completed -->
- [x] 6.2 Add structured progress output for update method, cycle, global step, epoch, example position, phase step, language-model loss, shared variance, evaluations, and checkpoint paths.
  <!-- status: completed -->
- [x] 6.3 Add a deterministic small-data integration test that crosses both optimizer phases, an epoch boundary, evaluation, checkpoint save, and resume without loading the full production models.
  <!-- status: completed -->
- [x] 6.4 Run the focused alternating-training tests, then run the complete pytest suite and record the exact five-epoch experiment command in the change handoff.
  <!-- status: completed -->
