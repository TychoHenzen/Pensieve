## 1. Pin the Official EGGROLL Estimator

- [x] 1.1 Add independent golden fixtures for B-first/A-second factors, antithetic signs, population-variance fitness normalization, pair scores, SGD ascent direction, and non-finite rejection. Confirm the normalization fixture fails against the current implementation before changing production code.
<!-- status: completed -->
  <!-- covers: train/eggroll-execution :: Matrix perturbations remain factorized during candidate evaluation :: Matrix candidate batch uses low-rank residuals -->
  <!-- covers: train/eggroll-execution :: Matrix perturbations remain factorized during candidate evaluation :: Antithetic factors share one seed -->
  <!-- covers: train/eggroll-execution :: Pseudo-gradient assembly aggregates low-rank factors :: Factorized matrix update matches reference update -->
  <!-- covers: train/eggroll-execution :: Pseudo-gradient assembly aggregates low-rank factors :: Update follows fitness ascent -->
  <!-- covers: train/eggroll-execution :: Pseudo-gradient assembly aggregates low-rank factors :: Non-finite candidate fitness -->
- [x] 1.2 Replace sample-standard-deviation normalization with the pinned population-variance formula while retaining factorized contraction and the one-dense-result memory bound.
<!-- status: completed -->
  <!-- covers: train/eggroll-execution :: Pseudo-gradient assembly aggregates low-rank factors :: Assembly has one dense matrix result -->
- [x] 1.3 Run the focused perturbation, factorized-kernel, update, and materialized-reference suites as an independently runnable estimator check.
<!-- status: completed -->

## 2. Separate Gradient and EGGROLL Optimizers

- [x] 2.1 Add an explicit ordered EGGROLL registry for the three declared matrix paths. Add tests proving gradient owns all ten paths and EGGROLL excludes vectors and scalars.
<!-- status: completed -->
  <!-- covers: train/stage0-training :: Training methods own separate optimizer parameter scopes :: Construct method-specific optimizers -->
  <!-- covers: train/eggroll-execution :: Matrix perturbations remain factorized during candidate evaluation :: Vector perturbation remains compatible -->
- [x] 2.2 Construct momentum-free SGD for EGGROLL and retain Adam for gradient training. Verify repeated EGGROLL updates leave every non-matrix tensor bitwise unchanged.
<!-- status: completed -->
  <!-- covers: train/stage0-training :: Training methods own separate optimizer parameter scopes :: Eggroll leaves non-matrix trainables unchanged -->
- [x] 2.3 Change Stage 0 EGGROLL defaults to population 128, candidate batch 8, rank 4, sigma 0.001, fitness batch 8, and SGD learning rate 0.1. Reject every invalid value before model or data loading.
<!-- status: completed -->
  <!-- covers: train/eggroll-execution :: EGGROLL uses safe Stage 0 defaults :: Default stabilized configuration -->
  <!-- covers: train/eggroll-execution :: EGGROLL uses safe Stage 0 defaults :: Invalid stabilized configuration -->
- [x] 2.4 Update optimized and materialized complete-step fixtures to compare matrix-only SGD state and untouched non-matrix tensors.
<!-- status: completed -->
  <!-- covers: train/eggroll-execution :: Optimized execution preserves EGGROLL training semantics :: One optimized step matches the reference step -->
  <!-- covers: train/eggroll-execution :: Optimized execution preserves EGGROLL training semantics :: Existing commands select the optimized path -->
- [x] 2.5 Run standalone one-step gradient and EGGROLL commands on fixtures to prove their optimizer types, parameter scopes, and defaults are independently usable.
<!-- status: completed -->

## 3. Add Deterministic Multi-Problem Fitness Batches

- [x] 3.1 Introduce a boundary-aware fitness-batch planner. Test ordered records, arithmetic candidate means, final partial batches, held-out isolation, and caps at epoch, observation-window, and logging boundaries.
<!-- status: completed -->
  <!-- covers: train/stage0-training :: Shared Stage 0 dataset contract :: Training modes share examples -->
  <!-- covers: train/stage0-training :: Shared Stage 0 dataset contract :: Eggroll aggregates a deterministic fitness batch -->
  <!-- covers: train/stage0-training :: Shared Stage 0 dataset contract :: Eggroll caps a boundary batch -->
  <!-- covers: train/stage0-training :: Shared Stage 0 dataset contract :: Held-out split isolation -->
- [x] 3.2 Refactor EGGROLL candidate evaluation to reuse one perturbation population across all problems, accumulate per-candidate metrics sequentially, release each problem's activations, and update once from mean fitness.
<!-- status: completed -->
- [x] 3.3 Extend step results and standalone progress with consumed-record count, next example position, aggregate loss and variance, and optimizer-call count.
<!-- status: completed -->
- [x] 3.4 Compare optimized and materialized multi-problem steps from identical states, including per-problem fitness, means, tensors, SGD state, metrics, and RNG.
<!-- status: completed -->
- [x] 3.5 Run one standalone Eggroll epoch on a small fixture whose record count is not divisible by eight. Assert each record appears exactly once and the final partial batch remains runnable.
<!-- status: completed -->

## 4. Integrate Example-Count Scheduling

- [x] 4.1 Change hysteresis progress to consumed examples and cap EGGROLL batches at observation boundaries. Preserve all switch, band, and invalid-controller behavior.
<!-- status: completed -->
  <!-- covers: train/alternating-cycle :: Average-variance hysteresis optimizer control :: Eggroll restores variance -->
  <!-- covers: train/alternating-cycle :: Average-variance hysteresis optimizer control :: Gradient detects collapse -->
  <!-- covers: train/alternating-cycle :: Average-variance hysteresis optimizer control :: Hysteresis band retains the active method -->
  <!-- covers: train/alternating-cycle :: Average-variance hysteresis optimizer control :: Eggroll batch ends at the observation boundary -->
  <!-- covers: train/alternating-cycle :: Average-variance hysteresis optimizer control :: Invalid variance controller -->
- [x] 4.2 Update epoch traversal to mix one-example gradient calls with multi-example EGGROLL calls while preserving exact order, partial windows, and the 2,850-example default total.
<!-- status: completed -->
  <!-- covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Observation window ends within an epoch -->
  <!-- covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Epoch boundary within an observation window -->
  <!-- covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Full default run -->
  <!-- covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Invalid epoch count -->
- [x] 4.3 Preserve complete-registry Adam state and matrix-only SGD state across both switch directions without reinitializing shared tensors.
<!-- status: completed -->
  <!-- covers: train/alternating-cycle :: Continuous model and optimizer state :: Switch from Eggroll to gradient training -->
  <!-- covers: train/alternating-cycle :: Continuous model and optimizer state :: Return to Eggroll -->
- [x] 4.4 Aggregate comparable per-example metrics and extend experiment positions with consumed examples, batch size, and method-specific optimizer-call counts.
<!-- status: completed -->
  <!-- covers: train/alternating-cycle :: Comparable phase measurements :: Compare phase metrics -->
  <!-- covers: train/alternating-cycle :: Comparable phase measurements :: Record experiment position -->
- [x] 4.5 Run a small alternating fixture with batch, observation, log, and epoch boundaries at different positions. Assert exact record visits, switches, metrics, and progress records.
<!-- status: completed -->

## 5. Migrate Checkpoint and Resume Contracts

- [x] 5.1 Extend standalone and alternating run configuration, optimizer manifests, metrics, and schedules with SGD type, matrix scope, fitness-batch size, consumed-example progress, optimizer-call counts, and stability-report identity.
<!-- status: completed -->
  <!-- covers: train/stage0-training :: Stabilized EGGROLL identity is resumable and incompatible changes are rejected :: Save stabilized Eggroll identity -->
- [x] 5.2 Reject legacy Adam EGGROLL state, non-matrix EGGROLL optimizer entries, impossible cursor accounting, and every guarded run-configuration mismatch before applying state.
<!-- status: completed -->
  <!-- covers: train/stage0-training :: Stabilized EGGROLL identity is resumable and incompatible changes are rejected :: Reject legacy Adam Eggroll checkpoint -->
  <!-- covers: train/alternating-cycle :: Resumable experiment checkpoints :: Reject incompatible resume configuration -->
- [x] 5.3 Add uninterrupted-versus-resumed tests at observation and epoch boundaries, including a preceding partial EGGROLL batch and exact next-candidate RNG equivalence.
<!-- status: completed -->
  <!-- covers: train/alternating-cycle :: Resumable experiment checkpoints :: Resume within an epoch -->
  <!-- covers: train/alternating-cycle :: Resumable experiment checkpoints :: Resume at an epoch boundary -->
  <!-- covers: train/eggroll-execution :: Optimized execution preserves EGGROLL training semantics :: Resume stays deterministic -->
- [x] 5.4 Save, inspect, and resume a small real `.ckpt` fixture through the public command. Confirm metadata uses SGD and only the declared matrix paths.
<!-- status: completed -->

## 6. Build the Absolute Stability Gate

- [x] 6.1 Define strict JSON and JSONL report models, canonical implementation digests, complete configuration identity, and snapshot-isolated fresh baseline evaluation over the fixed 64 held-out records.
<!-- status: completed -->
  <!-- covers: train/eggroll-stability-gate :: Stability gate measures a fresh absolute baseline :: Record the untrained baseline -->
  <!-- covers: train/eggroll-stability-gate :: Stability gate measures a fresh absolute baseline :: Baseline evaluation is isolated -->
- [x] 6.2 Implement deterministic training and evaluations at 8, 32, and 256 consumed examples with per-update relative matrix RMS, elapsed time, ETA, and early-stop evidence.
<!-- status: completed -->
  <!-- covers: train/eggroll-stability-gate :: Stability gate runs bounded development checkpoints :: Emit every healthy development checkpoint -->
  <!-- covers: train/eggroll-stability-gate :: Stability gate runs bounded development checkpoints :: Stop on absolute regression -->
- [x] 6.3 Implement final pass and failure classification for loss, exact accuracy, first-token accuracy, separation retention, and the 1% relative-RMS ceiling.
<!-- status: completed -->
  <!-- covers: train/eggroll-stability-gate :: Passing health requires task improvement and bounded updates :: Configuration passes the absolute gate -->
  <!-- covers: train/eggroll-stability-gate :: Passing health requires task improvement and bounded updates :: Configuration lacks task improvement -->
- [x] 6.4 Validate compatible passing reports before guarded workflows load models or data, and report every mismatched canonical field.
<!-- status: completed -->
  <!-- covers: train/eggroll-stability-gate :: Full EGGROLL workflows require a compatible passing report :: Start with a compatible passing report -->
  <!-- covers: train/eggroll-stability-gate :: Full EGGROLL workflows require a compatible passing report :: Reject a mismatched report -->
- [x] 6.5 Add `python -m train.run_eggroll_stability` with continuous checkpoint progress, bounded development execution, retained failure reports, and a final machine-readable summary.
<!-- status: completed -->
- [x] 6.6 Run the stability command on dependency-light fixtures for passing, early-regression, missing-improvement, stale-report, and mismatched-configuration outcomes.
<!-- status: completed -->

## 7. Gate Alignment Search and Long Runs

- [x] 7.1 Make EGGROLL alignment search validate its zero-weight stability result before positive trials. Emit `method_unhealthy` with no recommendation when it fails.
<!-- status: completed -->
  <!-- covers: train/eggroll-stability-gate :: Alignment search rejects an unhealthy zero-weight control :: Eggroll zero control is unhealthy -->
- [x] 7.2 Keep combined-search gradient execution independent when EGGROLL is unhealthy, including progress, results, and cleanup.
<!-- status: completed -->
  <!-- covers: train/eggroll-stability-gate :: Alignment search rejects an unhealthy zero-weight control :: Gradient search remains independent -->
- [x] 7.3 Require a compatible passing report for standalone runs beyond 256 examples and every alternating run. Add pre-model rejection tests for missing, failed, and stale reports.
<!-- status: completed -->
- [x] 7.4 Run fixture-backed public alignment-search and guarded-training commands to prove unhealthy EGGROLL cannot produce a recommendation or start a long run.
<!-- status: completed -->

## 8. End-to-End Verification

- [ ] 8.1 Run all focused EGGROLL, Stage 0 training, alternating scheduler, checkpoint, alignment-search, and stability-gate tests with `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`.
- [ ] 8.2 Run the complete project test suite and strict OpenSpec validation. Repair regressions without weakening scenarios or tolerances.
- [ ] 8.3 Update the CUDA equivalence and performance benchmark to report time and peak memory per consumed example for fitness-batch size eight. Require optimized outputs to match the materialized reference and preserve the existing speed and memory gate.
- [ ] 8.4 Run the real bounded CUDA stability command from a fresh state. Preserve its JSON and JSONL artifacts, parameter/configuration identity, checkpoint metrics, and pass or exact failure conditions.
- [ ] 8.5 Stop before full training. Review the bounded stability artifact as the independent practice test for the stabilized system, then use a new apply request for any threshold or default revision revealed by that evidence.
