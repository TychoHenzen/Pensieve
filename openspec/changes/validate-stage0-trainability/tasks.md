## 1. Define Evidence and Identity Contracts

- [x] 1.1 Add strict trainability report types, enums, canonical JSON serialization, duplicate-key rejection, finite-value checks, and cross-field validation.
- [x] 1.2 Add a diagnostic stability-report loader that accepts only a compatible failed report while preserving the production loader's passing-report requirement.
  <!-- covers: train/stage0-trainability-gate :: Investigation uses compatible fresh-state evidence :: Compatible failed report starts investigation -->
  <!-- covers: train/stage0-trainability-gate :: Investigation uses compatible fresh-state evidence :: Stale or passing report is rejected -->
- [x] 1.3 Build the fixed overfit, 32-record training, and 64-record held-out selections from the report identity. Expose no test or checkpoint input.
  <!-- covers: train/stage0-trainability-gate :: Investigation uses compatible fresh-state evidence :: Test and checkpoint inputs are unavailable -->
- [x] 1.4 Add initial-state and implementation digests covering both trainers, the objective, decoder, data selection, evaluator, runner, and report parser.
- [x] 1.5 Run focused report-parser and identity tests as an independently runnable evidence-contract check.

## 2. Implement the Shared-Objective Overfit Probe

- [x] 2.1 Add a fresh-state attempt runner for the fixed first training record at diagnostic learning rates `0.0001`, `0.001`, and `0.01`, with checkpoints at 0, 1, 4, 16, and 64 calls.
- [x] 2.2 Add exact, first-token, loss-ratio, finite-state, early-pass, and per-attempt non-finite handling without sharing state between learning rates.
  <!-- covers: train/stage0-trainability-gate :: One-record overfit probe falsifies the shared objective first :: Objective demonstrates memorization -->
  <!-- covers: train/stage0-trainability-gate :: One-record overfit probe falsifies the shared objective first :: One learning rate becomes non-finite -->
- [x] 2.3 Add fail-fast objective classification that preserves all completed attempts and skips causal and arm probes when no attempt passes.
  <!-- covers: train/stage0-trainability-gate :: One-record overfit probe falsifies the shared objective first :: Objective fails bounded memorization -->
- [ ] 2.4 Add deterministic tests proving each learning-rate attempt starts from identical tensors and RNG state and never emits a production recommendation.
- [ ] 2.5 Run the one-record probe against lightweight production-path fixtures as an independently runnable objective check.

## 3. Implement Causal Update Probes

- [ ] 3.1 Add complete-objective evaluation over the first eight training records before and after one isolated method update.
- [ ] 3.2 Capture the gradient and EGGROLL minimizing directions and compute the signed first-order predicted objective changes without changing production update math.
- [ ] 3.3 Add causal result classification for predicted and observed decreases, including exact `direction_mismatch` evidence.
  <!-- covers: train/stage0-trainability-gate :: Causal probes compare predicted and actual update direction :: Method update has the predicted effect -->
  <!-- covers: train/stage0-trainability-gate :: Causal probes compare predicted and actual update direction :: Update moves against its prediction -->
- [ ] 3.4 Add held-out separation, relative matrix RMS, finite-state, snapshot restoration, and `unsafe_update` checks for both methods.
  <!-- covers: train/stage0-trainability-gate :: Causal probes compare predicted and actual update direction :: Causal update damages held-out separation -->
- [ ] 3.5 Run focused gradient and EGGROLL causal tests with known aligned, reversed, oversized, and non-finite updates.

## 4. Implement Equal-Budget Method Arms

- [ ] 4.1 Add independent no-update, gradient-only, and EGGROLL-only arms from one canonical fresh-state manifest over the same ordered 32 records.
- [ ] 4.2 Count consumed examples consistently, cap EGGROLL batches at 8 and 32 records, and emit evaluations for every arm at 0, 8, and 32 examples.
  <!-- covers: train/stage0-trainability-gate :: Equal-budget arms isolate update-method behavior :: Arms receive equal evidence -->
- [ ] 4.3 Make the no-update arm advance only its diagnostic cursor and compare tensors, optimizer state, RNG state, and metrics with its baseline.
  <!-- covers: train/stage0-trainability-gate :: Equal-budget arms isolate update-method behavior :: No-update control detects drift -->
- [ ] 4.4 Add independent non-finite, relative-RMS, and separation early stops so one failed arm cannot cancel another.
  <!-- covers: train/stage0-trainability-gate :: Equal-budget arms isolate update-method behavior :: Unsafe arm stops independently -->
- [ ] 4.5 Add integration tests for identical record exposure, arm isolation, optimizer-call accounting, partial EGGROLL boundaries, and independent completion.
- [ ] 4.6 Run the three arms on lightweight production-path fixtures as an independently runnable method-comparison check.

## 5. Classify Findings and Eligibility

- [ ] 5.1 Implement method statuses from causal and 32-example evidence, requiring loss, exact, first-token, separation, and RMS conditions for `viable`.
  <!-- covers: train/stage0-trainability-gate :: Method findings require decoded improvement :: Method shows bounded trainability -->
  <!-- covers: train/stage0-trainability-gate :: Method findings require decoded improvement :: Loss improves without useful behavior -->
- [ ] 5.2 Implement overall classification precedence for inconclusive evidence, objective failure, bounded trainability, shared conflict, and method-specific failure.
  <!-- covers: train/stage0-trainability-gate :: Overall classification follows a deterministic precedence :: At least one method is viable -->
  <!-- covers: train/stage0-trainability-gate :: Overall classification follows a deterministic precedence :: Both methods repeat the observed conflict -->
  <!-- covers: train/stage0-trainability-gate :: Overall classification follows a deterministic precedence :: Control drift invalidates interpretation -->
- [ ] 5.3 Compute method-specific recalibration eligibility from the shared overfit and causal prerequisites and report every missing prerequisite.
  <!-- covers: train/stage0-trainability-gate :: Recalibration eligibility is method-specific and limited :: Causally supported method may be recalibrated -->
  <!-- covers: train/stage0-trainability-gate :: Recalibration eligibility is method-specific and limited :: Unsupported method remains blocked -->
- [ ] 5.4 Keep trainability evidence distinct from stability and five-seed credentials. Add tests proving it cannot authorize full gradient, EGGROLL, or alternating training.
  <!-- covers: train/stage0-trainability-gate :: Recalibration eligibility is method-specific and limited :: Trainability report is not a training gate -->
- [ ] 5.5 Run table-driven classification and eligibility tests covering every status and precedence branch.

## 6. Add the Diagnostic Command and Progress Artifacts

- [ ] 6.1 Add `python -m train.run_stage0_trainability` with required stability-report and final-output paths, optional progress path, and validation before model or data loading.
- [ ] 6.2 Write exclusive-create JSONL progress after every declared event and flush records so interruption leaves readable evidence.
- [ ] 6.3 Write one canonical final report for scientific pass or failure, return a nonzero exit for every non-viable outcome, and retain completed progress.
  <!-- covers: train/stage0-trainability-gate :: Reports preserve complete bounded evidence :: Failed investigation retains evidence -->
- [ ] 6.4 Reject either pre-existing output path before file modification, model construction, or dataset access.
  <!-- covers: train/stage0-trainability-gate :: Reports preserve complete bounded evidence :: Existing output is protected -->
- [ ] 6.5 Add command tests for argument validation, orchestration order, progress sequence, ETA fields, early exits, final schema, and exit codes.
- [ ] 6.6 Run the command on deterministic fixtures and inspect both JSON and JSONL as an independently runnable end-to-end practice test.

## 7. Verify Without Starting Training

- [ ] 7.1 Run the focused trainability, report, stability, gradient, EGGROLL, objective, and evaluation test modules.
- [ ] 7.2 Run the complete suite with `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`.
- [ ] 7.3 Run optimized-reference equivalence and the existing CUDA performance and memory gates to confirm diagnostic reuse did not change training math.
- [ ] 7.4 Run `openspec validate validate-stage0-trainability --strict --no-interactive` and require complete scenario coverage with zero regressions.
- [ ] 7.5 Run the bounded command against `gate_results/stabilize-eggroll-training/stability-seed0-cuda-20260826T1555.json`, retain its JSON and JSONL result, and stop without recalibration, full training, threshold changes, default selection, or archival.
