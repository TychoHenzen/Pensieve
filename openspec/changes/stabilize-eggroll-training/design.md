## Context

See `proposal.md` for motivation. The current trainer uses one stable ten-parameter registry and creates Adam for both update methods. EGGROLL perturbs all ten tensors, evaluates every candidate on one problem, normalizes fitness with sample standard deviation, and advances the scheduler once per optimizer call. The factor orientation, antithetic pairing, factorized projection, and ascent sign already match the pinned upstream reference and must remain unchanged.

The audit measured fresh matrix RMS values near `0.019` to `0.029`. A first Adam EGGROLL update changed those matrices by roughly `0.001` RMS, or 3.4% to 5.2%, before any optimizer calibration. The supplied search also showed that a zero-alignment EGGROLL control could improve loss while reducing cross-problem separation retention from `0.450` to `0.018`.

## Goals / Non-Goals

**Goals:**

- Preserve the verified low-rank execution path while matching the official score formula.
- Make EGGROLL update magnitude, parameter scope, and optimizer state explicit and independently resumable.
- Give one EGGROLL estimate evidence from several ordered training problems without changing epoch membership.
- Block long runs and alignment recommendations until a bounded absolute gate observes task improvement without representation collapse.
- Preserve deterministic resume across partial Eggroll batches, observation windows, logging boundaries, and epochs.

**Non-Goals:**

- Change the frozen Qwen or MiniLM assets, prompt, numerical target, latent-loop equation, alignment objective, or variance-controller thresholds.
- Tune population, rank, sigma, learning rate, fitness-batch size, or alignment weight during a production run.
- Resume Adam-based EGGROLL checkpoints under the new contract.
- Claim the bounded gate proves the five-seed Stage 0 acceptance gate.

## Decisions

### 1. Use explicit method-specific parameter registries and optimizers

Keep the existing complete ordered registry as the shared model and gradient registry. Add an explicit EGGROLL registry containing only `encoder.projection.weight`, `encoder.slot_queries`, and `latent_loop.projection.weight`. Construct Adam from the complete registry and momentum-free SGD from the EGGROLL registry.

An explicit name list is safer than selecting every two-dimensional tensor dynamically. A later two-dimensional parameter must not enter EGGROLL without a contract and checkpoint migration. Non-matrix tensors remain shared model state, so gradient phases can still update them.

The EGGROLL defaults become `sigma=0.001` and SGD learning rate `0.1`. This keeps candidate noise below the current `0.02` radius and targets a first-step matrix change below the gate's 1% relative-RMS ceiling. The stability gate remains authoritative if this initial calibration is still too large.

Alternative considered: retain Adam and reduce its learning rate. Adam's first update is approximately its learning rate per nonzero coordinate and elementwise normalization destroys the scale carried by the ES estimator. This does not address the observed mechanism.

### 2. Implement the upstream fitness formula once and verify it with golden values

Normalize finite population fitness with `var(unbiased=False)` and `sqrt(var + 1e-5)`. Form negative-minus-positive pair scores and retain the existing `sqrt(N) / N` population scale and descent sign.

Production factorized code and the materialized reference may share input validation, but the reference normalization must not call the production helper. Tests will include fixed golden values calculated from the pinned upstream formula. This prevents the previous failure where both paths copied the same incorrect normalization.

Alternative considered: preserve sample standard deviation because Adam mostly removes scale. The new optimizer is SGD, so the scale difference becomes observable and must match upstream exactly.

### 3. Treat Eggroll progress as consumed examples, not optimizer calls

Introduce a deterministic `FitnessBatch` containing the next ordered training records and their item identifiers. The batch planner chooses the minimum of configured batch size and the records remaining before the next epoch, observation-window, or structured logging boundary. Gradient receives a one-record batch. Eggroll evaluates every candidate on every record in its batch and averages per-record fitness before population normalization.

Candidate seeds and perturbations remain fixed across all problems in one optimizer call. The implementation loops over problems outside the existing candidate-evaluation batches, accumulates candidate metrics, and releases per-problem activations before the next problem. This avoids a `problem_count x population` activation allocation. With fitness-batch size eight, optimizer calls fall by about eight while model forwards per consumed example remain comparable.

`global_step`, dataset cursor, observation-window progress, and logging cadence continue to count consumed examples. Add total and method-specific optimizer-call counters. Checkpoints occur only after a complete optimizer call, because the planner caps batches at each checkpoint-producing boundary.

Alternative considered: reuse a fixed support set while advancing one primary example. That would expose selected records more than once per epoch and make the dataset contract harder to interpret.

### 4. Make the stability report a configuration proof, not a training checkpoint

Add `python -m train.run_eggroll_stability`. It constructs a fresh seed-0 state, hashes the ordered source files that define EGGROLL execution and evaluation, evaluates the fixed 64-record held-out development set, consumes at most 256 training records, and re-evaluates at 8, 32, and 256 consumed examples.

The command writes append-only JSONL progress plus a final JSON report. It records the complete asset identity, implementation digest, run configuration, baseline and checkpoint metrics, per-update relative matrix RMS changes, failure reasons, elapsed time, and ETA. Evaluation uses the existing snapshot-and-restore isolation contract.

Guarded commands accept `--stability-report`. They validate the full report before loading models or data and copy its digest into run configuration and checkpoints. The report is invalid after any guarded source digest or configuration field changes. A development invocation capped at 256 examples remains available to produce a new report.

Alternative considered: continue after warnings and rely on manual review. The completed alignment search already demonstrated that a relative optimum can look valid while absolute task quality remains unusable.

### 5. Gate alignment search on a healthy EGGROLL zero control

Run the zero-alignment EGGROLL configuration through the absolute gate before positive EGGROLL weights. An unhealthy zero control produces `method_unhealthy` and no recommendation. A combined gradient and EGGROLL search continues the gradient branch independently.

This deliberately separates two questions. The stability gate asks whether EGGROLL can learn without catastrophic optimizer or estimator behavior. The alignment search then asks which positive alignment weight best preserves conditioning for an already functional method.

### 6. Reject legacy EGGROLL optimizer state

The existing safe checkpoint container already records optimizer type, ordered paths, and group scalars. Extend run configuration and schedule metadata with fitness-batch size, EGGROLL parameter scope, consumed-example progress, optimizer-call counters, and stability-report identity. Reject Adam-based EGGROLL state and any non-matrix EGGROLL optimizer entry during metadata validation.

Do not translate Adam moments to SGD. There is no behavior-preserving conversion, and the earlier checkpoints did not pass the task gate.

## Risks / Trade-offs

- [SGD learning rate `0.1` is still miscalibrated] -> The relative-RMS ceiling fails the bounded gate before a long run. Adjusting the default requires a spec update and a new report.
- [Multi-problem candidate evaluation increases one optimizer call's duration] -> Keep problem activations sequential, show per-problem progress, and compare time per consumed example rather than time per optimizer call.
- [A batch boundary can complicate schedule accounting] -> Cap each batch at epoch, observation-window, and logging boundaries, and test every partial-boundary combination plus resume.
- [The absolute thresholds reject a potentially recoverable early trajectory] -> Evaluate only at declared checkpoints and retain the partial report. Threshold changes remain explicit contract changes.
- [Implementation changes invalidate reports frequently during development] -> Development runs can generate reports without a prior report. Only alignment selection and runs beyond 256 examples require compatibility.

## Migration Plan

1. Add golden normalization and optimizer-scope tests while the production path still fails them.
2. Introduce method-specific registries, SGD, safe defaults, and checkpoint validation. Keep gradient behavior unchanged.
3. Add fitness batching and example-count schedule state. Verify uninterrupted and resumed runs at every boundary.
4. Add the stability command and report validation. Run a bounded report on CUDA.
5. Gate alignment search, standalone EGGROLL, and alternating training on that report.
6. Re-run focused and full suites, optimized-reference equivalence, and the CUDA performance benchmark per consumed example.
7. Discard legacy Adam EGGROLL checkpoints. Start subsequent training from a fresh deterministic state.

Rollback removes the guarded commands and restores the previous defaults in code, but it cannot make new SGD checkpoints compatible with the old Adam contract. Preserve reports and checkpoints as diagnostic artifacts rather than converting them.
