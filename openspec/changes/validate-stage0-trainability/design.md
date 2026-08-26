## Context

See `proposal.md` for motivation. The implemented EGGROLL stability command already provides deterministic fresh-state construction, fixed held-out evaluation, state isolation, complete identity, source digests, JSON reporting, and early stopping. Its retained seed-0 CUDA report failed after 32 consumed examples: language-model loss improved from `10.06098` to `9.83429`, separation retention fell from `0.45030` to `0.39203`, decoded exact and first-token accuracy stayed at zero, and maximum relative matrix RMS remained below the existing ceiling.

That result does not distinguish five different causes: an objective that cannot memorize its target, an incorrect local update direction, an EGGROLL-only estimator problem, a shared loss-versus-behavior conflict, or an evaluation/control defect. Recalibrating production defaults before separating those causes would mix diagnosis with optimization.

## Goals / Non-Goals

**Goals:**

- Reuse the production objective, trainers, data contract, and evaluation metrics rather than creating a second training definition.
- Produce causal evidence from the smallest probe that can answer each question.
- Compare methods from byte-identical initial trainable state and equal record exposure.
- Make every failure actionable through deterministic classifications and method-specific prerequisites.
- Keep the maximum run bounded to three 64-call gradient attempts, two one-update causal probes, and three 32-record arms.

**Non-Goals:**

- Find a better production learning rate, sigma, population, rank, fitness-batch size, alignment weight, or separation threshold.
- Demonstrate generalization, five-seed acceptance, or readiness for full training.
- Change the latent equation, frozen assets, prompt, targets, EGGROLL factor orientation, ascent sign, or official normalization.
- Resume, translate, or produce a training checkpoint.
- Archive `stabilize-eggroll-training` or reinterpret its failed acceptance result.

## Decisions

### 1. Build one orchestration layer over existing production paths

Add a trainability runner that calls the existing Stage 0 data loader, fresh-state factory, gradient trainer, EGGROLL trainer, stability evaluator, and canonical identity helpers. Extract a shared helper only when current code cannot be called without duplicating behavior. The investigation must not maintain a parallel objective or decoder.

Alternative considered: implement small synthetic trainers for each probe. This could make tests fast, but a passing synthetic path would not establish that the production path is trainable.

### 2. Require the retained failed stability report as the experiment anchor

The command takes `--stability-report`, `--output`, and optional `--progress-output`. It reads the failed report with structural validation, then checks identity, implementation sources, EGGROLL settings, and exact held-out identifiers against the current command. A separate diagnostic loader accepts `status=failed` but does not weaken the production guard that requires `status=passed`.

This makes the investigation answer the observed failure rather than a nearby configuration. The report SHA-256 becomes part of the trainability identity.

Alternative considered: allow an unanchored investigation. That would make results difficult to connect to the failed gate and would permit silent configuration drift.

### 3. Use a preregistered overfit ladder to test the objective, not select defaults

Run three independent Adam attempts on one fixed record at learning rates `1e-4`, `1e-3`, and `1e-2`. Each receives the same freshly constructed parameters and initial RNG state. Checkpoints occur at 0, 1, 4, 16, and 64 optimizer calls. Stop an attempt on success or non-finite state, but continue the next rate from a clean state.

The ladder separates an objective-level failure from one poorly scaled default. Exact decoded memorization plus a 50% teacher-forced loss reduction is intentionally strict. A one-example model that cannot meet both conditions has not demonstrated the minimum mechanism needed for Stage 0.

The ladder is hard-coded and labeled diagnostic. Its winning rate is never copied into run defaults or emitted as a recommendation.

Alternative considered: use only the current `1e-4` default. Failure would not distinguish the objective from learning-rate scale. A broader adaptive search was rejected because it would become recipe tuning.

### 4. Measure causal direction on the same support before longer arms

For each method, snapshot a fresh state and evaluate the complete canonical training objective over the first eight records. Construct one update while retaining the method's estimated direction. Record the signed first-order prediction before applying the update. Then apply it once and reevaluate the same support plus the held-out records.

For gradient, the prediction is the dot product between the detached gradient and applied parameter delta. For EGGROLL, it is the dot product between the assembled minimizing pseudo-gradient and applied matrix delta. In both cases, a negative value predicts a lower objective. The observed objective delta uses the same objective components and weights that produced the update.

The causal probe uses a temporary arm, so it cannot advance or seed the 32-record comparison. Parameter tensors, optimizer state, and RNG state are hashed before and after snapshot restoration.

Alternative considered: infer correctness from loss after 32 records. That conflates update direction with data order, repeated updates, and representation collapse.

### 5. Compare record exposure rather than optimizer-call count

Create three independent arms from one canonical initial-state manifest. The no-update arm advances its cursor without training. Gradient consumes one ordered record per call. EGGROLL consumes the existing deterministic fitness batches and caps them at record 8 and record 32. All arms evaluate the same 64 held-out records at 0, 8, and 32 consumed examples.

The no-update arm is an evaluation and state-isolation control. Any tensor, optimizer, RNG, or metric drift makes the result inconclusive. Updated arms stop independently, so one failure does not discard evidence from another method.

Alternative considered: compare equal optimizer calls. One EGGROLL call consumes eight records while one gradient call consumes one, so that comparison would not control data exposure.

### 6. Separate method status, overall diagnosis, and workflow eligibility

Each method receives a causal status, a 32-record arm status, and a `recalibration_eligible` boolean. Eligibility requires the shared overfit pass plus that method's causal pass. It does not require the current 32-record recipe to pass, because a bounded causal success followed by deterioration is valid evidence for a later bounded recalibration proposal.

The overall classification follows the precedence in the spec. `inconclusive` overrides scientific interpretations. An overfit failure stops before method comparison. A viable arm establishes bounded trainability. Two finite loss improvements with no useful behavior establish the shared conflict. Remaining asymmetric results become method-specific failure.

This change introduces no recalibration search. A later proposal must cite the compatible report, choose only eligible methods, define its own bounded search, and retain the existing thresholds.

Alternative considered: let any loss decrease unlock recalibration. The failed stability gate already showed why loss alone is insufficient.

### 7. Extend the existing immutable evidence format

Use frozen report records with strict finite-value validation and canonical JSON. JSONL events include monotonic sequence number, kind, probe or arm, consumed examples or optimizer calls, elapsed time, and payload. The final JSON includes the complete event-derived result plus identity and configuration. Writers use exclusive creation and flush each JSONL record.

The implementation digest covers the investigation runner, objective, both trainers, state construction, data selection, decoder, evaluator, and report parser. Report parsing rejects duplicate keys, unknown fields, non-finite numbers, invalid enums, inconsistent checkpoints, digest mismatches, and eligibility that does not follow the recorded prerequisites.

Alternative considered: reuse training checkpoints as evidence. Checkpoints are mutable execution state and would unnecessarily authorize resume behavior.

## Risks / Trade-offs

- [The one-record ladder passes through memorization but the architecture still cannot generalize] -> Treat it only as an objective prerequisite. Require held-out decoded improvement for method viability and keep all full gates unchanged.
- [The fixed 64-call budget produces a false negative] -> Report complete trajectories and classify only the bounded claim. Changing the budget requires a later spec revision, not an ad hoc rerun.
- [A causal objective decrease is too small to separate from float noise] -> Record full precision, parameter movement, and repeated objective evaluation. A non-negative observed delta does not pass.
- [Three arms repeat expensive decoding] -> Reuse the existing batched held-out evaluator, emit progress and ETA, and stop unsafe arms at the first declared checkpoint.
- [The active stabilization change is not archived] -> Depend on its implemented report format without editing or archiving its artifacts. Keep this new capability in a separate path to avoid conflicting deltas.
- [A diagnostic learning rate is mistaken for a recommended default] -> Encode the ladder as a fixed probe, omit winner language, and reject its use as a full-training credential.

## Migration Plan

1. Add report models, strict parsing, classification, and fake-runner tests without touching production training defaults.
2. Add shared fresh-state and metric adapters needed by the diagnostic runner.
3. Add the overfit ladder and verify state equality across learning-rate attempts.
4. Add gradient and EGGROLL causal probes with explicit predicted and observed direction tests.
5. Add equal-budget arms, control-drift checks, independent early stopping, and progress output.
6. Add command validation and method-specific eligibility checks.
7. Run focused tests, the complete suite, strict OpenSpec validation, and coverage.
8. Run the command against the retained failed CUDA report. Preserve its outcome without starting recalibration or training.

Rollback removes the diagnostic command and report types. It does not modify training checkpoints or defaults, so no persisted model migration is required.
