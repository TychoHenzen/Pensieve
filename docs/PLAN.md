# Pensive - Evidence-gated learning roadmap

This file is the authoritative roadmap for Pensive. It defines the order in
which learning claims may be made and the evidence required before the next
step starts.

> **Current status.** The Stage -1 implementation and contract are complete.
> The five-seed reproduction evidence is still pending. Stage 0 code exists,
> but Stage 0 trainability and gate evidence are not accepted. No Stage 1 or
> later work may advance from code, unit tests, or training loss alone.

## Status vocabulary

Use these terms in plans, reports, and issue descriptions:

- **Code-complete:** the implementation and automated checks for a scope exist.
- **Trainability-demonstrated:** a bounded probe shows that the selected
  objective can produce the required behavior on fixed and held-out data.
- **Gate-passed:** the full multi-seed comparison against its matched baseline
  meets the stated threshold and has a saved report.
- **Blocked:** a required metric regresses, output becomes invalid or collapsed,
  retention falls, or repeatability fails. The next stage does not start.

Code-complete does not imply trainability-demonstrated. Trainability-
demonstrated does not imply gate-passed.

## Advancement rules

Every stage below records its model and scale, task or data, objective, matched
baseline, pre-training and post-training metrics, seed and compute budget, pass
condition, and stop or retreat condition.

The following rules apply to every stage:

1. Record the exact code revision, configuration, data identity, seeds, and
   hardware in the report.
2. Compare the changed system with a parameter-matched or protocol-matched
   baseline on the same inputs.
3. Report behavior before and after training or adaptation. Loss reduction is
   supporting evidence, never a gate by itself.
4. A regression in behavior, output validity, representation separation,
   retention, or repeatability blocks advancement even when loss improves.
5. Keep failed and partial reports. Do not rerun with changed thresholds or
   budgets until a new decision is recorded.

### Metric and failure rules

- **Valid output** means the required sequence length, vocabulary-only tokens,
  and a sequence that satisfies the declared recurrence. The Stage 0B fixed-set
  gate requires 100% valid output. Stage 0C and 0D require at least 99%.
- **Representation separation** is the mean pairwise cosine distance between
  L2-normalized final hidden states for distinct evaluation records. Stage 0D
  requires at least 0.10 for every seed and no more than a 0.02 decrease from
  its matched control.
- **Workspace collapse** uses `Workspace.collapse_stats()`: variance is the
  mean feature variance across slots, and covariance is the mean absolute
  off-diagonal slot covariance. For every multi-slot Stage 0E seed, variance
  must exceed 0.1, covariance must be finite, and post-training covariance
  may exceed pre-training covariance by at most `max(10%, 0.01)`.
- **Seed aggregation** is explicit. Per-seed safety and validity gates apply
  to every seed. Mean thresholds report mean and standard deviation. A stage
  that names a four-of-five rule uses that rule only for the named metric.
- Any non-finite value, invalid output, representation regression, or failed
  threshold blocks the stage. Preserve the report, mark the stage blocked, and
  retreat to the last passing rung. Do not change a threshold after seeing a
  failed result.

## Target system

Text and other input reach an encoder that writes concept vectors into a
persistent workspace. A slow core sets an abstract goal and passes a few
vectors through a narrow channel to a fast core. The fast core can update a
bounded memory during use. Most internal cycles emit nothing. An output gate
decides when to run the decoder and emit text. Idle time replays recent
episodes into the slow core.

This architecture remains a research hypothesis. The ladder below tests the
learning path before the project spends another large-model training run.

## Protected Fashion-MNIST baseline

The first protected behavioral reference is the existing class-incremental
Fashion-MNIST routine. It is a baseline for later claims, not a Stage 0 result.

- **Implementation:** `scripts/show_gate.py`,
  `eval/stream/generators/split_classify.py`, and the five baseline
  implementations under `eval/baselines/`.
- **Model and scale:** a 2-layer, width-400 ReLU MLP with 784 input features
  and 10 output classes. Naive, EWC, replay, and joint subjects use the same
  architecture and optimizer settings.
- **Task and data:** Fashion-MNIST pixels, five class-incremental tasks, two
  classes per task, 2,000 training examples per task, and 200 probes per task.
  The stream identity must include `data_source="fashion-mnist"`.
- **Objective:** supervised class prediction on the observed examples.
- **Reference command:**
  `python scripts/show_gate.py > gate.log 2>&1`.
- **Reference report:**
  `openspec/changes/archive/2026-08-17-stage-minus-1-remainder/cycling-experiment-results.md`.
  It records three seeds per configuration from 2026-08-17.
- **Comparison scope:** the protected report compares four methods: naive,
  EWC, replay, and joint. The fifth implementation, frozen, remains a
  harness sanity floor and is not part of the protected reference comparison.
- **Reference values:** on the sequential configuration, replay is 77.0%
  and joint is 84.5% mean final accuracy. On the 10-cycle configuration,
  replay is 82.6%. Naive is 19.8% sequential, and EWC is 31.1% at 10 cycles.
- **Known confound:** the 50-cycle and 200-cycle configurations use a
  50-iteration floor, so they consume more optimizer iterations than the
  sequential configuration. They are diagnostic comparisons, not the sole
  protected baseline.

The archived command identifies the historical routine, but it is not itself a
reproducible gate: the current script draws three random 32-bit seeds and does
not record their values. Recovery must use a pinned runner with explicit
canonical seeds `[0, 1, 2, 3, 4]`, the recorded code revision, dataset and
stream identities, environment, method configuration, and per-seed results.
It covers five cycle configurations, four comparison methods, both datasets,
and the frozen sanity floor. It passes only when the report contains all five
seed-level results and the sequential replay and joint means are each within
2 percentage points of the recorded reference. A mismatch stops the ladder
and triggers protocol or data investigation.

## Ordered Fashion-MNIST to LLM ladder

The first four rungs are the bridge to the larger LLM experiment. They are
part of Stage 0 evidence and must pass in order.

### Stage 0A: recover the protected baseline

- **Model and scale:** the 2-layer, width-400 MLP described above.
- **Task and data:** the exact Fashion-MNIST and MNIST cycling streams used by
  `scripts/show_gate.py`.
- **Objective:** supervised class prediction.
- **Matched baseline:** naive, EWC, replay, and joint subjects with the same
  architecture, stream, and optimizer settings.
- **Pre/post metrics:** final accuracy per task, retention matrix, mean and
  standard deviation across seeds, and total optimizer iterations.
- **Budget:** five configurations, four comparison methods, two datasets, and
  five canonical seeds `[0, 1, 2, 3, 4]`, with 2,000 examples per task and
  200 probes per task. The frozen subject is a sanity floor, not a gate method.
- **Pass condition:** reproduce the protected reference within 2 percentage
  points for sequential replay and joint mean accuracy, with all seed-level
  records present.
- **Stop or retreat:** preserve the historical report as reference evidence,
  diagnose the mismatch, and do not start the transformer rung.

### Stage 0B: fixed-set tiny causal-transformer trainability

- **Model and scale:** a decoder-only causal transformer with 2 layers,
  `d_model=128`, 4 attention heads, `d_ff=512`, and context length 66 so the
  same model can evaluate 64-token sequences. Use an 18-token vocabulary:
  `BOS`, hexadecimal symbols `0..F`, and `EOS`. Keep the trainable parameter
  count below 1 million.
- **Task and data:** generate deterministic hexadecimal sequences from record
  IDs `s=0..255`. Set `x_0=floor(s/16)`, `x_1=s mod 16`, and
  `x_t=(x_(t-1)+2*x_(t-2)+t) mod 16` for `t>=2`. Use record IDs `0..95` for
  training and hold out `96..127`. The first two symbols make every record
  identity unique, so the train, held-out, and long-context record sets have
  zero canonical sequence overlap. Assert that disjointness before training.
- **Objective:** next-token cross-entropy over the complete causal sequence.
- **Matched baseline:** an untrained control and a parameter-matched 2-layer
  GRU using the same tokenization, split, seeds, and optimizer budget.
- **Pre/post metrics:** token accuracy, exact-sequence accuracy, recurrence
  validity, non-finite count, and parameter movement on the fixed evaluation
  subset. The subset is exactly record IDs `s=0..31`; checkpoint evaluation
  performs no optimizer updates.
- **Budget:** five seeds `0..4`, AdamW at `3e-4`, batch size 32, at most 2,000
  updates per seed, and checkpoints at updates 0, 100, 500, 1,000, and 2,000.
- **Pass condition:** every seed reaches at least 99% token accuracy and 95%
  exact-sequence accuracy on the fixed subset by update 2,000. Every emitted
  sequence must contain only valid tokens and the required length, and every
  seed must show finite, nonzero parameter movement.
- **Stop or retreat:** a non-finite state or a failed fixed-set gate means the
  objective has not demonstrated bounded trainability. Reduce the task before
  increasing model size or starting a latent modification.

### Stage 0C: held-out autoregressive behavior

- **Model and scale:** the Stage 0B transformer, with no architecture or
  tokenizer change.
- **Task and data:** the same recurrence, with the 32 held-out record IDs
  `96..127` never used for updates. Add a 64-token evaluation sequence from
  record IDs `128..159` for long-context behavior. These IDs remain disjoint
  from training and fixed-set records.
- **Objective:** the same next-token cross-entropy, with the fixed-set subset
  retained only as a memorization control.
- **Matched baseline:** the Stage 0B parameter-matched GRU and the untrained
  transformer, evaluated on the same held-out records.
- **Pre/post metrics:** fixed-set and held-out token accuracy, exact-sequence
  accuracy at lengths 32 and 64, recurrence validity, and seed spread.
- **Budget:** five seeds, the Stage 0B optimizer budget, and no more than one
  additional evaluation pass over the held-out records.
- **Pass condition:** mean held-out token accuracy is at least 90%, mean
  exact-sequence accuracy at length 32 is at least 75%, length-64 recurrence
  validity is at least 99%, and the transformer is no more than 5 percentage
  points below the matched GRU on exact-sequence accuracy. At least four of
  five seeds must reach 70% exact accuracy at length 32.
- **Stop or retreat:** if fixed-set accuracy passes but held-out behavior does
  not, classify the result as memorization and do not apply a latent change.

### Stage 0D: one controlled latent modification

- **Model and scale:** the Stage 0B transformer plus one shared four-step
  residual latent refinement over the final hidden state before the output
  head. Keep the tokenizer, `d_model`, data, optimizer, and update budget
  fixed. Match total parameters with a non-recurrent residual control.
- **Task and data:** the Stage 0C held-out recurrence at lengths 32 and 64.
- **Objective:** unchanged next-token cross-entropy. The modification is the
  only changed learning component.
- **Matched baseline:** the parameter-matched non-recurrent control trained
  from the same initial state and seed.
- **Pre/post metrics:** exact accuracy at both lengths, token accuracy,
  recurrence validity, representation separation, and parameter movement.
- **Budget:** five paired seeds, 2,000 updates, and one fixed evaluation pass
  per checkpoint.
- **Pass condition:** the latent treatment improves length-64 exact accuracy
  by at least 5 percentage points over the matched control, loses no more than
  2 points at length 32, preserves at least 99% valid outputs, has at least
  0.10 representation separation for every seed, and passes the Stage 0C
  held-out thresholds.
- **Stop or retreat:** if the treatment fails, retain the simpler transformer
  as the research result and record the latent modification as rejected. Do
  not compensate by changing several mechanisms at once.

### Stage 0E: larger latent-core experiment

- **Model and scale:** the current Stage 0 implementation, Pythia-160M with
  768 hidden dimensions, a configurable workspace of 1, 4, 8, 16, 32, or 64
  slots, default 16, and a fixed latent-loop count of 8. The Pythia backbone
  and MiniLM edge encoder remain frozen. Only the documented wrapper and
  projection parameters train.
- **Task and data:** GSM8K train data for training and the GSM8K test split
  for evaluation. Keep the recovered Fashion-MNIST report as a separate
  continual-learning reference, not as a substitute for GSM8K evidence.
- **Objective:** teacher-forced answer-token loss with the existing VICReg
  variance and covariance collapse guards.
- **Matched baseline:** the same Pythia-160M checkpoint in token chain-of-
  thought mode, using the prompt and deterministic generation in
  `eval/gate/token_cot_baseline.py`.
- **Pre/post metrics:** token-CoT accuracy before comparison, latent mean and
  standard deviation over at least five seeds, exact answer validity, slot
  ablation results, and workspace variance and covariance. Record the
  untrained and trained latent results separately.
- **Budget:** the existing 10-epoch training entry point, five evaluation
  seeds, the `{1,4,8,16,32,64}` slot ablation, and no unbounded recipe search.
- **Pass condition:** the token baseline and latent report both exist, the
  latent evaluation uses at least five seeds, latent mean accuracy is at least
  the token baseline, all emitted answers are valid, and multi-slot runs stay
  above the workspace healthy variance threshold of 0.1 without a collapsed
  representation. The raw gate report and the supporting collapse and
  ablation reports must be saved.
- **Stop or retreat:** if the token baseline is below 5%, replace the task
  only through a new documented decision. If latent behavior falls below the
  token baseline, outputs become invalid, or slots collapse, return to Stage
  0C or 0D diagnosis. Do not start Stage 1.

The current Pythia implementation is code-complete, but this roadmap does not
claim that Stage 0E has passed. No current trainability proposal or report is
present on `master`; the Stage 0B and 0C rungs above define the prerequisite
for that future implementation scope. Historical OpenSpec proposals remain
evidence of past design decisions and are not current gate reports.

## Downstream architecture stages

These stages start only after Stage 0E passes. Each stage keeps the same
evidence vocabulary and must publish a fresh report before the next stage.

### Stage 1: persistent state and two clocks

- **Model and scale:** one fixed-size recurrent replacement for the Stage 0E
  core, parameter-matched to it within 10%. Select Mamba or xLSTM before code.
- **Task and data:** `assoc`, `difficulty-mix`, and `split-classify` streams
  with explicit session boundaries and at least 10,000 events per seed.
- **Objective:** answer prediction plus adaptive halting and output-gate loss.
- **Matched baseline:** the frozen Stage 0E subject on the same long stream.
- **Pre/post metrics:** short-stream versus long-stream accuracy and
  retention, valid emissions, and Spearman correlation between item difficulty
  and compute steps.
- **Budget:** five seeds, four session boundaries, and at most two GPU-hours
  per seed.
- **Pass condition:** long-stream accuracy is within 5 percentage points of the
  short-stream baseline, retention does not fall by more than 2 points, and
  compute-step correlation is at least 0.5.
- **Stop or retreat:** keep the Stage 0E transformer with a bounded sliding
  window. Do not add fast weights.

### Stage 2: fast-weight hippocampus

- **Model and scale:** the passing Stage 1 core plus a bounded fast-weight
  module. Report slow-core and fast-weight parameters separately.
- **Task and data:** 100 taught facts on `assoc`, five retention tasks, and
  hostile-input streams using the existing event flag.
- **Objective:** online self-supervised prediction with surprise-weighted,
  clipped, trust-region updates.
- **Matched baseline:** the same Stage 1 subject with fast updates disabled.
- **Pre/post metrics:** time to first use, old-task retention, update norm,
  non-finite count, and attack success rate.
- **Budget:** five seeds, 100 facts, 10,000 stream events, and three attack
  configurations per seed.
- **Pass condition:** the new fact is first used within three probes, old-task
  retention is within 2 points of the frozen baseline, and hostile-input
  attack success is at most 5%.
- **Stop or retreat:** replace live updates with a bounded external read/write
  memory and retain the rest of the design.

### Stage 3: idle consolidation

- **Model and scale:** the passing Stage 2 system plus a latent replay
  generator and idle scheduler. No raw input is stored by the primary design.
- **Task and data:** five hot-learned skills followed by ten idle
  consolidation cycles and the same retention probes.
- **Objective:** replay training of the slow core with EWC protection.
- **Matched baseline:** Stage 2 without consolidation and the Stage -1 raw
  replay reference where applicable.
- **Pre/post metrics:** old-skill retention, consolidation gain, new-skill
  accuracy, idle budget consumed, and a raw-data storage audit.
- **Budget:** five seeds, five skills, ten idle cycles, and a fixed replay
  episode count per cycle.
- **Pass condition:** final old-skill retention beats the replay-free baseline
  by at least 5 points, no old skill falls by more than 2 points during
  consolidation, and the audit finds no raw input storage.
- **Stop or retreat:** use a bounded compressed episodic buffer and record that
  the no-raw-data property was not achieved.

### Stage 4: bottlenecked dual module

- **Model and scale:** a Planner and Worker with total parameters matched to
  the passing Stage 3 single stack within 1%.
- **Task and data:** the Stage 3 streams and a held-out difficulty mix.
- **Objective:** task loss plus the selected information-bottleneck objective
  and credit-assignment method.
- **Matched baseline:** the equal-parameter single stack from Stage 3.
- **Pre/post metrics:** accuracy, retention, compute per input, and channel
  bandwidth.
- **Budget:** five paired seeds and 10,000 events per stream.
- **Pass condition:** the split improves mean held-out accuracy or retention by
  at least 3 points without a greater than 2-point regression in the other
  primary metric.
- **Stop or retreat:** delete the split and retain the single stack if it does
  not beat its matched baseline. Record the measured failure.

### Stage 5: always-on runtime

- **Model and scale:** the surviving Stage 3 single stack or Stage 4 split,
  with no architecture change during runtime validation.
- **Task and data:** seven consecutive days of local use with a fixed probe set
  taught on day 1, at least two input sessions per day, and three planned
  restarts.
- **Objective:** interactive response, online learning, and idle consolidation
  under the runtime clock.
- **Matched baseline:** a persistence-only wrapper with learning and
  consolidation disabled.
- **Pre/post metrics:** day-1 and day-7 probe accuracy, post-restart recall,
  valid output rate, crash recovery, and idle work completed.
- **Budget:** three independent seven-day runs with the same probe protocol.
- **Pass condition:** day-7 probe accuracy is within 2 points of day 1 after
  all restarts, valid output remains at 100%, and no persistent state is lost.
- **Stop or retreat:** do not ship the runtime. Return to the last passing
  offline stage and record the failure mode.

## Cross-cutting work

- **Audit channel:** retain the narration decoder so latent state can be
  inspected without treating narration as proof of reasoning.
- **Latent collapse guards:** report variance and covariance from the first
  latent objective. A lower loss never overrides a collapsed representation.
- **Poisoning:** test hostile input before any fast-weight or always-on claim.
- **Instrumentation:** report halting steps, memory write magnitude, channel
  bandwidth, consolidation gain, seed spread, and exact run identity.

## Kill criteria

- The protected Fashion-MNIST baseline cannot be reproduced from its recorded
  configuration and data identity.
- The tiny transformer cannot pass fixed-set trainability or held-out behavior
  without increasing scope beyond the defined budget.
- The controlled latent change does not improve its declared long-context gate.
- The larger latent core remains below the matched token baseline or collapses
  its workspace.
- Fast updates cannot be stabilized, or consolidation cannot transfer without
  forgetting under its retreat option.

Any kill criterion ends the relevant integration claim. The result is written
up and the project retreats to the last passing rung instead of adding another
model or stage.

## Repository and implementation boundaries

Historical OpenSpec changes under `openspec/changes/archive/` remain unchanged.
They explain earlier architecture decisions but do not establish current
status. The current PBI changes the roadmap and status documents only. It does
not implement the tiny transformer, launch training, weaken Stage -1 or Stage 0
contracts, or delete historical artifacts.
