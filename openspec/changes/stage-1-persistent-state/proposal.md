## Refinement pass: Stage 1 - Persistent state and two clocks

This change is a planning pass. It produces a concrete implementation
plan for Stage 1, not the implementation itself. The deliverable is a
`docs/STAGE-1.md`, a locked pass condition, a spec-delta list, and a
contract document for Stage 2.

### Entry condition

Stage 0 gate passed. A latent-reasoning subject matches token
chain-of-thought accuracy on the gate task (dataset and tolerance
defined by the Stage 0 pass condition). The workspace, edge codecs, and
latent loop are built and measured. The audit channel and latent collapse
guards are operational.

### What Stage 1 must produce

From `PLAN.md`:

1. Replace the backbone with a fixed-size recurrent state, Mamba or
   xLSTM style, so the state carries across the whole stream and never
   resets.
2. Add adaptive halting on the inner loop, PonderNet style, so a cycle
   can run without emitting anything.
3. Add the output gate: a separate learned decision about when to speak.

Gate: performance holds across an artificially long stream broken into
fake sessions. Compute per input tracks item difficulty.

### Contract obligations inherited

Everything from Stage 0's contract, plus:

- **`restore()` must be exact for recurrent state.** A fixed-size
  recurrent state that persists across the whole stream must still
  snapshot and restore exactly. This is harder than Stage 0's workspace
  because the recurrent state accumulates over thousands of events.
  Floating-point drift across a save/load cycle would break probe
  isolation.
- **`cost()` must reflect adaptive halting.** The `steps` counter in
  `CostCounters` must count inner-loop iterations faithfully. PonderNet
  halting produces a variable step count per input. The
  `difficulty-mix` generator and its correlation test depend on this
  signal being real.
- **`idle()` still does nothing.** Stage 1 subjects receive `Idle`
  events but do not consolidate. That is Stage 3. The refinement must
  confirm that the recurrent state does not update during idle.

### Open decisions the refinement must close

1. **Which recurrent backbone.** Mamba (structured state space) and
   xLSTM (extended LSTM with exponential gating) are the two candidates
   PLAN.md names. They differ in how they handle long-range memory, how
   they scale, and how easy they are to snapshot. The refinement must
   name the architecture, justify the pick, and specify the model size
   that keeps parameter matching feasible against the Stage -1
   baselines.

2. **How to replace the Stage 0 backbone.** Stage 0 wraps a pretrained
   transformer and feeds hidden states back. Stage 1 replaces that
   transformer with a recurrent model. The refinement must specify
   whether the recurrent model is pretrained (on what data, for how
   long) or trained from scratch during Stage 1, and whether the edge
   codecs from Stage 0 survive unchanged or need retraining.

3. **Adaptive halting design.** PonderNet learns a per-step halting
   probability. The refinement must specify the halting head
   architecture, the regularization (geometric prior penalty from the
   PonderNet paper), the minimum and maximum step counts, and how the
   halting decision interacts with the output gate.

4. **Output gate design.** "A separate learned decision about when to
   speak" is underspecified. The refinement must decide: is the output
   gate a binary classifier on the workspace state? Does it share
   parameters with the halting head? How is it trained (when is "speak"
   the right answer in the training data)? How does `observe()` return
   an emission versus `None`?

5. **Gate stream and metrics.** The gate says "performance holds across
   an artificially long stream broken into fake sessions." The
   refinement must specify: which generator produces this stream, how
   long it is, what "fake sessions" means (presumably
   `Boundary(SESSION_END)` events), and what "holds" means
   quantitatively (accuracy within what tolerance of a short-stream
   baseline). The second clause, "compute per input tracks item
   difficulty," maps to the `difficulty-mix` generator and the Spearman
   correlation test. The refinement must state the minimum correlation
   coefficient.

6. **Training procedure.** Stage 1 trains a new backbone from scratch
   or fine-tunes. The refinement must specify the training data, the
   objective (next-latent-state prediction? reconstruction? a mix?),
   the schedule, and whether training happens offline or interleaved
   with the evaluation stream.

### Cross-cutting items this stage owns

- **Instrumentation: halting steps.** The `halting_steps` field in
  `eval/instrumentation.py` is defined but empty. Stage 1 is the first
  stage that fills it. The refinement must specify how the per-event
  step count flows into the run record.

### Required outputs of this refinement pass

1. `docs/STAGE-1.md` - phased implementation plan.
2. A locked pass condition document.
3. A spec-delta list.
4. `docs/STAGE-1-CONTRACT.md` for Stage 2.

### Retreat option

PLAN.md names no retreat for Stage 1. If the recurrent backbone cannot
hold performance across a long stream, the issue is architectural. The
refinement should name a fallback (e.g., keep the Stage 0 transformer
with a sliding window) even though PLAN.md does not.

### Estimated effort

Refinement pass: 1-2 sessions. Implementation: weeks, because
pretraining a recurrent backbone from scratch is the likely path.

### Impact

- Modified package: `core/` gains the recurrent backbone, halting head,
  and output gate.
- Modified package: `workspace/` may need changes if the concept-slot
  interface changes for the recurrent state.
- New or extended: `train/` gains the backbone pretraining scripts.
- `eval/instrumentation.py` starts receiving real `halting_steps` data.
- `pyproject.toml` gains dependencies for the chosen backbone (e.g.,
  `mamba-ssm` or an xLSTM implementation).
