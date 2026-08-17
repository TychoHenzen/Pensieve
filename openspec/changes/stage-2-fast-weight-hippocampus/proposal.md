## Refinement pass: Stage 2 - Fast-weight hippocampus

This change is a planning pass. It produces a concrete implementation
plan for Stage 2, not the implementation itself. The deliverable is a
`docs/STAGE-2.md`, a locked pass condition, a spec-delta list, and a
contract document for Stage 3.

### Entry condition

Stage 1 gate passed. The recurrent backbone holds performance across a
long stream broken into fake sessions. Adaptive halting produces a step
count that correlates with item difficulty. The output gate works.

### What Stage 2 must produce

From `PLAN.md`:

1. Add a memory module whose weights update by gradient descent during
   use, Titans or test-time-training style, writing surprising input
   more strongly.
2. Add the safety machinery in the same stage, not later: weight decay,
   gradient clipping, and a trust region on the fast weights.
3. Add a hard firewall so the fast store can only reach the slow core
   through vetted replay.

Gate: teach the running system a new fact mid-stream. It must use that
fact right away, with no training run. Frozen-core skill must show no
measurable loss over a long stream afterward.

### Contract obligations inherited

Everything from Stage 1's contract, plus:

- **`restore()` must be exact for fast weights.** The fast-weight
  module updates its weights during use. `snapshot()` must capture both
  the slow-core state and the current fast weights. `restore()` must
  roll the fast weights back exactly. This is the hardest restore
  contract in the whole project, because the fast weights change on
  every input.
- **Probe isolation with online learning.** The subject learns from
  every `observe()` call by updating fast weights. Probe isolation
  must undo those updates. If `answer()` also triggers fast-weight
  updates (because the subject processes the probe query), restore
  must undo those too.
- **The `assoc` generator tests this gate directly.** Teach a
  key-value fact, then probe recall at a distance. The `assoc`
  generator already does this. The refinement must confirm whether
  `assoc` is sufficient for the gate or whether additional stream
  patterns are needed.

### Open decisions the refinement must close

1. **Which fast-weight mechanism.** Titans and test-time training (TTT)
   are the two candidates PLAN.md names. They differ in how gradients
   flow, what the learning rule looks like, and how much compute each
   update costs. The refinement must name the mechanism, justify the
   pick, and specify the fast-weight module's size relative to the slow
   core.

2. **Surprise-weighted writing.** "Writing surprising input more
   strongly" requires a surprise signal. The refinement must specify
   what computes surprise (prediction error? reconstruction loss? a
   separate novelty detector?), how surprise scales the learning rate
   or gradient magnitude, and whether the surprise signal is itself
   learned or heuristic.

3. **Safety machinery details.** Weight decay, gradient clipping, and a
   trust region are named. The refinement must specify: what the trust
   region constrains (L2 distance from initialization? KL divergence?),
   what the clipping threshold is, and whether these are
   hyperparameters or learned. The refinement must also define what
   "stable" means quantitatively for the gate.

4. **Firewall design.** "The fast store can only reach the slow core
   through vetted replay." The refinement must specify what the
   firewall is architecturally (a separate module? a read gate? a
   gradient block?), what "vetted" means (who vets, by what criterion),
   and how the firewall interacts with the Stage 3 consolidation loop.

5. **Gate: "no measurable loss."** The gate says frozen-core skill must
   show no measurable loss. The refinement must define "measurable" in
   terms of the retention matrix: what tolerance on R[i][j] for old
   tasks, how many tasks, how long the stream runs, and how many seeds.

6. **Gate: "use that fact right away."** The refinement must define
   "right away" in terms of the `time_to_first_use` metric: within how
   many stream positions of teaching must the first correct answer
   appear?

7. **Poisoning attacks.** From PLAN.md's cross-cutting section: "A
   model that learns from its input can be taught lies. The firewall,
   the trust region, and the vetted-replay path are the defense. Attack
   them in Stage 2, not after deployment." The harness already defines
   a `hostile` flag on events. The refinement must specify what attack
   streams look like, what the attack success metric is, and what
   passing the defense means.

### Cross-cutting items this stage owns

- **Poisoning and attack streams.** Design the attack, run it, and
  measure the defense. The stream schema already has the `hostile` flag.
- **Instrumentation: memory write magnitude.** The
  `memory_write_magnitude` field in `eval/instrumentation.py` is
  defined but empty. Stage 2 fills it. The refinement must specify what
  this measures (L2 norm of the fast-weight update? the gradient norm
  before clipping?) and how it flows into the run record.

### Required outputs of this refinement pass

1. `docs/STAGE-2.md` - phased implementation plan.
2. A locked pass condition document.
3. A spec-delta list.
4. `docs/STAGE-2-CONTRACT.md` for Stage 3.

### Retreat option

From PLAN.md: "Replace live weight updates with an external read and
write memory store. The rest of the design survives." The refinement
must specify what "external memory store" means concretely enough that
retreating to it is a known-cost decision, not a second research
project.

### Estimated effort

Refinement pass: 1-2 sessions (heavy paper reading for Titans and TTT).
Implementation: weeks, because online gradient descent at inference time
is the core research contribution and will need extensive tuning.

### Impact

- New package: `memory/` gains the fast-weight module, the surprise
  signal, the safety machinery, and the firewall.
- Modified: `core/` to wire the fast-weight module into the reasoning
  loop.
- New or extended: `train/` gains scripts for any offline pretraining
  the fast-weight module needs.
- `eval/instrumentation.py` starts receiving real
  `memory_write_magnitude` data.
- New attack-stream configurations and defense metrics.
