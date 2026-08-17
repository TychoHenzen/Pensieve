## Refinement pass: Stage 4 - Bottlenecked dual module

This change is a planning pass. It produces a concrete implementation
plan for Stage 4, not the implementation itself. The deliverable is a
`docs/STAGE-4.md`, a locked pass condition, a spec-delta list, and a
contract document for Stage 5.

Stage 4 is "openly on probation" (PLAN.md). The independent analysis of
the Hierarchical Reasoning Model found its brain-shaped split was not
what produced its results. This stage may be deleted. The refinement
pass must be honest about that risk.

### Entry condition

Stage 3 gate passed. Consolidation moves hot-learned skills into the
slow core. Old skills survive. The system beats a replay-free baseline
on retention without storing raw data.

Additional prerequisite: the Stage 3 results show where the single-stack
architecture's limits are. If the single stack already saturates the
gate benchmarks, Stage 4 has no room to improve, and the refinement
should say so.

### What Stage 4 must produce

From `PLAN.md`:

1. Split the core into a slow Planner and a fast Worker, joined only
   by a narrow channel of a few vectors per planner tick.
2. Regularize the channel with an information-bottleneck objective so
   each side has to send abstractions rather than raw state.
3. Solve credit assignment across the two clocks, starting from deep
   supervision or a learned halting value function.

Gate: the split beats a single stack of equal parameter count on the
same streams. If it does not, delete it, keep the single stack, and
record the two-hemisphere idea as inspiration that was tried and
dropped.

### Contract obligations inherited

Everything from Stage 3's contract, plus:

- **Parameter matching is the gate's core mechanism.** The split must
  be compared against a single stack of equal parameter count. The
  `eval/baselines/param_match.py` enforcer applies here: the planner
  plus the worker must have the same total parameters as the single
  stack. The refinement must specify how parameters are counted across
  the two modules and the channel.
- **All existing gates must still pass.** Splitting the core must not
  regress Stages 0-3 gate results. The refinement must specify whether
  all prior gates are re-run or a subset.

### Open decisions the refinement must close

These decisions depend heavily on measured Stage 1-3 results. The
refinement pass for Stage 4 cannot begin until those results exist.
The list below names what the pass must address, not what can be
answered today.

1. **Planner/Worker split ratio.** How are parameters divided between
   the two modules? The refinement must specify a default split and an
   ablation range.

2. **Channel width and tick rate.** "A few vectors per planner tick."
   The refinement must specify the channel dimension, the planner tick
   rate relative to the worker step rate, and whether these are fixed
   or learned.

3. **Information-bottleneck objective.** The refinement must specify the
   IB loss term, its weight relative to the task loss, and how it is
   optimized (jointly or alternating).

4. **Credit assignment.** "Starting from deep supervision or a learned
   halting value function." The refinement must choose one, justify it,
   and specify the training signal that reaches the planner through the
   bottleneck.

5. **Gate: "beats a single stack."** The refinement must specify: on
   which streams, by how much (a minimum margin?), over how many
   seeds, and on which metrics (accuracy? retention? compute
   efficiency? all three?).

### Cross-cutting items this stage owns

- **Instrumentation: channel bandwidth.** The `channel_bandwidth`
  field in `eval/instrumentation.py` is defined but empty. Stage 4
  fills it. The refinement must specify what this measures (mutual
  information estimate? raw L2 norm of the channel vectors?) and how
  it flows into the run record.

### Required outputs of this refinement pass

1. `docs/STAGE-4.md` - phased implementation plan.
2. A locked pass condition document with an explicit deletion criterion.
3. A spec-delta list.
4. `docs/STAGE-4-CONTRACT.md` for Stage 5, written in two variants:
   one where Stage 4 ships, one where Stage 4 is deleted.

### Retreat option

From PLAN.md: "If it does not [beat the single stack], delete it, keep
the single stack, and record the two-hemisphere idea as inspiration that
was tried and dropped."

The retreat is deletion, not degradation. The refinement must specify
what "record" means: a written post-mortem with the measured numbers and
the conclusion.

### Estimated effort

Refinement pass: 1 session, but it cannot produce useful decisions
until Stage 3 results exist. Implementation: weeks if it proceeds, zero
if the single stack wins.

### Impact

- Modified: `core/` splits into planner and worker submodules with a
  channel between them.
- Modified: `train/` gains scripts for the dual-module training with
  the IB objective.
- `eval/instrumentation.py` starts receiving real `channel_bandwidth`
  data.
- If Stage 4 is deleted, `core/` stays as it was after Stage 3, and
  this change is archived with the post-mortem.
