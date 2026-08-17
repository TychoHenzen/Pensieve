## Refinement pass: Stage 3 - Idle consolidation

This change is a planning pass. It produces a concrete implementation
plan for Stage 3, not the implementation itself. The deliverable is a
`docs/STAGE-3.md`, a locked pass condition, a spec-delta list, and a
contract document for Stage 4.

### Entry condition

Stage 2 gate passed. The fast-weight hippocampus learns new facts
mid-stream and uses them immediately. Frozen-core skill shows no
measurable loss. The firewall, trust region, and safety machinery are
operational. Poisoning attacks have been run and the defense measured.

### What Stage 3 must produce

From `PLAN.md`:

1. Build the replay generator. It dreams up past-like episodes from the
   hippocampus rather than storing raw input.
2. Train the slow core on the replayed stream during idle periods, with
   elastic-weight-consolidation protection on weights that matter to
   old skills.
3. Add the idle scheduler. It detects no input and enters inward mode.
   It yields at once when input arrives. It partly clears the
   hippocampus after a good transfer.

Gate: after learning several skills hot, consolidation moves them into
the slow core and old skills survive. Must beat a replay-free baseline
on retention without storing raw data.

### Contract obligations inherited

Everything from Stage 2's contract, plus:

- **`idle()` becomes active.** Stage -1 and Stages 0-2 subjects ignore
  `idle(budget)`. Stage 3 is the first stage where `idle()` does real
  work: it runs the consolidation loop. The budget parameter limits how
  much consolidation happens per idle event.
- **`snapshot()`/`restore()` must cover consolidation state.** The slow
  core's EWC Fisher information, the replay generator's parameters, and
  the hippocampus clearing state all need to round-trip exactly.
- **The replay baseline from Stage -1 is the reference.** The gate
  says "beat a replay-free baseline on retention." The Stage -1 replay
  baseline stores raw data and replays it. The Stage 3 system must
  match or beat that retention without storing raw data. The refinement
  must define the comparison precisely.

### Open decisions the refinement must close

1. **Replay generator architecture.** The generator "dreams up
   past-like episodes from the hippocampus." The refinement must
   specify: is this a VAE (like the Stage -1 replay baseline), a
   diffusion model, or a direct readout from fast weights? What does
   "past-like" mean in the latent workspace (concept slots, not
   pixels)? How is the generator trained, and on what signal?

2. **EWC on the slow core.** The Stage -1 EWC baseline collapsed in
   class-incremental Split-MNIST (accuracy near chance). The refinement
   must explain why EWC works here when it did not work there.
   Candidates: the replay stream provides a rehearsal signal that EWC
   alone lacks, the slow core's parameter space is more amenable, or a
   different consolidation objective is needed. The refinement must
   name the objective and justify it.

3. **Idle scheduler design.** The refinement must specify: how does the
   scheduler detect "no input" (wall-clock timeout? explicit idle
   events from the runtime?)? What triggers "inward mode"? How does it
   yield when input arrives (interrupt mechanism vs. cooperative
   check)? How much consolidation happens per idle budget unit?

4. **Hippocampus clearing.** "It partly clears the hippocampus after a
   good transfer." The refinement must define: what "good transfer"
   means (a metric threshold on the slow core's post-consolidation
   accuracy?), what "partly" means (which fast weights decay, by how
   much?), and what happens if transfer is not good (retry? keep the
   fast weights?).

5. **Gate: "old skills survive."** The retention matrix R[i][j] is the
   instrument. The refinement must specify: how many skills, how long
   the stream, what the minimum R[i][last] must be for each old skill,
   and over how many seeds.

6. **Gate: "without storing raw data."** The refinement must define
   the audit that proves no raw data is stored. Is it an architectural
   guarantee (the replay generator has no buffer), a runtime check, or
   a formal proof that the generator's capacity is smaller than the
   data it replays?

7. **Training the slow core on replay.** The refinement must specify
   the training loop: how many replay episodes per idle unit, what loss
   function, what learning rate, and how EWC regularization interacts
   with the replay gradient.

### Cross-cutting items this stage owns

- **Instrumentation: consolidation gain.** The `consolidation_gain`
  field in `eval/instrumentation.py` is defined but empty. Stage 3
  fills it. The refinement must specify what this measures (change in
  slow-core accuracy on old tasks after one consolidation cycle?) and
  how it flows into the run record.

### Required outputs of this refinement pass

1. `docs/STAGE-3.md` - phased implementation plan.
2. A locked pass condition document.
3. A spec-delta list.
4. `docs/STAGE-3-CONTRACT.md` for Stage 4.

### Retreat option

From PLAN.md: "Keep a compressed episodic buffer instead of pure
generative replay." The refinement must specify what "compressed
episodic buffer" means concretely: what format, what compression, what
capacity, and how it feeds the slow-core training loop. This retreat
trades the "no raw data" property for a simpler mechanism.

### Estimated effort

Refinement pass: 1-2 sessions. Implementation: weeks. Generative replay
in latent space is novel; the replay generator has no obvious
off-the-shelf architecture.

### Impact

- New package: `consolidate/` gains the replay generator, the idle
  scheduler, and the EWC-protected training loop.
- Modified: `memory/` for the hippocampus clearing logic.
- Modified: `core/` for the slow-core training interface.
- New or extended: `train/` gains consolidation training scripts.
- `eval/instrumentation.py` starts receiving real `consolidation_gain`
  data.
- `Idle` events in the stream, previously ignored, now trigger real
  work.
