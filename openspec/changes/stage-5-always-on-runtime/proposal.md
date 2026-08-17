## Refinement pass: Stage 5 - The always-on runtime

This change is a planning pass. It produces a concrete implementation
plan for Stage 5, not the implementation itself. The deliverable is a
`docs/STAGE-5.md`, a locked pass condition, and a spec-delta list.

### Entry condition

Stage 3 gate passed (at minimum). Stage 4 has either shipped or been
deleted. The core architecture is frozen: the single stack from Stage 3
or the dual module from Stage 4, whichever survived. The fast-weight
hippocampus, consolidation loop, and all safety machinery are
operational.

### What Stage 5 must produce

From `PLAN.md`:

1. Wire the pieces into one long-lived process: perceive, encode, plan,
   loop, write memory, occasionally speak, and consolidate whenever
   idle.
2. Persist all state to disk so the process can restart with its memory
   intact.
3. Ship a small chat surface. A system that learns from one user needs
   a user.

Gate: the system runs for a week of real use and remembers across
restarts. Teach it something on day one. It must still know that thing
on day seven.

### Contract obligations inherited

Everything from the surviving prior stage's contract, plus:

- **`snapshot()`/`restore()` becomes disk persistence.** The existing
  contract requires exact in-memory round-trips. Stage 5 extends this
  to on-disk serialization and deserialization across process
  restarts. The refinement must confirm that the serialization format
  preserves the exact-restore guarantee. Floating-point reproducibility
  across save/load is the specific risk.
- **`idle()` drives real consolidation on a real clock.** In the eval
  harness, `idle(budget)` is called by the runner. In the runtime,
  idle is detected by absence of user input over a wall-clock timeout.
  The refinement must bridge these two: the runtime calls `idle()` on
  the same subject interface, but triggered by a real clock instead of
  a stream event.
- **The eval harness still applies.** The runtime is a deployment
  wrapper, not a replacement for the harness. The gate requires one
  week of real use, but the system must still pass the harness tests
  (staged gates from prior stages) after that week. The refinement
  must specify how harness runs happen on the deployed system.

### Open decisions the refinement must close

These decisions depend on the final architecture (post Stage 3 or 4).
They name what the pass must address, not what can be answered today.

1. **Process architecture.** One long-lived Python process, or a
   supervisor that restarts a worker? The refinement must specify the
   process model, signal handling (input arrival interrupts
   consolidation), and crash recovery.

2. **Disk persistence format.** The refinement must specify the
   serialization format for all persistent state: slow-core weights,
   fast weights, workspace, consolidation state, EWC Fisher matrices.
   `torch.save` with atomic write-then-rename is the Stage -1
   precedent. The refinement must confirm it is sufficient or propose
   an alternative.

3. **Chat surface.** "A small chat surface" is deliberately minimal.
   The refinement must specify: terminal-based (stdin/stdout), a local
   web UI, or both? What does the UI show beyond the text exchange?
   Does it expose the audit channel (narration decoder from Stage 0)?

4. **Input/idle boundary.** The refinement must specify the timeout
   from last input to idle-mode entry, whether the timeout is
   configurable, and how input arrival interrupts a running
   consolidation cycle (cooperative yielding vs. preemption).

5. **Gate: "remembers across restarts."** The refinement must specify
   how restarts happen during the one-week gate (scheduled? random?
   simulated crash?), how many restarts, and what "remembers" means
   quantitatively (accuracy on a probe set taught on day one, measured
   daily).

6. **Gate: one week of real use.** The refinement must define "real
   use": who uses the system (the developer? a scripted user
   simulation?), what they do (conversation? fact teaching? skill
   training?), and how the week is structured (hours per day, variety
   of topics). A fully scripted week is reproducible but not real. A
   fully manual week is real but not reproducible. The refinement must
   choose a point on this spectrum.

### Cross-cutting items this stage owns

No new cross-cutting items. All four instrumentation fields should be
flowing by this point. The audit channel, collapse guards, and
poisoning defenses are operational from prior stages.

### Required outputs of this refinement pass

1. `docs/STAGE-5.md` - phased implementation plan.
2. A locked pass condition document.
3. A spec-delta list.
4. No contract document for a next stage. Stage 5 is the last stage.

### Retreat option

PLAN.md names no retreat for Stage 5. The runtime is engineering, not
research. If the pieces from Stages 0-3 work, wiring them into a
process is a known problem. The risk is in the one-week gate's
definition, not in the implementation.

### Estimated effort

Refinement pass: 1 session. Implementation: days to weeks, mostly
engineering rather than research. The one-week gate takes one wall-clock
week by definition.

### Impact

- New package: `runtime/` gains the main loop, idle detector, disk
  persistence, and chat surface.
- Modified: nothing in `eval/`, `core/`, `memory/`, or `consolidate/`.
  Stage 5 wraps them; it does not change them.
- New dependency: a web framework if the chat surface is browser-based
  (e.g., Flask or FastAPI for a minimal local server).
