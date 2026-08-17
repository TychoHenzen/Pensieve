## Context

Phases 1-2 of Stage -1 are complete: the stream core, all three generators, rendering, token counting, and hashing pass 158 tests. See proposal.md for the full scope of remaining work. The existing code lives in `eval/stream/`. The subject protocol, metrics, run infrastructure, baselines, and the reproduction gate are all unbuilt.

## Goals / Non-Goals

**Goals:**
- Build the subject protocol and oracle subjects (phase 3) as the immediate next step, fully specced and ready for direct implementation.
- Build the probe log and metrics (phase 4) once the subject protocol exists.
- Build instrumentation (phase 5) alongside or after metrics.
- Build checkpoint-resume (phase 6) before long baseline runs.
- Build the five baselines and the reproduction gate (phases 7-8) last, because they depend on everything above.
- Freeze v1 (phase 9) after the gate passes.

**Non-Goals:**
- No model research. The toy models exist only to exercise the harness.
- No real latent-reasoning code. That is Stage 0.
- No GPU optimization of baselines beyond what the paper protocol requires.

## Decisions

### 1. Subject protocol as an abstract base class, not a Protocol

The subject interface uses `abc.ABC` with `@abstractmethod`. A Protocol (structural typing) would let any object with the right methods pass, but subjects carry mutable state and need explicit opt-in. ABC forces implementers to inherit and gets caught at instantiation rather than at first call.

Alternative: `typing.Protocol` with runtime_checkable. Rejected because it cannot enforce snapshot-restore semantics at type-check time, and the inheritance signal is valuable for a small set of known subjects.

### 2. Oracle subjects live in `eval/subject/oracles/`, one file each

Five oracles (perfect_memory, forgetful, chance, task_wiper, cheater) each get their own module. They import the subject ABC and nothing from `torch`. They are pure-Python so they run fast in CI without GPU.

Alternative: all oracles in one file. Rejected because the cheater oracle has fundamentally different semantics (it is supposed to fail isolation) and deserves separate test coverage.

### 3. Probe isolation is a harness-side wrapper, not a subject responsibility

The harness calls `snapshot()`, `answer()`, `restore()`. The subject never knows it is being isolated. This keeps the trust boundary clear: the subject cannot opt out of isolation.

The read-only flag is an optional optimization. A subject must pass a side-by-side equivalence check before the harness trusts it.

### 4. Metrics are pure functions over the probe log

Each metric function takes the probe log (a list of `ProbeLogEntry` dataclasses) and returns a result. No metric reads internal subject state. This keeps metrics testable: hand-craft a probe log with known values, check the output.

### 5. Run infrastructure uses JSON, not a database

Run records are JSON files in a directory. The probe log is newline-delimited JSON (one row per line, append-only). Checkpoints are PyTorch `torch.save` files written atomically via write-to-temp-then-rename. No SQLite, no HDF5 - the data is small and the access pattern is append-then-read-once.

Alternative: SQLite. Rejected because it adds a dependency for no gain at this scale, and atomic writes to a single-writer append log are simpler as files.

### 6. Baselines use a shared small MLP architecture

All five baselines share a configurable MLP (depth, width, activation). The parameter-matching runner counts `sum(p.numel() for p in model.parameters())`. The default architecture matches what van de Ven et al. 2020 used for their Split-MNIST experiments. Depth, width, and optimizer are recorded in every report.

### 7. MNIST data loaded lazily, cached on disk

`torchvision.datasets.MNIST` downloads to a configured data directory. The generator binds into it at generation time when `data_source="mnist"` is set. The download happens once and is cached. The repository stores no MNIST data.

### 8. Reproduction gate is a script, not a test

The gate is a script in `scripts/run_gate.py` that runs the three methods across five seeds, computes metrics, checks the pass condition, and writes a report. It is not a pytest test because it takes GPU-hours to run. A smaller smoke test in `tests/` verifies the gate machinery works on a tiny 2-task stream in seconds.

## Risks / Trade-offs

- [Snapshot-restore overhead on GPU models] GPU state (model weights, optimizer state, RNG state) is large. `snapshot()` must deep-copy all of it. For Stage -1 toy models this is cheap. For later stages it may need streaming to disk. Mitigation: measure snapshot time as a fraction of run time in the gate report.

- [EWC Fisher information computation cost] Computing the Fisher matrix after each task adds a full pass over the task's data. For Stage -1 toy models this is fine. Mitigation: compute Fisher only at task boundaries, not continuously.

- [Generative replay quality on MNIST] The generator must produce recognizable digits for replay to work. A small VAE is the standard choice. Mitigation: use the same architecture and hyperparameters as van de Ven et al. 2020.

- [Tolerance selection for the gate] The tolerance must be written before the first run. Too tight means random variation fails the gate. Too loose means the gate does not catch real bugs. Mitigation: pull the exact numbers from the paper, run three preliminary seeds, and set the tolerance at 3 standard deviations of the observed spread.

## Open Questions

- The exact tolerance values for the reproduction gate depend on numbers pulled from the van de Ven et al. 2020 paper. These will be filled in during the baselines phase, before the first gate run. This does not change the specs or the approach.
- Whether the generative replay baseline should use a VAE or a simpler generator. The paper uses a VAE. If its complexity is disproportionate for Stage -1, a simpler generator that still reproduces the result is acceptable. This will be resolved when implementing the replay baseline.
