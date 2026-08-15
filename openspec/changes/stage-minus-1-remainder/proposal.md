## Why

Phases 1 and 2 of STAGE-MINUS-1.md are complete: the stream core passes 158 tests, all three generators produce renderable events, and the STREAM-REWORK fixes have landed. Phases 3 through 9 remain. The harness cannot measure anything until the subject protocol exists (phase 3), cannot score anything until the metrics exist (phase 4), and cannot pass the reproduction gate (phase 8) until baselines run real continual-learning methods on real data. The gate is the exit criterion for Stage -1. Nothing in Stage 0 starts without it.

## What Changes

- Add the subject protocol: `observe`, `answer`, `idle`, `snapshot`/`restore`, `cost`. Probe isolation via snapshot-restore, with an integrity test that catches a cheater.
- Add the probe log and the full metric set: task accuracy, retention matrix, backward and forward transfer, compute per input, time to first use. Each metric gets an oracle test against a subject whose correct values are known on paper.
- Add cost counters (inner-loop steps, estimated FLOPs, wall time) and the instrumentation schema for later signals (halting steps, memory write magnitude, channel bandwidth, consolidation gain).
- Add checkpoint-resume and crash recovery for long runs.
- Add the five baselines (naive sequential, joint training, frozen, EWC, generative replay), the parameter-matching runner, and the real Split-MNIST data binding.
- Run the reproduction gate: Split-MNIST class-incremental, five seeds, compare naive/EWC/replay against van de Ven et al. 2020.
- Freeze v1: version the subject interface, stream schema, and metric definitions.

## Capabilities

### New Capabilities
- `eval/subject-protocol`: The subject interface (`observe`, `answer`, `idle`, `snapshot`/`restore`, `cost`), probe isolation via snapshot-restore, the isolation integrity test, and the oracle subjects (perfect memory, forgetful, chance, task wiper, cheater).
- `eval/metrics`: The probe log schema, the metric set (task accuracy, retention, backward/forward transfer, compute per input, time to first use), and oracle tests that validate each metric against paper-derived expected values.
- `eval/instrumentation`: Cost counters (steps, FLOPs, wall time), the extensible instrumentation schema for halting steps, memory write magnitude, channel bandwidth, and consolidation gain, and the difficulty-mix correlation test.
- `eval/persistence`: Checkpoint-resume by stream position and wall clock, atomic writes, crash recovery, checkpoint retention policy, and the resume-matches-unbroken test.
- `eval/baselines`: The five baseline implementations (naive sequential, joint training, frozen, EWC, generative replay), the parameter-matching runner, and the real Split-MNIST data binding for split-classify.
- `eval/reproduction-gate`: The gate protocol - Split-MNIST class-incremental, five-seed sweep, comparison against van de Ven et al. 2020, written pass condition, and the diagnostic path for when numbers do not reproduce.

### Modified Capabilities
- `eval/generators/split-classify`: Binding real MNIST digit data through the existing `source` field, which the current spec defines but leaves empty.

## Impact

- New package: `eval/subject/` for the protocol, oracle subjects, and isolation machinery.
- New package: `eval/metrics/` for the probe log, metric computations, and oracle metric tests.
- New package: `eval/run/` for config resolution, run records, checkpointing, and resumption.
- New package: `eval/baselines/` for the five baseline implementations and the parameter matcher.
- New file(s) in `scripts/` for the gate reproduction runner.
- New dependency: `torch` and `torchvision` (required for baselines and MNIST).
- `eval/stream/generators/split_classify.py` gains a data-binding path for real features.
- `pyproject.toml` gains the torch dependency and any test fixtures for MNIST.
