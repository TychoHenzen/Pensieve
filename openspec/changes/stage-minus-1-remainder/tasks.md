## 1. Subject protocol and isolation (phase 3)

- [x] 1.1 Create `eval/subject/__init__.py` with the `Subject` ABC: `observe(event)`, `answer(probe)`, `idle(budget)`, `snapshot()`, `restore(state)`, `cost()`. Define `CostCounters` as a frozen dataclass with `steps`, `flops`, `wall_seconds`.
- [x] 1.2 Create `eval/subject/isolation.py` with `isolated_answer(subject, probe)` that snapshots, calls `answer`, restores, and returns the answer. This is the harness-side isolation wrapper.
- [x] 1.3 Create `eval/subject/oracles/perfect_memory.py`. Stores every observed key-value pair, answers any probe it has seen. Pure Python, no torch.
- [x] 1.4 Create `eval/subject/oracles/forgetful.py`. Remembers only the most recent item. Answers correctly if the probe asks about the last thing it saw, wrong otherwise.
- [x] 1.5 Create `eval/subject/oracles/chance.py`. Answers randomly from the stream's vocabulary. Seeded for reproducibility.
- [x] 1.6 Create `eval/subject/oracles/task_wiper.py`. Learns the current task perfectly, destroys the previous task on every `Boundary(TASK_SWITCH)`.
- [x] 1.7 Create `eval/subject/oracles/cheater.py`. Reads the truth answer during `answer()` and stores it. Learns from probes on purpose.
- [x] 1.8 Write `tests/subject/test_protocol.py`: test that `Subject` cannot be instantiated without all methods, test `CostCounters` immutability and defaults, test cost monotonicity on a simple oracle.
- [x] 1.9 Write `tests/subject/test_isolation.py`: the integrity test. Run the cheater oracle with and without isolation. Verify the isolated run erases probe learning. Verify an honest oracle (perfect_memory) produces identical results in both modes.
- [x] 1.10 Write `tests/subject/test_snapshot_restore.py`: snapshot, feed noise, restore, verify outputs match the no-noise run. Run on at least two different oracles.
- [x] 1.11 Write `tests/subject/test_oracles.py`: test each oracle's expected behavior on a short assoc stream. Perfect memory recalls everything, forgetful only the last, chance lands near chance rate (binomial test), task wiper drops old tasks on boundary, cheater learns from probes.

## 2. Metrics (phase 4)

- [x] 2.1 Create `eval/metrics/__init__.py` with the `ProbeLogEntry` dataclass (position, probe_id, task_id, teaching_position, correct, cost_counters).
- [x] 2.2 Create `eval/metrics/accuracy.py`: `task_accuracy(log)` returning per-task, per-probe-class, and pooled accuracy.
- [x] 2.3 Create `eval/metrics/retention.py`: `retention_matrix(phases, task_order)` returning `R[i][j]`.
- [x] 2.4 Create `eval/metrics/transfer.py`: `backward_transfer(R)` and `forward_transfer(log, chance_rates)`.
- [x] 2.5 Create `eval/metrics/compute.py`: `compute_per_input(log)` returning steps, flops, wall_seconds derived from CostCounters differences.
- [x] 2.6 Create `eval/metrics/first_use.py`: `time_to_first_use(log, cutoff)` returning median and cutoff share.
- [x] 2.7 Write `tests/metrics/test_accuracy.py`: hand-crafted probe logs with known correct accuracy values.
- [x] 2.8 Write `tests/metrics/test_retention.py`: perfect-memory log (all 1.0), task-wiper log (diagonal 1.0, off-diagonal chance).
- [x] 2.9 Write `tests/metrics/test_transfer.py`: backward transfer strongly negative for task-wiper, zero for perfect-memory. Forward transfer zero for chance oracle.
- [x] 2.10 Write `tests/metrics/test_first_use.py`: perfect-memory first use equals distance to next probe.
- [x] 2.11 Oracle metric integration tests: run each oracle on a short stream through the full pipeline (subject -> probe log -> all metrics) and check every value against paper-derived expectations.

## 3. Instrumentation (phase 5)

- [x] 3.1 Add the four optional instrumentation fields (`halting_steps`, `memory_write_magnitude`, `channel_bandwidth`, `consolidation_gain`) to the run record schema. All default to None.
- [x] 3.2 Create a `VariableComputeOracle` that spends more inner-loop steps on longer arithmetic chains (for use with difficulty-mix).
- [x] 3.3 Write `tests/instrumentation/test_difficulty_correlation.py`: run the variable-compute oracle on a difficulty-mix stream, verify Spearman correlation between steps and difficulty is positive and significant.

## 4. Persistence (phase 6)

- [x] 4.1 Create `eval/run/__init__.py` with the `RunRecord` class: config, code revision, stream hash, environment, probe log path, metric summary path, checkpoint paths.
<!-- status: completed -->
- [x] 4.2 Create `eval/run/config.py`: typed config resolution from a file plus overrides, config hash for the run id.
<!-- status: completed -->
- [x] 4.3 Create `eval/run/checkpoint.py`: atomic checkpoint writes (write-to-temp, rename), checkpoint loading, RNG state save/restore.
<!-- status: completed -->
- [x] 4.4 Create `eval/run/runner.py`: the main run loop that processes a stream through a subject, writes probe log entries, and manages checkpoints on both position and wall-clock schedules.
<!-- status: completed -->
- [x] 4.5 Create `eval/run/retention_policy.py`: checkpoint retention policy (keep last plus every Nth), cleanup of old checkpoints.
<!-- status: completed -->
- [x] 4.6 Write `tests/run/test_config.py`: same config produces same run id, overrides work, env-var paths accepted.
<!-- status: completed -->
- [x] 4.7 Write `tests/run/test_checkpoint.py`: atomic write survives simulated interruption, round-trip save/load, RNG state preserved.
<!-- status: completed -->
- [x] 4.8 Write `tests/run/test_resume.py`: a killed-and-resumed run produces the same probe log as an unbroken run (on a short stream with an oracle subject).
<!-- status: completed -->
- [x] 4.9 Write `tests/run/test_retention_policy.py`: policy keeps last plus every Nth, deletes others.
<!-- status: completed -->

## 5. Baselines and MNIST binding (phase 7) - refinement

This phase has open questions that must be resolved before implementation starts. The first task investigates and resolves them.

- [x] 5.1 **Investigation**: read van de Ven et al. 2020 (the paper and any released code) to extract the exact architecture, hyperparameters, optimizer, and EWC/replay details. Record the exact reported accuracy numbers for class-incremental Split-MNIST (naive, EWC, replay). Determine whether the paper's code can be wrapped as a subject. Write findings to a short note in the change directory.
<!-- status: completed -->
- [x] 5.2 Add `torch` and `torchvision` to `pyproject.toml` dependencies.
<!-- status: completed -->
- [x] 5.3 Create the shared MLP in `eval/baselines/model.py` with configurable depth, width, activation.
<!-- status: completed -->
- [x] 5.4 Implement the naive sequential baseline in `eval/baselines/naive.py`.
<!-- status: completed -->
- [x] 5.5 Implement the joint training baseline in `eval/baselines/joint.py`.
<!-- status: completed -->
- [x] 5.6 Implement the frozen baseline in `eval/baselines/frozen.py`.
<!-- status: completed -->
- [x] 5.7 Implement the EWC baseline in `eval/baselines/ewc.py`.
<!-- status: completed -->
- [x] 5.8 Implement the generative replay baseline in `eval/baselines/replay.py`. Architecture decision (VAE vs simpler) follows from the investigation in 5.1.
<!-- status: completed -->
- [x] 5.9 Create `eval/baselines/param_match.py`: parameter counting, tolerance check, architecture metadata in report.
<!-- status: completed -->
- [x] 5.10 Add MNIST data binding to `eval/stream/generators/split_classify.py`: `data_source="mnist"` config option, 784-element feature vectors, source naming, hash incorporation.
<!-- status: completed -->
- [x] 5.11 Write `tests/baselines/test_baselines_smoke.py`: each baseline runs on a tiny 2-task synthetic stream without crashing, implements the subject protocol, and produces a valid probe log.
<!-- status: completed -->
- [x] 5.12 Write `tests/baselines/test_param_match.py`: matching passes, mismatched rejects, report includes metadata.
<!-- status: completed -->
- [x] 5.13 Write `tests/stream/test_mnist_binding.py`: MNIST features have 784 elements, source is set, hash changes with data source.
<!-- status: completed -->

## 6. Reproduction gate (phase 8) - refinement

This phase depends on the exact numbers from 5.1 and a working set of baselines. The first task locks the pass condition.

- [ ] 6.1 **Investigation**: using the numbers from 5.1, write the pass condition file (`openspec/changes/stage-minus-1-remainder/pass_condition.md`): tolerance per method, ordering requirement, reproducibility requirement, seed count. Lock these before any gate run.
- [ ] 6.2 Create `scripts/run_gate.py`: run naive, EWC, and replay on class-incremental Split-MNIST across five seeds, compute metrics, check pass condition, write report.
- [ ] 6.3 Write `tests/gate/test_gate_smoke.py`: the gate machinery runs on a tiny 2-task synthetic stream in seconds, produces a report, and checks the pass condition format.
- [ ] 6.4 If the gate fails, wrap the paper's released code as a subject (if feasible per 5.1) and run it through the harness to split instrument bugs from method bugs.
- [ ] 6.5 Run the gate for real: five seeds, deterministic kernels, full Split-MNIST. Record the report.

## 7. Freeze (phase 9)

- [ ] 7.1 Version the subject interface as v1: add `SUBJECT_PROTOCOL_VERSION = "1"` to `eval/subject/__init__.py`.
- [ ] 7.2 Version the stream schema as v1: add `STREAM_SCHEMA_VERSION = "1"` to `eval/stream/__init__.py`.
- [ ] 7.3 Version the metric definitions as v1: add `METRICS_VERSION = "1"` to `eval/metrics/__init__.py`.
- [ ] 7.4 Write the one-page contract document (`docs/STAGE-MINUS-1-CONTRACT.md`) that Stage 0 reads: what the subject interface is, what the stream schema is, what the metrics are. Contracts only, not code.
- [ ] 7.5 Update `docs/STAGE-MINUS-1.md` to reflect any changes this work made to what was originally promised.
