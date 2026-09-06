# autoresearcherUI relevance to Pensive

**Date:** 2026-09-06
**Scope:** Review the upstream `Fchaubard/autoresearcherUI` repository and compare its architecture with the current Pensive repository.
**Decision:** Relevant as an experiment control plane and observability reference. Not suitable as a direct dependency or as Pensive's evaluation authority.

## Executive answer

`autoresearcherUI` addresses a real problem around Pensive: long-running research needs a visible queue of experiments, baseline comparisons, progress telemetry, failure handling, and a record of why a run was kept or discarded. Its strongest ideas for Pensive are:

1. A pre-compute scoping gate that turns a research question into citation-backed ideas with cheap kill tests.
2. A code and preflight gate before expensive runs.
3. A structured idea queue and research journal.
4. A local dashboard for scalar progress, logs, hardware health, and artifacts.

The direct application should be a thin Pensive adapter. Pensive must retain ownership of stream identity, probe isolation, metric calculation, multi-seed gates, checkpoint compatibility, and final claims. `autoresearcherUI` can display those results and schedule bounded experiments, but its headline metric must not replace Pensive's gate verdict.

## What the external project actually is

The upstream README describes a self-hosted cockpit that combines an autonomous research agent, experiment tracking, GPU monitoring, literature review, code review, and paper-mode workflows. Its default flow scopes a direction before GPU use, runs a baseline, explores an idea queue, and later hands selected results to an author workflow. See the [upstream README](https://github.com/Fchaubard/autoresearcherUI/blob/main/README.md).

The implementation is a single FastAPI process with a static dashboard, SQLite metadata, DuckDB metric storage, subprocess training runs, and background monitor services. The architecture is stated in the [README architecture section](https://github.com/Fchaubard/autoresearcherUI/blob/main/README.md#architecture) and the data model separates relational run metadata from metric time series in [`models.py`](https://github.com/Fchaubard/autoresearcherUI/blob/main/backend/app/models.py).

Its orchestrator has a narrower contract than the product description suggests. It launches an experiment repository's `train.py`, extracts one configured scalar from stdout, compares that value with a baseline, and marks the run kept, discarded, unclear, or crashed. That behavior is visible in [`orchestrator.py`](https://github.com/Fchaubard/autoresearcherUI/blob/main/backend/app/orchestrator.py). The end-to-end test proves the wiring with a FakeAgent, CPU training, and a bundled tiny-SGD project. The real Claude path is explicitly documented as a stub in [`agent.py`](https://github.com/Fchaubard/autoresearcherUI/blob/main/backend/app/agent.py), so the test does not prove real Claude-driven research or GPU experiment correctness. See [`tests/e2e_test.py`](https://github.com/Fchaubard/autoresearcherUI/blob/main/tests/e2e_test.py).

The tracking SDK is intentionally small. `arui.log()` sends numeric scalar points asynchronously, while `summary` and `log_artifact()` carry run-level values and artifact paths. See [`arui/__init__.py`](https://github.com/Fchaubard/autoresearcherUI/blob/main/arui/__init__.py). This is useful for a dashboard, but it is not a replacement for structured probe logs or exact gate reports.

The scoping gate is the most transferable research workflow. It searches arXiv and Semantic Scholar, asks a council to pressure-test the direction, presents the result for user revision, then writes directives, a literature review, lessons, and a setup brief before starting the agent. The flow is implemented in [`scoping.py`](https://github.com/Fchaubard/autoresearcherUI/blob/main/backend/app/scoping.py). The code-bless path adds another useful pattern: deterministic preflight checks run before model review, and the run API rejects ordinary runs until reviewers approve the codebase. See [`council.py`](https://github.com/Fchaubard/autoresearcherUI/blob/main/backend/app/council.py) and [`api.py`](https://github.com/Fchaubard/autoresearcherUI/blob/main/backend/app/api.py).

There are maturity and deployment caveats. The README calls the repository `v0.1.0`, while the GitHub Releases page currently lists `v0.0.2` as the latest release. The `main` branch is active, with commits dated 2026-09-05. If Pensive adopts code or an API shape, it should pin a reviewed commit rather than track `main`: [README](https://github.com/Fchaubard/autoresearcherUI/blob/main/README.md), [releases](https://github.com/Fchaubard/autoresearcherUI/releases), [commit history](https://github.com/Fchaubard/autoresearcherUI/commits/main).

## Fit with Pensive

Pensive already has the parts that must remain authoritative. [`docs/PLAN.md`](PLAN.md) defines staged gates and requires matched comparisons. [`docs/STAGE-MINUS-1-CONTRACT.md`](STAGE-MINUS-1-CONTRACT.md) defines seeded non-stationary streams, isolated probes, retention and transfer metrics, compute counters, and reproducible run identity. The current evaluation runner drives a `Subject` through those streams, snapshots around probes, and resumes from checkpoints in [`eval/run/runner.py`](../eval/run/runner.py). The current alternating trainer has its own phase scheduler, held-out evaluation, JSON progress records, and identity-bearing checkpoints in [`train/run_alternating.py`](../train/run_alternating.py) and [`train/alternating_checkpoint.py`](../train/alternating_checkpoint.py).

That makes the projects complementary rather than interchangeable:

| Capability | Relevance to Pensive | Correct boundary |
|---|---:|---|
| Run dashboard, logs, hardware status | High | Read derived Pensive manifests and progress records. |
| Idea queue and cheap kill tests | High | Queue bounded changes against a fixed Pensive identity. |
| Literature scoping before compute | Medium-high | Use for Stage 1+ design choices and major experiments. |
| Code-bless and preflight | High | Prefer deterministic Pensive tests and specs. LLM review is advisory. |
| Baseline-versus-run charting | Medium-high | Display comparisons, but let Pensive gate code decide pass/fail. |
| Multi-GPU scheduler | Medium, later | Add only after one-device reproducibility and resource isolation are proven. |
| Autonomous source editing | Low, risky | Do not give it unrestricted ownership of the Pensive checkout. |
| Paper Mode | Low now, useful later | Add after a mechanism has passed its measured gate and needs ablations. |

## Important mismatches

### 1. One `train.py` and one scalar do not describe Pensive

The external orchestrator assumes an experiment can be launched as `python train.py` and judged by one stdout metric. Pensive has separate training runners, evaluation subjects, stream generators, baselines, gate scripts, and structured artifacts. A single scalar such as `answer_exact_match` cannot represent probe isolation, retention, backward transfer, forward transfer, compute per input, or five-seed reproducibility.

Use the external headline metric only as a dashboard summary. The Pensive adapter must preserve the complete gate report and probe log as the canonical evidence.

### 2. Pensive identity is stricter

Pensive binds results to dataset revisions, ordered item identities, stream configuration, seeds, code identity, and checkpoint state. An experiment queue that changes a prompt, dataset selection, evaluator, or seed without changing the run identity would make the dashboard misleading. Every queued idea therefore needs an immutable resolved manifest before launch.

### 3. Parallel runs can invalidate comparisons

`autoresearcherUI` is designed to keep GPU slots full. Pensive's frozen Qwen path is expensive and sensitive to device, memory, and deterministic settings. Two concurrent runs on one consumer GPU can change runtime, memory pressure, and failure behavior. Start with one active Pensive run per device. Add concurrency only with a measured resource and reproducibility gate.

### 4. The storage models differ

The external dashboard uses SQLite and DuckDB. Pensive's evaluation design uses plain JSON files and NDJSON probe logs, and its `RunRecord` already names the expected config, revision, stream hash, environment, probe log, and metric-summary files in [`eval/run/__init__.py`](../eval/run/__init__.py). The current runner writes the probe log and checkpoints, but does not yet materialize every named run-record file. The cleanup ledger also records that instrumentation fields are defined but not attached to run records. This is a Pensive-side integration gap, not a reason to make the external database canonical. See [`CLEANUP_LEDGER.md`](../CLEANUP_LEDGER.md).

### 5. The default operational model is not Pensive's

The external installer expects Bash, tmux, and optional cloudflared, and the real agent path uses the Claude Code CLI. Its [`setup.sh`](https://github.com/Fchaubard/autoresearcherUI/blob/main/setup.sh) is aimed at a Linux or macOS GPU node. Pensive is currently developed from Windows PowerShell with a local `.venv`. A native Windows integration would need a separate launcher and process-management layer. The external system's use of a dangerous permissions mode is also incompatible with handing an autonomous agent unrestricted control of the Pensive checkout.

## Recommended application plan

### Phase A: build a read-only Pensive run adapter

Do this before adding autonomous idea execution. The adapter should wrap existing commands rather than move training logic into a new service.

For each run, create an immutable manifest containing:

- the Git revision and exact command;
- the resolved training or evaluation configuration;
- the dataset revision, ordered selection, stream hash, and seed;
- the device and relevant package/runtime identity;
- the baseline or gate identity used for comparison;
- paths to checkpoints, JSONL progress, probe logs, and final reports.

Reuse Pensive's existing `config_hash`, `training_identity`, checkpoint metadata, and gate report formats. Send only derived scalar progress to a dashboard. Keep the raw artifacts local and make the dashboard link to them.

### Phase B: expose Pensive-specific telemetry

Map existing records to dashboard fields without flattening their meaning:

- training: `global_step`, `epoch`, active method, loss, shared variance, held-out exact match, optimizer calls, elapsed time, and ETA;
- evaluation: per-probe correctness, task accuracy, retention, transfer, first-use distance, and cost-counter deltas;
- lifecycle: queued, preflight, running, checkpointed, resumed, gate-passed, gate-failed, crashed, or manually stopped.

The UI may show a `best` curve, but it must also show the exact gate criteria and the artifact that produced each value. A run that improves loss but fails the Stage 0 or Stage -1 contract must remain failed or inconclusive.

### Phase C: use the idea queue as a bounded research queue

Represent each idea as one change, one expected mechanism, one cheap kill test, and one acceptance gate. Keep the queue stages small:

1. import and unit smoke check;
2. one development-limited run;
3. one controlled comparison against the protected baseline;
4. the required multi-seed gate;
5. full training or ablation only after the earlier stage passes.

The queue may prioritize work, but it must not silently mutate the Pensive experiment identity or skip a required gate. Human approval should remain at the boundary where an idea becomes a real branch or a long GPU run.

### Phase D: adapt the preflight and code-bless gate

Replace generic reviewer checks with Pensive checks:

- targeted tests for the changed subsystem;
- strict OpenSpec validation when a spec is involved;
- dataset and prompt identity validation;
- checkpoint resume compatibility;
- `git diff --check` and repository lint checks;
- a development run that demonstrates actual learning or evaluation behavior, not only imports.

An LLM council can find missing tests or contradictory assumptions, but it must not be the final authority for a scientific gate. Pensive's deterministic reports remain authoritative.

### Phase E: add paper workflow later

Paper Mode could consume passed gate artifacts and schedule ablations, but it should wait until a mechanism has a meaningful result. Before then, a paper UI risks turning incomplete experiments into polished claims.

## What not to do

- Do not replace `eval/run/runner.py` with the external runner.
- Do not send only a headline metric and discard Pensive's probe log or gate report.
- Do not let a generic agent edit the main Pensive checkout with unrestricted permissions.
- Do not enable multi-GPU parallelism before measuring reproducibility and memory isolation.
- Do not make SQLite or DuckDB the source of truth for Pensive experiment identity.
- Do not launch long training from a queue until the protected baseline and the cheap kill test are explicit.

## Recommendation and stop condition

Adopt the workflow concepts and, if desired, a small reviewed subset of the tracking protocol. Do not install `autoresearcherUI` as a runtime dependency of Pensive. The first useful implementation is a read-only adapter around `train.run_alternating`, `scripts/run_gate.py`, and the existing run artifacts.

Stop the integration if it starts duplicating gate logic, weakening run identity, changing checkpoint semantics, or making dashboard status more authoritative than the reproducible artifact on disk. The integration is successful when an interrupted and resumed Pensive run has the same identity and gate result, while the dashboard adds visibility without changing the result.

## Limitations

This review inspected the upstream source, README, release page, and tests. It did not install or run `autoresearcherUI`, connect it to a GPU, or test a Windows integration. The proposed adapter is therefore an architectural recommendation, not an operational compatibility result.

## Sources

- [autoresearcherUI README](https://github.com/Fchaubard/autoresearcherUI/blob/main/README.md)
- [autoresearcherUI orchestrator](https://github.com/Fchaubard/autoresearcherUI/blob/main/backend/app/orchestrator.py)
- [autoresearcherUI agent abstraction](https://github.com/Fchaubard/autoresearcherUI/blob/main/backend/app/agent.py)
- [autoresearcherUI experiment SDK](https://github.com/Fchaubard/autoresearcherUI/blob/main/arui/__init__.py)
- [autoresearcherUI scoping gate](https://github.com/Fchaubard/autoresearcherUI/blob/main/backend/app/scoping.py)
- [autoresearcherUI code-bless gate](https://github.com/Fchaubard/autoresearcherUI/blob/main/backend/app/council.py)
- [autoresearcherUI run API](https://github.com/Fchaubard/autoresearcherUI/blob/main/backend/app/api.py)
- [autoresearcherUI end-to-end test](https://github.com/Fchaubard/autoresearcherUI/blob/main/tests/e2e_test.py)
- [Pensive implementation plan](PLAN.md)
- [Pensive Stage -1 contract](STAGE-MINUS-1-CONTRACT.md)
- [Pensive evaluation runner](../eval/run/runner.py)
- [Pensive run-record definitions](../eval/run/__init__.py)
