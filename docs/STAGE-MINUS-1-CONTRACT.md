# Stage -1 Contract (v1)

What Stage 0 must implement, and what it will be measured by. Contracts only.
Read this instead of the harness internals.

## Versions

| Interface | Constant | Value |
|---|---|---|
| Subject protocol | `SUBJECT_PROTOCOL_VERSION` (`eval/subject/__init__.py`) | `"1"` |
| Stream schema | `STREAM_SCHEMA_VERSION` (`eval/stream/__init__.py`) | `"1"` |
| Metrics | `METRICS_VERSION` (`eval/metrics/__init__.py`) | `"1"` |

## 1. Subject interface v1

A Stage 0 subject must implement all six methods. No partial implementations;
the harness calls every one of them during a run.

- `observe(event: Event) -> object | None`. The subject may learn from the
  event. Returns an emission or `None`.
- `answer(probe: Probe) -> str`. Called only inside probe isolation (below).
  Must not mutate persistent state in a way that survives `restore`.
- `idle(budget: int) -> None`. Called for `Idle` events. Stage -1 subjects
  may ignore it; later stages consolidate during it.
- `snapshot() -> object`. Captures full subject state.
- `restore(state: object) -> None`. Returns the subject to exactly the state
  a matching `snapshot()` captured. This is the isolation contract: restore
  must be exact, not approximate.
- `cost() -> CostCounters`. Cumulative compute counters, monotonically
  non-decreasing across the run. `CostCounters` fields:
  - `steps: int` (default `0`)
  - `flops: int` (default `0`)
  - `wall_seconds: float` (default `0.0`)

### Isolation guarantee

Every probe answer is measured through snapshot-answer-restore:
`snapshot()`, then `answer(probe)`, then `restore(state)` with the
snapshotted state. The subject must come out of `restore` indistinguishable
from a subject that never saw the probe. This is enforced by a harness-side
test, not a courtesy the subject extends: the subject runs forward, gets
snapshotted, is fed noise, and is restored; it must then reproduce the
outputs of a run that never saw the noise. A subject that learns from probes
despite this cycle fails the isolation test and its numbers are rejected.

A cheaper read-only-flag isolation mode exists, but only after the subject
passes a side-by-side check against snapshot/restore on a short stream.
Snapshot/restore is the reference; the flag is never assumed correct on its
own.

## 2. Stream schema v1

A stream is a deterministic, seeded sequence of `StreamItem`s. Rendering
never leaks truth, probe id, task id, teaching position, or the hostile flag
into the text a subject reads.

### Event kinds (`eval/stream/events.py`)

- `Observe(payload)`. Input the subject may learn from.
- `Probe(probe_id, task_id, query, teaching_position, token_distance)`. A
  scored question, isolated from learning.
- `Idle(budget)`. No input; reaches the subject only through `idle()`,
  never rendered as text.
- `Boundary(kind, hidden_from_subject)`. Marks a session end, task switch,
  or distribution shift. `kind` is one of `SESSION_END`, `TASK_SWITCH`,
  `DISTRIBUTION_SHIFT`. May be hidden from the subject's view.

All four share `position: int`, `narration: str | None`, `hostile: bool`.

### StreamItem (`eval/stream/truth.py`)

`StreamItem(event, truth)`. Only a `Probe` may carry a `ProbeTruth`; every
other event's `truth` must be `None`. `subject_view()` strips truth and
drops any `Boundary` marked `hidden_from_subject`. `harness_view()` yields
everything, unfiltered, for scoring.

### Generators shipped in v1

`assoc`, `split-classify`, `difficulty-mix`. Each reports a chance rate for
its probe class, computable on paper (1/vocab size, 1/classes-per-task, and
1/1000 respectively).

### Guarantees

- **Seeding.** A generator takes a config and a seed and yields the same
  sequence every time. Two runs with the same config and seed produce the
  same stream and the same probe log.
- **Hashing.** A stream is addressed by a content hash covering the config,
  the seed, the generator version, the render version, and the corpus id.
  A run record stores this hash so a later run can prove it saw the same
  stream and wording.

## 3. Metrics v1

Every metric is computed from the probe log, never from inside the subject.
The probe log is one `ProbeLogEntry` per probe:

`ProbeLogEntry(position, probe_id, task_id, teaching_position, correct,
cost_counters)`

### Metric set

- **Task accuracy** (`eval/metrics/accuracy.py`). Per task, per probe
  class, and pooled.
- **Retention matrix** (`eval/metrics/retention.py`). `R[i][j]`: accuracy
  on task `i` probes measured in the phase after task `j` finished, for
  `j >= i`; `None` where `j < i`.
- **Backward transfer** (`eval/metrics/transfer.py`). Per task,
  `R[i][last] - R[i][i]`, skipping tasks where either value is unavailable.
- **Forward transfer** (`eval/metrics/transfer.py`). Per task (excluding
  the first), accuracy on that task's probes seen before the task was
  taught, minus that task's chance rate.
- **Compute per input** (`eval/metrics/compute.py`). Three separate series
  derived from consecutive `cost_counters` deltas: `steps`, `flops`,
  `wall_seconds`.
- **Time to first use** (`eval/metrics/first_use.py`). Per taught fact
  (identified by `probe_id` + `teaching_position`), the stream-position
  distance to the first correct answer. Never-correct facts get a
  configured `cutoff` value. Reported as median plus `cutoff_share`
  (mean is not reported; it hides cutoff facts).

## 4. Version policy

Any breaking change to the subject interface, the stream schema, or the
metric definitions bumps its version constant. A version bump voids
comparisons against runs recorded under the old version, on purpose.
Non-breaking additions (new optional fields, new generators) do not require
a bump.
