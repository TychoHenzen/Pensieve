# Stage -1 Implementation Plan: Harness Before Model

Scope: everything in `PLAN.md` Stage -1, start to finish. No model research happens
here. The deliverable is a measurement instrument plus the proof that the
instrument works.

The gate for the whole stage: the harness reproduces a known continual-learning
result on a toy model. Until that passes, no Stage 0 code gets written.

## What Stage -1 must produce

1. A streaming benchmark harness. A benchmark is a long non-stationary stream with
   labeled probe points, not a static test set.
2. A metric set: task accuracy, retention, forward and backward transfer, compute
   per input, time to first use of a newly taught fact.
3. A baseline runner that puts a matched-parameter conventional model on the same
   stream with the same seeds.
4. Run logging and checkpointing good enough for jobs that run for days.
5. A reproduction of a published continual-learning result, produced by the harness
   itself and matching the paper within a stated tolerance.

Everything else in this document exists to make those five things trustworthy.

## The central design problem

The system under test learns from everything it sees. That breaks the usual split
between training data and test data. Two consequences drive the whole design.

First, measuring changes the subject. A probe is input. If the subject writes memory
while answering a probe, the act of measuring teaches it. So probe handling needs
explicit isolation, described below, and that isolation must be part of the harness
rather than a courtesy the subject extends.

Second, order is part of the benchmark. Two runs over the same items in a different
order are two different benchmarks. So a stream is a seeded, reproducible object
with a version, not a folder of files.

## Component 1: the stream

A stream is a deterministic, seeded sequence of typed events.

Event kinds:

- `Observe`. Input the subject may learn from.
- `Probe`. A question with a known answer, scored, and isolated from learning. It
  carries a self-contained `query` naming what it asks, and, when a token counter
  is named, a measured `token_distance` back to what taught the answer.
- `Idle`. A block of wall time or step budget with no input. It renders to no
  text at all. It reaches the subject through `idle(budget)` instead. Rendering
  it would make it input, against the "no input" in its own definition.
  Stage -1 subjects ignore it. Stage 3 subjects consolidate during it. It is in
  the schema from day one so no stream needs rewriting later.
- `Boundary`. A marker for a fake session end, a task switch, or a distribution
  shift. Visible to the harness, optionally hidden from the subject.

A stream generator takes a config and a seed and yields events. It also yields the
ground truth needed to score each probe, on a side channel the subject never sees.
A separate rendering step turns one event into the text block a subject reads.
It never lets a truth answer, a probe id, a task id, a teaching position, or the
hostile flag reach that text. So two subjects reading the same stream see
identical wording.

An `Observe` renders by the shape of its payload, not by its event class. Payload
shapes vary per generator while the class does not. One branch per class would
therefore print whatever a payload happened to be. A taught fact reads
as `[fact] key = value`, a labeled example reads as `[example] features=(...)
label=...`, and a prose span reads as bare text with no marker around it. A
payload matching no known shape raises rather than falling back to a repr. The
same feature formatter writes a `split-classify` probe query, so a probe and the
examples it is compared against never differ in wording or precision.

`scripts/show_stream.py` prints this text for every generator, beside the truth
each probe is scored against. Read it before changing a payload or a rendering.

Stage -1 ships these generators:

- `assoc`. Fact learning over a closed vocabulary. Teach `key -> value` pairs,
  drawn from a fixed word list, at known positions, with prose filler between
  them and a probe query built from the key. Filler comes off one forward walk
  through the corpus per stream. Consecutive spans therefore continue one
  document instead of jumping between unrelated parts of the snapshot. Probes are
  placed at controlled event distances, because the scheduler places probes by
  event. Each probe
  also carries a token distance, measured from the rendered text between
  teaching and probe once rendering is done. This is the direct test of "do
  you recall a fact from far back". It is the one generator where the correct
  answer is known by construction rather than by a dataset.
- `split-classify`. A classification stream split into tasks presented in
  sequence, with probes on every task seen so far and a `Boundary(TASK_SWITCH)`
  between tasks. The `Observe` payload holds `features`, a `label`, and an
  optional `source` naming a dataset item. `source` stays empty for now. Real
  digit data can bind through it later, as a config change. That data is
  Split-MNIST (the Modified National Institute of Standards and Technology set
  of written digits, split into tasks of two digits each). This is the
  generator that hosts the reproduction gate.
- `difficulty-mix`. Arithmetic chains whose length is the difficulty label,
  kept on the truth channel only so a subject cannot read it off the query.
  Answers run mod 1000, so the answer space stays bounded regardless of chain
  length.

Every generator reports a chance rate for its probe class, computable on
paper. That rate is 1 in the vocabulary size for `assoc`, 1 over
`classes_per_task` for `split-classify`, and 1 in 1000 for `difficulty-mix`.

A stream is addressed by a content hash. That hash covers the config, the seed,
the generator version, the render version, and the corpus id. A run record
stores that hash, so a later run can prove it saw the same stream, the same
wording, and the same filler source.

## Component 2: the subject protocol

One interface, implemented by baselines now and by every later stage.

- `observe(event) -> Optional[Emission]`. The subject may emit or stay silent.
- `answer(probe) -> Answer`. Called only inside probe isolation.
- `idle(budget) -> None`.
- `snapshot() -> State` and `restore(state)`. Used for probe isolation and for
  checkpointing.
- `cost() -> CostCounters`. Cumulative compute counters, described below.

Probe isolation works like this. The harness snapshots subject state, calls `answer`,
then restores. Restore is the contract. A subject that cannot restore exactly cannot
be measured. So `snapshot` and `restore` get a test that runs a subject forward,
snapshots, feeds noise, and restores. The subject must then produce the same outputs
as a run that never saw the noise. That test belongs to the harness, not to the
subject. Every subject must pass it before its numbers count.

There is a cheaper isolation mode: a read-only flag the subject honors. A subject may
use it only after passing a side-by-side check against snapshot and restore on a short
stream. Snapshot and restore stays the reference.

## Component 3: metrics

Every metric comes from the probe log, never from inside the subject. The probe log
is a table with one row per probe. A row holds five fields plus the cost counters at
that moment. The five are the stream position, the probe id, the task id, the teaching
position of the item, and whether the answer was right.

- **Task accuracy.** Per task, per probe class, and pooled.
- **Retention.** Accuracy on task `i` probes measured after task `j` finished, for
  all `j > i`. Stored as the standard task matrix `R[i][j]`.
- **Backward transfer.** Change in old-task accuracy caused by later learning,
  derived from `R`.
- **Forward transfer.** Accuracy on a task before it was taught, against a chance
  baseline for that task.
- **Compute per input.** Three separate numbers, because they drift apart: inner loop
  steps, estimated floating-point operations, and wall time. Each one is
  matched against the difficulty label from `difficulty-mix`.
- **Time to first use.** The gap in stream positions between teaching a fact and the
  first probe the subject answers right. A fact never learned gets a cut-off value
  instead. Report the median and the share of facts that hit the cut-off. A plain
  mean would hide them.

Every metric ships with an oracle test, described in the self-validation section.

## Component 4: baselines

A stage with no baseline number is not done, so the baseline runner is a first-class
component, not a script.

Baselines for Stage -1:

- **Naive sequential.** Same architecture, plain stochastic gradient descent (SGD) as
  the stream arrives. The forgetting floor.
- **Joint training.** Trained on all tasks at once, offline. The retention ceiling.
- **Frozen.** No learning at all. The no-adaptation floor.
- **EWC.** A published continual-learning method.
- **Generative replay.** The second published method, and the one whose published
  numbers the gate reproduces.

The runner enforces parameter matching. It counts the parameters of the subject and
of the baseline. It then refuses to report a comparison outside a stated tolerance.
Matching is on the count alone. The report also records depth, width, and optimizer.
Equal counts with very different shapes make a weak comparison, and the report should
show that rather than hide it.

## Component 5: run infrastructure

Later stages run for days on one workstation, so this has to be boring and solid.

- **Run record.** One directory per run. Holds the resolved config, the code
  revision, the stream hash, the environment, the probe log, the metric summary, and
  the checkpoints.
- **Config.** One typed config object, resolved from a file plus overrides, hashed
  into the run id. No values read from the environment at runtime except paths.
- **Repeatable runs.** Seeded generators, seeded setup, seeded data order, recorded
  library versions. An exact repeat on GPU is not always available. So the harness
  records whether the exact-repeat kernels were on, and the report states which mode
  produced the number.
- **Checkpoint and resume.** Written on a schedule, by stream position and by wall
  clock. A resumed run must continue the same stream at the same position with the
  same seed state. One test checks that a resumed run and an unbroken run produce the
  same probe log.
- **Crash and shutdown.** Assume the box reboots. Checkpoints are written in one
  atomic step, and the newest complete one wins.
- **Storage budget.** Long runs produce a lot of logs. Probe logs are append-only and
  compact. Checkpoint retention is a policy in the config, defaulting to last plus
  every Nth.

## Component 6: harness self-validation

This is the part that is easy to skip and expensive to skip. The harness must be
tested against subjects whose correct metrics are known in advance.

Oracle subjects:

- **Perfect memory.** Answers every probe for anything it has seen. Retention must
  come out at the maximum, backward transfer zero, time to first use equal to the
  distance to the next probe.
- **Forgetful.** Answers only from the most recent item. Retention must fall to chance
  past that short window.
- **Chance.** Random answers. Every accuracy metric must land on the chance rate for
  that probe class, within sampling error.
- **Task wiper.** Learns the current task perfectly and destroys the one before it.
  Backward transfer must come out strongly negative, at a value you can work out on
  paper.
- **Cheater.** Learns from probes. The test that checks probe isolation must catch it,
  and the run must be rejected. This oracle is what proves isolation works.

Each oracle has an expected metric value derived on paper before the code runs. A
mismatch is a harness bug. These tests run in CI and they are the reason to believe
any later number.

## Component 7: the reproduction gate

The gate is: the harness reproduces a known continual-learning result on a toy model.

Target: class-incremental `Split-MNIST`, the written digit set cut into tasks of two
digits each. Compare naive sequential, EWC, and generative replay. Score them against
the numbers reported in van de Ven, Siegelmann and Tolias, "Brain-inspired replay for
continual learning with artificial neural networks", Nature Communications 11:4069,
2020. The shape of the result matters most. In the class-incremental setting, naive
and EWC sit near chance while replay stays high. Pull the exact percentages from the
paper before locking the tolerance. Do not take them from memory.

Why this target. It is cheap enough to run many times on one consumer GPU. The paper
spells the protocol out in full. It exercises the same machinery Stage 3 depends on.
It also has a well-known failure mode: EWC collapses in the class-incremental setting.
So a harness that reports EWC doing fine is plainly broken.

Pass condition, to be written down before the first run:

- Each method lands within a stated tolerance of the paper's reported accuracy. Fix
  the tolerance in advance. Run at least five seeds and report the spread.
- The ordering of the three methods matches the paper.
- The run is reproducible: two runs with the same config and seed give the same
  probe log.

If the numbers do not reproduce, the fault sits in one of two places. Either the
harness is wrong or the rewrite of the method is wrong. To tell them apart, run the
paper's own released code through the harness as a subject. That is the fastest split
between an instrument bug and a code bug, and it is worth the wiring.

A second, smaller reproduction on `Permuted-MNIST` is a stretch goal, not a gate. It
guards against tuning the harness to one task.

## Order of work

Each phase ends in something runnable. Nothing is built two phases ahead of its test.

1. **Skeleton.** Repository layout from `PLAN.md`. Packaging and the test runner.
   Lint, typed config, run record, logging. Ends when a null subject runs a two-event
   stream and writes a valid run directory.
2. **Stream core.** Event types, generator protocol, seeding, hashing. Adds the
   `assoc` generator. Ends with a stream that replays the same way twice.
3. **Subject protocol and isolation.** The interface, snapshot and restore, the
   isolation test, and the cheater oracle. Ends when a test proves isolation works by
   failing once isolation is removed.
4. **Metrics.** The probe log, the metric set, and every oracle subject. Ends when all
   oracle expectations match the values worked out on paper.
5. **Instrumentation.** Cost counters and the event schema for later signals. Ends
   with `difficulty-mix` showing that a deliberately variable-compute oracle produces
   the expected correlation.
6. **Persistence.** Checkpoint, resume, and the crash test. Ends when a killed and
   resumed run matches an unbroken one.
7. **Baselines.** The five baselines, the `split-classify` generator, and parameter
   matching. Ends with all five running on one stream and producing a comparison
   table.
8. **Gate.** The reproduction, the seed sweep, and the written pass condition. Ends
   with a report.
9. **Freeze.** Version the subject interface, the stream schema, and the metric
   definitions as v1. Write the short document that Stage 0 reads. A later stage may
   extend these. Any breaking change bumps the version, which voids old run
   comparisons on purpose.

Phases 2 through 5 are the risky ones, because that is where a wrong definition
becomes load-bearing. Phases 6 and 7 are ordinary engineering.

## Cross-cutting concerns

**Instrumentation schema, defined now.** `PLAN.md` requires logging halting steps,
memory write magnitude, channel bandwidth, and consolidation gain. None of those
exist yet. Stage -1 defines the log schema and the reporting path for all four, and
ships them as optional fields that current subjects leave empty. Defining them late
means re-running old experiments to fill in the columns.

**Compute accounting.** The claim "spend more compute on the harder item" needs a
compute number that wall-clock noise cannot game. Counters count steps, one by one,
and nothing is inferred. Wall time is recorded but never the primary number.

**Audit channel.** Stage 0 builds the narration decoder. The harness needs the hole
for it now. Each event gets an optional narration field in the run record. A viewer
prints a stream position with its narration beside it. Cheap now, awkward later.

**Latent collapse guards.** Stage -1 subjects do not need this, but the detector does
not change later. It measures the variance and covariance of a batch of workspace
states. The harness ships that metric now, with a test on made-up collapsed and
healthy states. Stage 0 then starts with a working detector, instead of meeting
collapse by accident.

**Poisoning and attack streams.** Stage 2 attacks the memory. The stream schema needs
an event flag that marks an item as hostile. The metric set needs an attack success
rate. Both are defined here, so the Stage 2 attack is a config rather than a rewrite.

**Exact repeats versus speed.** The kernels that give an exact repeat run slower. The
policy: gate runs and oracle tests use them, long exploratory runs need not. The
report always states which mode ran. A number from the fast mode never settles a gate.

**Statistical honesty.** Every headline number carries a seed count and a spread.
Single-seed comparisons are reported as anecdotes and cannot pass a gate. The
tolerance for a comparison is written before the run, not after seeing it.

**Cost and time budget.** Stage -1 targets a single consumer GPU. The gate
reproduction must fit in a few GPU-hours per seed, so a seed sweep stays routine. If
it does not, shrink the toy model rather than cut the seed count.

**Documentation.** One page per component, written when phase 9 freezes it. Each page
describes the contract, not the code behind it. Stage 0 should be able to write a
subject without reading harness internals.

## Kill and retreat criteria for this stage

- The reproduction fails, and the paper's own released code fails through the harness
  too. Then the harness is wrong and phases 3 to 5 get redone. That is a cost, not a
  kill.
- The reproduction fails, but the released code passes through the harness. Then the
  rewrite of the method is wrong. Fix the method and keep the harness.
- Probe isolation cannot be made exact for a class of subject. Then retreat to the
  read-only flag plus a written side-by-side check. Record the leftover leak as a
  known limit on every number that subject produces.
- Stage -1 has no kill criterion of its own. Without the harness the project cannot
  measure anything, and no later stage can be proved wrong.

## What Stage -1 explicitly does not do

Stage -1 builds no latent reasoning and no recurrent backbone. It builds no fast
weights, no consolidation, and no runtime. The toy models here exist only to exercise
the instrument. Any time spent making them good is time spent on the wrong stage.
