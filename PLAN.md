# Pensive - Implementation Plan

Super-high-level plan for the system described in `Reference.md`: a **Latent Cortex
with a Fast-Weight Hippocampus and an Idle Consolidator**. One model, one user,
always on. It reasons in latent vectors. It learns during use. It consolidates
when idle.

This plan is staged. Every stage ends in a measured gate. A stage that misses its
gate either falls back to the named retreat option or stops the project. Nothing
gets built on top of an unmeasured stage.

## Target system, in one paragraph

Text and other input hit an encoder that writes a small set of concept vectors into
a persistent workspace. A slow Planner sets an abstract goal and passes a few
vectors through a narrow channel to a fast Worker. The Worker loops over the
workspace until a learned halting rule says the thought is done. Each loop reads and
writes a fast-weight memory whose weights change during use. Most loops emit nothing.
An output gate decides when to run the decoder and emit text. When input stops, an
idle loop replays recent memory episodes and trains the slow core on them, so
today's hot learning becomes tomorrow's stable skill.

## Scale and platform assumptions

- Single workstation with one consumer GPU, scaling to a rented multi-GPU box for
  the pretraining stages only.
- Model size in the 100M to 1B range. This is a mechanism study, not a scale race.
  Every claim is a comparison against a matched baseline of the same size.
- Python, PyTorch, Hugging Face for the frozen edge codecs.
- Nothing in the design assumes cloud serving. The idle loop needs a machine that
  is genuinely idle, which is the point of building for one user on one device.

## Repository layout

```
pensive/
  workspace/      concept slots, persistent state container
  codecs/         text encoder and decoder at the edges
  core/           planner, worker, bottleneck channel, halting
  memory/         fast-weight hippocampus, write and read rules
  consolidate/    generative replay, EWC, the idle scheduler
  runtime/        the always-on loop, clocks, output gate
  eval/           streaming benchmarks, metrics, baselines
  train/          stage scripts and configs
```

## Stage -1: Harness before model

Nothing here is a research result. It exists so every later claim is measurable.

- Streaming benchmark harness. A benchmark is a long, non-stationary stream with
  labeled probe points, not a static test set. Each probe asks one thing. Do you
  recall a fact from 100k tokens back? Did you keep the old skill after learning
  the new one? Did you spend more compute on the harder item?
- Metrics: task accuracy, retention of old skills, forward and backward transfer,
  compute per input, time to first use of a newly taught fact.
- Baseline runner. Every stage compares against a matched-parameter conventional
  model on the same stream. A stage with no baseline number is not done.
- Run logging and checkpointing from day one, because later stages run for days.

Gate: the harness reproduces a known continual-learning result on a toy model.

## Stage 0: Latent core

Reasoning happens vector to vector. Tokens exist only at the edges.

- Wrap a small open model. Feed its last hidden state back as the next input
  instead of decoding a token, COCONUT style.
- Move the workspace from one vector to a set of 8 to 64 concept slots. This is the
  first real departure from the papers, so measure it against the single-vector
  version directly.
- Build the edge codecs: a sentence-embedding encoder into workspace slots, and a
  decoder back out. Freeze them at first.

Gate: latent reasoning matches token chain-of-thought accuracy on grade-school math
word problems. If it loses badly, the premise of the whole design is weak. Stop and
reconsider rather than continuing.

## Stage 1: Persistent state and two clocks

- Replace the backbone with a fixed-size recurrent state, Mamba or xLSTM style, so
  the state carries across the whole stream and never resets.
- Add adaptive halting on the inner loop, PonderNet style, so a cycle can run
  without emitting anything.
- Add the output gate: a separate learned decision about when to speak.

Gate: performance holds across an artificially long stream broken into fake
sessions. Compute per input tracks item difficulty.

## Stage 2: Fast-weight hippocampus

This is the pivotal stage. Everything before it is assembly of known parts.

- Add a memory module whose weights update by gradient descent during use, Titans
  or test-time-training style, writing surprising input more strongly.
- Add the safety machinery in the same stage, not later. That means weight decay,
  gradient clipping, and a trust region on the fast weights. Add a hard firewall
  too, so the fast store can only reach the slow core through vetted replay.

Gate: teach the running system a new fact mid-stream. It must use that fact right
away, with no training run. Frozen-core skill must show no measurable loss over a
long stream afterward.

Retreat option if online updates will not stay stable: replace live weight updates
with an external read and write memory store. The rest of the design survives.

## Stage 3: Idle consolidation

- Build the replay generator. It dreams up past-like episodes from the hippocampus
  rather than storing raw input.
- Train the slow core on the replayed stream during idle periods, with
  elastic-weight-consolidation protection on weights that matter to old skills.
- Add the idle scheduler. It detects no input and enters inward mode. It yields at
  once when input arrives. It partly clears the hippocampus after a good transfer.

Gate: after learning several skills hot, consolidation moves them into the slow core
and old skills survive. Must beat a replay-free baseline on retention without
storing raw data.

Retreat option: keep a compressed episodic buffer instead of pure generative replay.

## Stage 4: Bottlenecked dual module

Last, and openly on probation. The independent analysis of the Hierarchical
Reasoning Model found its brain-shaped split was not what produced its results.

- Split the core into a slow Planner and a fast Worker, joined only by a narrow
  channel of a few vectors per planner tick.
- Regularize the channel with an information-bottleneck objective so each side has
  to send abstractions rather than raw state.
- Solve credit assignment across the two clocks, starting from deep supervision or
  a learned halting value function.

Gate: the split beats a single stack of equal parameter count on the same streams.
If it does not, delete it, keep the single stack, and record the two-hemisphere idea
as inspiration that was tried and dropped.

## Stage 5: The always-on runtime

- Wire the pieces into one long-lived process: perceive, encode, plan, loop, write
  memory, occasionally speak, and consolidate whenever idle.
- Persist all state to disk so the process can restart with its memory intact.
- Ship a small chat surface. A system that learns from one user needs a user.

Gate: the system runs for a week of real use and remembers across restarts. Teach
it something on day one. It must still know that thing on day seven.

## Cross-cutting work

- **Audit channel.** Latent thought cannot be read. Keep an optional narration
  decoder that turns workspace state into text for inspection. Build it in Stage 0
  while the mapping is still simple.
- **Latent collapse guards.** Any latent prediction objective can collapse to a
  constant. Use stop-gradients, an exponential-moving-average target encoder, and
  variance and covariance regularization from the first latent objective onward.
- **Poisoning.** A model that learns from its input can be taught lies. The
  firewall, the trust region, and the vetted-replay path are the defense. Attack
  them in Stage 2, not after deployment.
- **Instrumentation.** Log halting steps, memory write magnitude, channel bandwidth,
  and consolidation gain on every run. These are the signals that say which
  mechanism is actually paying.

## Kill criteria

- Stage 0 latent reasoning is far behind token reasoning, and the gap does not close
  with the multi-slot workspace.
- Stage 2 online updates cannot be stabilized, and the external-memory retreat also
  fails to give mid-stream learning.
- Stage 3 consolidation cannot transfer without forgetting under any retreat option.

Any of these means the integration claim does not hold at this scale. The honest
move is then to write up what failed rather than keep bolting on stages.

## Open problems this plan does not solve

Training the whole thing end to end is not possible. So the plan trains it in
pieces and accepts that the seams are rough. Credit assignment across two clocks is
unsolved. Nobody knows how to score a continual, always-on system yet. That is why
Stage -1 builds a harness instead of picking an existing test set. The integration
itself is the research contribution, and it is genuinely open.
