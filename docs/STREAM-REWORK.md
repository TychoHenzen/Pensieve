# Stream rework: make the event stream feedable, then add two generators

Scope: phase 2 of `STAGE-MINUS-1.md` reopened, plus the two generators that phase
left unbuilt. No subject code, no metrics code, no model work.

This document is the plan source for `/dod-guard:step-by-step`. The matching
session state lives in `.step-session/steps.json`.

## Why reopen phase 2

The current stream replays the same way twice and hashes stably. That part is done
and stays. Three holes make it unusable by any subject that reads text.

1. A `Probe` carries no question. `eval/stream/events.py` gives it `probe_id`,
   `task_id`, and `teaching_position`. The key it asks about stays in the
   `Observe` at `teaching_position`. A subject reading the probe cannot answer it.
2. Nothing turns an event into text. `serialize.py` exists for hashing and it
   includes the truth channel, so it is not a subject view. Each subject would
   invent its own wording, and two baselines would stop running the same benchmark.
3. `recall_distances` counts events. `STAGE-MINUS-1.md` promises recall distances
   measured in tokens. Nothing converts between the two.

Three more problems sit in the `assoc` payload rather than the schema.

4. Keys and values are random 32-bit numbers rendered as `key-3585642846`. A wrong
   answer then mixes two failures: the memory lost the fact, or the decoder
   dropped one digit of a long copy. Retention is the thing being measured.
5. Filler events are synthetic pairs behind a `filler-` prefix. Two things are
   wrong with that. The prefix hands a learning subject a rule for skipping
   filler. The synthetic pairs also make the span between a teaching and its
   probe unlike anything a deployed memory would ever read.
6. Chance accuracy against a free-form string is effectively zero. Forward
   transfer scores against a chance baseline, so it breaks. The Chance oracle
   must land on the chance rate within sampling error, so it breaks too.

Fixes 4 and 6 are one change: draw keys and values from one closed vocabulary of
short words. The answer becomes a single choice, chance becomes
`1 / len(vocabulary)` and is known on paper, and a wrong answer means forgetting.

Fix 5 is a separate change: filler becomes real prose from a pinned slice of The
Pile. A taught fact then has to survive continuous text rather than a run of
synthetic pairs.

## Decisions taken, so no step has to guess

**Token counting stays dependency-free.** Stage -1 must not pull in a tokenizer to
count distances. `render.py` exposes a counter callable. The default splits on a
regular expression and is honest about being an approximation. A run record stores
the counter name it used. Stage 0 passes the real tokenizer in, and the field says
which one produced the number.

**Distance stays in events, and tokens get measured.** Placing probes by token
offset would make the scheduler depend on the rendering. The rendering would then
sit inside the stream identity twice. Instead each `Probe`
gains a measured `token_distance`, filled at generation time from the rendered
text between teaching and probe. The config keeps event distances. The report can
then state a real token number instead of a promise.

**Filler prose comes from a pinned slice of The Pile.** The source is
`monology/pile-uncopyrighted`. Its validation file already has Books3 and the
other copyrighted subsets stripped out. A fetch script downloads that file. It
decompresses the file as a stream, keeps documents until it holds enough text,
and writes a plain snapshot pinned by sha256. The repository stores the script and
the hash, never the data. Generation then runs offline and needs no package,
because the snapshot is plain text. `zstandard` is an extra for the script alone,
since Python 3.13 ships no zstd decoder.

**A taught fact stays its own event.** It sits between prose spans rather than
hidden inside a sentence. A subject can therefore spot a fact and skip the prose.
That is accepted. The fact is a needle in a stack of hay. What gets measured is
whether the fact survives the distance, not whether the subject can find it.
Burying the fact inside a sentence would also make the probe query ambiguous.

**The rendering version and the corpus id enter the stream hash.** Take two runs
sharing a config, a seed, and a generator version. If their wording differs, they
are two different benchmarks. Two different prose snapshots differ that way too.

**`split-classify` carries a dataset hole, not pixels.** The reproduction gate
needs Split-MNIST (the Modified National Institute of Standards and Technology set
of written digits) later. Split cuts that set into tasks of two digits each. The
payload holds `features`, `label`, and an optional `source` naming a dataset
item. Synthetic features fill it now. Binding the real digit data later is then a
config change rather than a schema change.

**Difficulty must cost real work.** An item labeled hard has to be harder to
answer, or "compute per input tracks difficulty" cannot be measured. So
`difficulty-mix` asks for arithmetic chains, and difficulty is the chain length.
The label lives on the truth channel. A subject that could read it would fake the
correlation.

## Step list

Thirteen steps. Each ends with a passing test file. Dependencies are recorded in
`steps.json` and they matter more than array order.

1. **Probe carries its question.** Add a required `query` field to `Probe`.
2. **Closed vocabulary.** New `eval/stream/vocab.py`. One deterministic word list,
   drawn without replacement inside a stream, with a stated chance rate.
2b. **Corpus.** A fetch script plus `eval/stream/corpus.py`. Prose spans by seed
   from a pinned snapshot, a corpus id, and a small committed fixture so the
   tests never need the download.
3. **Rework the assoc payload.** Vocabulary keys and values, prose filler drawn
   from the corpus, `query` filled from the key.
4. **Rendering.** New `eval/stream/render.py`. One event to one text block, a
   `RENDER_VERSION`, and `rendered_subject_view`. No truth and no probe id ever
   reach the text.
5. **Rendering and corpus enter the hash.** `stream_hash` takes the render version
   and the corpus id too.
6. **Token counting.** A counter callable, a default regular-expression counter,
   and a recorded counter name.
7. **Measured token distance.** `Probe` gains `token_distance`, filled by the
   generator from rendered text.
8. **Generator registry.** `config.generator` resolves to a generator by name, so
   a config alone can build a stream.
9. **`split-classify`.** Tasks in sequence, probes on every task seen so far,
   `Boundary(TASK_SWITCH)` between tasks.
10. **`difficulty-mix`.** Arithmetic chains at labeled depths, interleaved at
    random, difficulty on the truth channel.
11. **Chance rate on paper.** Each generator reports the chance accuracy of its
    probe classes, and a test checks a random answerer lands there.
12. **Docs.** Update `STAGE-MINUS-1.md` where this plan changed what it promised.

## Corrections found by reading the rendered stream

`scripts/show_stream.py` printed the text a subject would actually read. Three
things in the list above had landed in letter but not in sense. All three are
now fixed, at `RENDER_VERSION` 2, `assoc` version 6 and `split-classify`
version 2.

1. **A payload printed as a Python dict repr.** Rendering had one branch per
   event class. So an `Observe` printed `{'key': 'from', 'value': 'hear'}`,
   quotes included, and a feature vector printed 17 digits of float noise per
   value. Rendering now dispatches on payload shape. An unknown shape raises
   instead of falling back to a repr. A `split-classify` probe query
   goes through the same feature formatter as the examples it is compared
   against.
2. **Filler was fragments, not continuous text.** Each span drew an independent
   random start, so consecutive filler jumped topic every 15 words. Fix 5 asked
   for a fact that has to survive continuous text, and a run of unrelated
   fragments is the thing it was replacing. A stream now takes one forward walk
   through the corpus. Wrapping the prose in `[observe] {'text': ...}` cost a
   further 13% on top of every measured token distance, and that wrapper is
   gone: prose renders bare.
3. **`Idle` was rendered as text.** An `Idle` is defined as a block of time with
   no input, and the subject protocol delivers it through `idle(budget)`.
   Rendering it as `[idle] budget=4` made it input. `render.carries_text`
   returns False for it, `render_event` refuses it, and `rendered_subject_view`
   drops it. In `difficulty-mix` that removed text from half of all events.

## What this plan does not do

It writes no subject, no probe log, and no metric. Those are phases 3 and 4 of
`STAGE-MINUS-1.md` and they start after this lands. It also does not bind real
MNIST data. Step 9 leaves the hole and stops there.
