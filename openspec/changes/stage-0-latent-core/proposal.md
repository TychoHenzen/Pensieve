## Refinement pass: Stage 0 - Latent core

This change is a planning pass. It produces a concrete implementation plan
for Stage 0, not the implementation itself. The deliverable is a
`docs/STAGE-0.md` comparable to `docs/STAGE-MINUS-1.md`, plus a locked
pass condition and a spec-delta list against the openspec tree.

### Entry condition

Stage -1 gate passed. The reproduction gate ran five seeds on
class-incremental Split-MNIST and matched van de Ven et al. 2019 within
tolerance (`gate_results/gate_report.json`). The subject protocol, stream
schema, and metric definitions are frozen at v1
(`docs/STAGE-MINUS-1-CONTRACT.md`).

### What Stage 0 must produce

From `PLAN.md`:

1. Wrap a small open model. Feed its last hidden state back as the next
   input instead of decoding a token, COCONUT style.
2. Move from a single vector to a set of 8-64 concept slots. Measure the
   multi-slot version against the single-vector version directly.
3. Build edge codecs: a sentence-embedding encoder into workspace slots,
   and a decoder back out. Freeze them at first.

Gate: latent reasoning matches token chain-of-thought accuracy on
grade-school math word problems.

### Contract obligations inherited from Stage -1

The Stage 0 subject must implement all six methods of the v1 subject
protocol: `observe`, `answer`, `idle`, `snapshot`, `restore`, `cost`.

Two constraints are load-bearing and non-obvious:

- **`restore()` must be exact.** The workspace state, including all
  concept slots, must round-trip through `snapshot()`/`restore()` with
  zero information loss. Probe isolation depends on this. Any workspace
  representation that cannot be captured and restored exactly (floating
  point state included) fails the isolation integrity test before its
  numbers count.
- **Parameter matching.** The baseline comparison requires the Stage 0
  subject to have a comparable parameter count to the baselines it is
  measured against. `eval/baselines/param_match.py` enforces this with a
  configurable tolerance.

### Open decisions the refinement must close

1. **Which base model to wrap.** PLAN.md says "a small open model."
   Candidates include GPT-2 (124M), Pythia-160M, or a smaller Pythia
   variant. The choice constrains the workspace dimension, the codec
   design, and the compute budget per seed. The refinement must name the
   model and justify the pick.

2. **COCONUT integration path.** The COCONUT paper feeds the last hidden
   state back as the next input embedding. The Stage 0 subject wraps this
   inside the v1 protocol: `observe()` encodes input into workspace slots,
   then runs N latent reasoning steps feeding hidden states back without
   decoding. `answer()` decodes from the workspace state. The refinement
   must specify how many latent steps run per `observe`/`answer`, whether
   that count is fixed or adaptive, and how the latent loop maps onto the
   `cost()` counters.

3. **No generator hosts the gate.** The gate wants "grade-school math word
   problems." The three v1 generators are:
   - `assoc`: fact recall over a closed vocabulary.
   - `split-classify`: classification with task switches (hosts the
     reproduction gate).
   - `difficulty-mix`: arithmetic chains answered mod 1000.

   `difficulty-mix` tests arithmetic, not word problems. Grade-school math
   word problems require reading a natural-language problem statement and
   producing a numerical answer. The refinement must choose one of:
   - (a) Add a fourth generator that wraps a real dataset (GSM8K or a
     subset).
   - (b) Redefine the gate against `difficulty-mix`, accepting that
     arithmetic chains are a weaker test of latent reasoning than word
     problems.
   - (c) Use an external evaluation harness for the gate and bind it into
     the run infrastructure.

   The choice must be locked before the first gate run.

4. **Concept slot count and ablation.** PLAN.md says 8-64. The refinement
   must specify the default count, the ablation sweep (which values,
   how many seeds), and the comparison against a single-vector version.

5. **Encoder architecture.** "Sentence-embedding encoder into workspace
   slots" leaves open which embedding model, how a variable-length input
   maps to a fixed number of slots, and whether the encoder is pretrained
   and frozen or trained from scratch. The refinement must name the
   encoder, the mapping strategy, and the training policy.

6. **Decoder architecture.** The decoder turns workspace slots back into
   text. The refinement must specify whether it is a separate model, a
   projection head, or the base model's own decoder run once, and how
   `answer()` invokes it.

7. **Training procedure.** Stage -1 baselines train at task boundaries.
   Stage 0 is the first subject that is not a baseline. The refinement
   must specify what trains, on what data, for how long, and whether
   training happens inside the stream or outside it.

### Cross-cutting items this stage owns

From PLAN.md's cross-cutting section, Stage 0 is responsible for:

- **Audit channel.** Build the narration decoder that turns workspace
  state into text for inspection. "Build it in Stage 0 while the mapping
  is still simple." The harness already reserves an optional `narration`
  field on each event in the run record. The refinement must specify the
  decoder architecture and how narration is triggered.

- **Latent collapse guards.** "Use stop-gradients, an EMA target encoder,
  and variance and covariance regularization from the first latent
  objective onward." The refinement must specify which guards apply to
  which loss terms, and what the collapse detection metric reports (the
  instrumentation schema in `eval/instrumentation.py` already defines the
  fields).

### Required outputs of this refinement pass

1. `docs/STAGE-0.md` - phased implementation plan with the same level of
   detail as `docs/STAGE-MINUS-1.md`. Every phase ends in something
   runnable.
2. A locked pass condition document under this change directory, written
   before the first gate run. It must name the dataset, the tolerance, the
   seed count, and the comparison target.
3. A spec-delta list: which openspec specs this stage adds or modifies.
4. A contract document for Stage 1, analogous to
   `docs/STAGE-MINUS-1-CONTRACT.md`.

### Retreat option

PLAN.md: "If it loses badly, the premise of the whole design is weak.
Stop and reconsider rather than continuing."

Stage 0 has no retreat. The gate is a go/no-go on the entire project.

### Estimated effort

The refinement pass itself (reading papers, choosing models, writing the
plan): 1-2 sessions. The implementation that follows: weeks of GPU time
across multiple sessions, because it includes pretraining.

### Impact

- New packages: `workspace/`, `codecs/`.
- New or extended: `train/` (first training scripts).
- New generator or gate evaluation path (depending on decision 3).
- New dependency: a pretrained sentence-embedding model (e.g. from
  Hugging Face `sentence-transformers`).
- `pyproject.toml` gains dependencies for the base model, the encoder,
  and any new dataset.
- No changes to the existing `eval/` harness code. Stage 0 implements
  the v1 protocol; it does not extend it.
