## Context

Stage -1 delivered a streaming benchmark harness with five baselines, three
generators, and a reproduction gate. The v1 subject protocol, stream schema,
and metric definitions are frozen (`docs/STAGE-MINUS-1-CONTRACT.md`). Stage 0
builds the first non-baseline subject: a latent reasoning system that operates
vector-to-vector, with tokens only at the edges. See `proposal.md` for
motivation and scope.

## Goals / Non-Goals

**Goals:**

- Wrap Pythia-160M so its hidden states feed back as input embeddings
  (COCONUT-style latent loop), with no token decoding between steps.
- Replace the single hidden-state vector with a configurable set of 1-64
  concept slots, default 16. Measure multi-slot against single-vector directly.
- Build frozen edge codecs: a sentence-embedding encoder into slots, a decoder
  back to text.
- Ship the narration decoder (audit channel) while the workspace-to-text
  mapping is still simple.
- Add a GSM8K generator to host the Stage 0 gate.
- Wire latent collapse guards into the first training objective.

**Non-Goals:**

- Adaptive halting (Stage 1).
- Fast-weight memory (Stage 2).
- Unfreezing the codecs (later decision after initial results).
- Extending the v1 subject protocol. Stage 0 implements it, does not change it.

## Decisions

### D1. Base model: Pythia-160M

**Choice**: EleutherAI Pythia-160M (hidden dim 768, 12 layers, 12 heads).

**Why over GPT-2 (124M)**: Pythia was trained with documented, reproducible
methodology. The full model suite (70M through 12B) gives a scaling ladder
if Stage 0 succeeds. Both have hidden dim 768, so the workspace and codec
dimensions are the same.

**Why not Pythia-70M**: Hidden dim 512 gives a smaller workspace per slot. The
mechanism study benefits from the extra capacity at 160M, and the compute cost
per seed stays within a few GPU-hours on a consumer card.

**Alternatives considered**: GPT-2 (124M) was the other candidate. Comparable
size, wider community tooling, but less reproducible training provenance. The
decision is weak - either model works. Pythia's scaling ladder tipped it.

### D2. COCONUT integration: fixed latent steps, workspace as slot set

**Choice**: After `observe()` encodes input into workspace slots, the latent
loop runs a fixed number of forward passes through Pythia-160M. Each pass
reads the current slot vectors as input embeddings (one slot per position in
the model's sequence), produces hidden states, and writes them back to the
slots through a learned projection. `answer()` runs the same loop, then feeds
the final slot states through the decoder.

**Latent step count**: Fixed at N per cycle, configurable (default 8). Not
adaptive in Stage 0; adaptive halting is Stage 1. The count is a
hyperparameter in the training and evaluation config.

**Cost mapping**: Each latent step increments `cost().steps` by 1 and
`cost().flops` by the forward-pass flop count of Pythia-160M on a sequence
of length equal to the slot count.

**Why fixed, not adaptive**: Adaptive halting (PonderNet-style) adds a learned
stopping decision, which is a research problem on its own. Stage 0 proves
that latent reasoning works at all. Stage 1 adds halting.

### D3. Encoder: sentence-transformers with cross-attention slot mapping

**Choice**: The encoder has two parts.

1. A frozen pretrained sentence encoder (`all-MiniLM-L6-v2`, output dim 384)
   maps input text to token-level embeddings.
2. A learned cross-attention module maps those embeddings to workspace slots.
   It uses N learned query vectors (one per slot, dim 768) that attend over
   the projected (384 to 768 via a linear layer) token embeddings. Output: N
   vectors of dim 768, written directly to the workspace slots.

**What is frozen**: The `all-MiniLM-L6-v2` weights. The linear projection
(384 to 768) and the cross-attention parameters train.

**Why cross-attention over simpler mappings**: A linear projection from a
single sentence embedding to (N, 768) loses token-level information. Cross-
attention preserves it and handles variable-length input naturally. The
learned queries give each slot a chance to specialize.

**Why `all-MiniLM-L6-v2`**: Small (22M params), fast, well-tested, available
via `sentence-transformers`. The encoder is frozen, so quality of the
pretrained representations matters more than size. Alternatives:
`all-mpnet-base-v2` (higher quality, 110M params, slower). Start small; swap
if encoding quality limits results.

**Dependency**: `sentence-transformers` added to `pyproject.toml`.

### D4. Decoder: Pythia-160M autoregressive generation from slot embeddings

**Choice**: Feed the workspace slot vectors as input embeddings to
Pythia-160M's forward pass (slots occupy the first N positions). The model's
existing language modeling head produces logits. Decode tokens autoregressively
from those logits.

**Why reuse the base model**: The base model already has a language modeling
head. Training a separate decoder doubles the parameter count and adds a
surface for bugs. Using the same model for the latent loop and for decoding
keeps the representation space aligned.

**Frozen policy**: The Pythia-160M weights in the decoder are the same weights
used in the latent loop. They share one set of parameters. Freezing applies to
the pretrained weights. The learned projection that maps hidden states back to
input embeddings (in the latent loop) does train.

### D5. Training procedure

**What trains**: The cross-attention slot mapping (encoder side), the hidden-
state-to-embedding projection (latent loop), and the language modeling head's
adaptation layer if one is added. The pretrained Pythia-160M weights and the
sentence encoder weights are frozen.

**On what data**: GSM8K training split. Each problem is an (input, answer)
pair. The loss is language modeling loss on the answer tokens, given the
workspace state after encoding the input and running the latent loop.

**For how long**: Configurable epochs/steps. Start with 10 epochs over the
GSM8K train split (~7.5k problems), measure loss curves, and adjust. Each
epoch on a consumer GPU (RTX 3090 or similar) should take under 30 minutes
with the frozen backbone.

**Training happens outside the stream**: The subject is pretrained, then
evaluated on the harness stream. Stage 0 does not train during the stream
(that is Stage 2's fast-weight learning). This is consistent with how the
Stage -1 baselines work: each baseline is trained, then the trained model
runs the stream.

### D6. Latent collapse guards

**Choice**: VICReg-style regularization applied to the workspace slot vectors
during training.

- **Variance term**: penalizes low variance of each slot dimension across a
  batch. Prevents all slots from collapsing to the same vector.
- **Covariance term**: penalizes high covariance between slot dimensions.
  Prevents dimensions from becoming redundant.
- **Stop-gradients**: applied on the target side of the latent loop's
  projection. The EMA target encoder tracks a moving average of the
  projection weights, and the loss computes similarity against the EMA
  target's output with stop-gradient on the target.

**Detection metric**: The `InstrumentationFields` already defines
`memory_write_magnitude`. The collapse detector computes slot variance and
slot covariance from the workspace state and reports them via the
instrumentation schema. The harness shipped a test for this detector in
Stage -1 using synthetic collapsed and healthy states.

**When the guard fires**: The variance and covariance terms are added to the
training loss with configurable weights (default 1.0 each). The EMA decay
rate is configurable (default 0.99).

### D7. Narration decoder

**Choice**: A small linear projection from the workspace slot vectors (N x
768) to a fixed-length text description. The projection maps each slot to a
short token sequence, decoded via the Pythia-160M language modeling head. The
narration runs as a side-channel: it reads the workspace but does not modify
it.

**Triggering**: Controlled by a boolean in the run config (`narration:
true/false`, default `false`). When enabled, narration runs after each event
and writes to the `narration` field on the run record.

**Why build it now**: PLAN.md says "Build it in Stage 0 while the mapping is
still simple." The workspace has 16 slots of 768 dims. That is a tractable
input for a linear narration decoder. By Stage 2, the workspace state
includes fast weights, making narration much harder to retrofit.

### D8. GSM8K generator

**Choice**: A new generator `gsm8k` that wraps the GSM8K dataset from
Hugging Face (`gsm8k`). Each problem becomes an `Observe` event (the word
problem text) followed by a `Probe` event (asking for the numerical answer).
The truth channel carries the ground-truth answer.

**Subset support**: A `problem_count` config parameter limits how many
problems the generator uses. Default is all (~7.5k test, ~7.5k train).

**Chance rate**: Effectively zero. GSM8K answers are free-form integers. The
generator reports 0.0 as the chance rate.

**Stream structure**: No task switches or boundaries. The gate measures
accuracy on a single pass through the problems, not continual learning
across tasks. This matches the proposal's gate definition: "latent reasoning
matches token chain-of-thought accuracy."

**Dependency**: `datasets` (Hugging Face) added to `pyproject.toml`.

## Risks / Trade-offs

**[Frozen codecs may limit accuracy]**: The encoder and decoder are frozen
pretrained models with only thin learned layers on top. If the representation
gap between the sentence encoder's space and Pythia's embedding space is too
large, the cross-attention mapping may not bridge it.
Mitigation: unfreezing the codecs is a planned escape hatch. Run the frozen
version first. If accuracy is far below the chain-of-thought baseline,
unfreeze and re-run.

**[Latent loop may not converge with fixed steps]**: A fixed step count means
some problems get too few steps (answer is wrong) and some get wasted steps
(compute is wasted). This is expected in Stage 0.
Mitigation: Stage 1 adds adaptive halting. In Stage 0, sweep the step count
{4, 8, 16, 32} to find the best fixed value.

**[GSM8K may be too hard for 160M parameters]**: Chain-of-thought on GSM8K
requires reasoning that 160M-parameter models struggle with even in token
space. If the token-CoT baseline itself scores near zero, the gate comparison
is meaningless.
Mitigation: Run the token-CoT baseline first. If it scores below 5% on
GSM8K, substitute a simpler math dataset (e.g., MAWPS or a filtered GSM8K
subset with 1-2 step problems) and update the gate definition.

**[Parameter matching against baselines]**: The Stage 0 subject has Pythia-
160M (160M params) plus the sentence encoder (22M params) plus learned
projections. The baselines from Stage -1 are 2x400 MLPs (~320K params). These
are not comparable. The Stage 0 gate compares latent reasoning against token
chain-of-thought on the same model (Pythia-160M), not against Stage -1
baselines.
Mitigation: clearly separate the two comparisons in the gate report. The
Stage -1 baselines show continual-learning behavior. The Stage 0 gate shows
reasoning mode (latent vs. token).

**[Two Pythia-160M forward passes per latent step]**: The latent loop and the
EMA target encoder each run a forward pass. This doubles the compute per
latent step.
Mitigation: the EMA target runs with `torch.no_grad()`, which is cheaper than
a full training forward pass. The overhead is roughly 50% rather than 100%.
