## Stage 0: Latent core

This change implements Stage 0 from PLAN.md. It builds a latent reasoning
subject that operates vector-to-vector, with tokens only at the edges,
and passes a gate comparing latent reasoning accuracy against token
chain-of-thought on grade-school math word problems.

### Entry condition

Stage -1 gate passed. The reproduction gate ran five seeds on
class-incremental Split-MNIST and matched van de Ven et al. 2019 within
tolerance (`gate_results/gate_report.json`). The subject protocol, stream
schema, and metric definitions are frozen at v1
(`docs/STAGE-MINUS-1-CONTRACT.md`).

### What this change delivers

1. A workspace built on configurable concept slots (1 to 64, default 16),
   each a 768-dim vector holding all latent state. Snapshot and restore
   are bitwise-identical.
2. A latent loop that wraps Pythia-160M: each step reads the slot vectors
   as input embeddings, runs a forward pass, and writes hidden states back
   to the slots through a learned projection. No token decoding between
   steps.
3. An encoder that maps variable-length text into workspace slots via a
   frozen pretrained sentence encoder (all-MiniLM-L6-v2) and a learned
   cross-attention module.
4. A decoder that feeds workspace slot vectors into Pythia-160M's language
   modeling head and generates text autoregressively.
5. A narration decoder (audit channel) that reads the workspace and
   produces human-readable text on a side channel.
6. VICReg-style latent collapse guards wired into the training objective.
7. A GSM8K generator that wraps the GSM8K dataset as a deterministic
   seeded stream of Observe and Probe events.
8. A training pipeline: frozen Pythia-160M backbone, frozen sentence
   encoder, training only the cross-attention mapping, the hidden-state
   projection, and (if needed) an adaptation layer on the LM head.
   Trained on GSM8K train split.
9. A v1-compliant subject implementing all six protocol methods (observe,
   answer, idle, snapshot, restore, cost).
10. Gate evaluation: latent reasoning vs token chain-of-thought on
    GSM8K test split, slot count ablation {1,4,8,16,32,64}, and a cycling
    sweep on FashionMNIST alongside MNIST.

### Contract obligations inherited from Stage -1

The Stage 0 subject must implement all six methods of the v1 subject
protocol: `observe`, `answer`, `idle`, `snapshot`, `restore`, `cost`.

Two constraints are load-bearing and non-obvious:

- **`restore()` must be exact.** The workspace state, including all
  concept slots, must round-trip through `snapshot()`/`restore()` with
  zero information loss. Probe isolation depends on this.
- **Parameter matching.** The baseline comparison requires the Stage 0
  subject to have a comparable parameter count to the baselines it is
  measured against. `eval/baselines/param_match.py` enforces this with a
  configurable tolerance.

### Decisions closed by the design

All seven open decisions from PLAN.md are closed in `design.md`:

1. Base model: Pythia-160M (D1)
2. COCONUT integration: fixed latent steps, workspace as slot set (D2)
3. Encoder: sentence-transformers with cross-attention slot mapping (D3)
4. Decoder: Pythia-160M autoregressive from slot embeddings (D4)
5. Training: frozen backbone on GSM8K train split (D5)
6. Collapse guards: VICReg variance + covariance + EMA target (D6)
7. Narration decoder: linear projection through Pythia LM head (D7)
8. GSM8K generator: wraps Hugging Face GSM8K dataset (D8)

### Risks / Trade-offs

See `design.md` for the full risk analysis. The three that most affect
implementation order:

- Frozen codecs may limit accuracy. Escape hatch: unfreeze and retrain.
- Fixed latent step count wastes compute on easy problems. Accepted for
  Stage 0; adaptive halting is Stage 1.
- GSM8K may be too hard for 160M params. Run the token-CoT baseline
  first. If it scores below 5%, substitute a simpler dataset.

### Retreat option

None. The gate is go/no-go on the entire project.

### Prior results: curriculum cycling experiment

Replay peaks at 10 cycles on both MNIST and FashionMNIST, closing
roughly half the gap to the joint oracle. FashionMNIST is harder and
more discriminating. The gate evaluation includes a cycling sweep to
measure whether latent reasoning handles re-exposure better than replay.

### Impact

- New packages: `workspace/`, `codecs/`, `train/`.
- New generator: `eval/generators/gsm8k/`.
- New subject: `eval/subjects/latent_core/`.
- `pyproject.toml` gains: `transformers`, `sentence-transformers`,
  `datasets`, `torch` (if not already present).
- No changes to the existing `eval/` harness code. Stage 0 implements
  the v1 protocol; it does not extend it.

### Estimated effort

Weeks of GPU time across multiple sessions. Includes pretraining the
learned projection layers on GSM8K and running the gate evaluation with
multiple seeds and slot count ablation.
