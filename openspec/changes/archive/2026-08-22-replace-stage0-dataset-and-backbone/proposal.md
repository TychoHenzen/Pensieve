## Why

The Stage 0 gate is invalid because its frozen Pythia-160M token baseline scores below the declared 5% floor on GSM8K. The repository's Stage 0 design calls for a simpler math dataset when that happens, while the replacement backbone must provide a meaningful frozen token baseline.

## What Changes

- Replace the Stage 0 training and gate dataset with the filtered `MU-NLPC/Calc-mawps` configuration and pin its revision.
- Replace the shared frozen language-model backbone with pinned `Qwen/Qwen2.5-0.5B-Instruct` and use its official chat template.
- **BREAKING** Change workspace slots and their connected projections from 768 to Qwen's 896 hidden dimensions, with latent updates taken from layer 12.
- Add numerical answer normalization so equivalent fractions and decimals compare equal within a declared tolerance.
- Make dataset, backbone, revision, and evaluated-item identity part of checkpoints and gate result compatibility.
- **BREAKING** Reject GSM8K/Pythia checkpoints and stale result caches instead of migrating them.
- Preserve the Stage 0 pass rule: the full 520-item token baseline must reach 5%, and the five-seed latent mean must meet or exceed it.

## Capabilities

### New Capabilities

- `eval/generators/calc-mawps`: Deterministic filtered Calc-MAWPS streams with pinned data identity and finite numerical targets.
- `eval/stage0-gate`: Dataset-aware token and latent evaluation, numerical equivalence scoring, cache compatibility, and the unchanged Stage 0 pass rule.
- `train/stage0-training`: Shared Calc-MAWPS and frozen-Qwen contracts for standalone gradient, standalone Eggroll, and alternating training.

### Modified Capabilities

- `workspace/concept-slots`: Change each workspace slot from 768 to 896 dimensions.
- `codecs/encoder`: Project MiniLM token embeddings directly into 896-dimensional workspace slots.
- `codecs/decoder`: Decode 896-dimensional slots through the shared frozen Qwen backbone.
- `core/latent-loop`: Use Qwen2.5-0.5B-Instruct, model-neutral token embeddings, and hidden layer 12 for latent updates.
- `train/alternating-cycle`: Bind schedules and checkpoints to Calc-MAWPS and Qwen identities and reject incompatible resumes.

## Impact

The change affects the Stage 0 stream registry, all three training entry points, shared training state, workspace and codec shapes, latent-loop model access, held-out evaluation, token and latent gate runners, checkpoint metadata, gate result JSON, and their tests. Existing GSM8K stream use remains available outside the Stage 0 training and gate defaults. Existing Pythia/GSM8K checkpoints and gate results become incompatible by design. The project continues to use the existing `datasets`, `transformers`, `torch`, and `sentence-transformers` dependencies.
