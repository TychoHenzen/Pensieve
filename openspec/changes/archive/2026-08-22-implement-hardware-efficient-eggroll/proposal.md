## Why

Stage 0 EGGROLL training takes about 13 seconds per example on the current CUDA run because it expands low-rank perturbations into full weight matrices and repeatedly recomputes the frozen Qwen context. The implementation needs to preserve EGGROLL's factorized computation and avoid backbone work whose results are discarded before another multi-hour experiment is useful.

## What Changes

- Keep every EGGROLL perturbation as low-rank factors during candidate evaluation and pseudo-gradient assembly. Do not materialize one full perturbed weight matrix per candidate.
- Share each unperturbed projection across candidates and apply candidate-specific low-rank residuals with batched factor operations.
- Reuse the frozen Qwen question-prefix state across candidates and latent steps when causal equivalence is proven.
- Run latent passes only through the Qwen transformer layers needed to produce the selected layer-12 state. Do not compute discarded vocabulary logits or later layers.
- Add reference-path equivalence tests for candidate fitness, post-loop slots, pseudo-gradients, optimizer updates, and deterministic resumed execution. Pin factor draw order and update direction to the official EGGROLL implementation at https://github.com/ESHyperscale/HyperscaleES.
- Add a reproducible CUDA benchmark gate. The optimized path must achieve at least a 3x median speedup over the reference path on the same device and workload without increasing peak allocated CUDA memory.
- Preserve the existing commands, defaults, objective, candidate order, antithetic pair seeds, checkpoint schema, and logged training record fields. Correct the seeded matrix-factor row order to official EGGROLL semantics: draw `B` from the first `in_features` rows and `A` from the remaining `out_features` rows.

## Capabilities

### New Capabilities

- `train/eggroll-execution`: Defines factorized candidate evaluation, factorized update assembly, equivalence checks, execution telemetry, and the performance acceptance gate.

### Modified Capabilities

- `core/latent-loop`: Allows a cached and partial-backbone execution path only when it produces the same selected layer-12 slot states as the declared full Qwen execution.

## Impact

- Primary implementation areas are `train/eggroll_trainer.py`, the shared latent-loop execution code, and their tests.
- Standalone EGGROLL and alternating training use the optimized path through their existing command-line interfaces.
- The reference path remains test-only for equivalence and benchmarking. It is not a second production trainer.
- No dataset, objective, optimizer hyperparameter, checkpoint format, or evaluation-gate contract changes in this change.
