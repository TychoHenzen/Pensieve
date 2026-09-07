## Why

Gradient training collapses workspace-slot variance while Eggroll evolution increases it, and neither mode alone provides a stable training trajectory. A fixed-budget alternating experiment is needed to test whether the two update methods can preserve representation diversity while improving GSM8K loss across complete multi-epoch training runs.

## What Changes

- Add an alternating training entry point that starts with Eggroll and switches update methods every 500 examples by default.
- Continue one dataset traversal across phase boundaries so every configured epoch still processes the full training set exactly once.
- Default to the existing five-epoch experiment length while allowing the epoch count and phase budget to be configured.
- Use one shared post-latent-loop slot variance metric in both phases so phase results are comparable.
- Preserve one model state across phase changes while retaining separate optimizer state for Eggroll and gradient updates.
- Record phase, cycle, global step, epoch position, language-model loss, shared variance, and evaluation results.
- Save resumable checkpoints that restore the dataset position, active phase, model parameters, and both optimizer states.

## Capabilities

### New Capabilities

- `train/alternating-cycle`: Fixed-budget alternation between Eggroll and gradient updates over complete multi-epoch dataset traversals.

### Modified Capabilities

None.

## Impact

- Adds a training orchestrator and command-line entry point under `train/`.
- Refactors the existing trainers enough to share model parameters, a common step result, and the same variance measurement without changing their standalone commands.
- Adds unit and integration tests for phase scheduling, epoch coverage, state continuity, metrics, checkpoint resume, and deterministic short runs.
- Uses the existing PyTorch, Transformers, GSM8K, and checkpoint dependencies.
