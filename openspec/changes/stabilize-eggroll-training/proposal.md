## Why

Pensive's EGGROLL factor orientation and ascent sign match the official reference, but the current Stage 0 training recipe does not. A fresh-state audit found that the Adam-based, single-problem update collapses question separation and leaves exact accuracy at zero, while the alignment-weight search can still report a relative optimum against that failed control.

## What Changes

- Match the official EGGROLL population-fitness normalization exactly, including population variance and epsilon placement.
- **BREAKING**: Give EGGROLL an independent momentum-free SGD optimizer, default its perturbation scale to `0.001`, and restrict EGGROLL updates to matrix parameters. Gradient training keeps Adam over the complete declared trainable registry.
- Evaluate every EGGROLL candidate over a deterministic batch of training problems. Count consumed examples, not optimizer calls, for dataset, epoch, observation-window, logging, and resume progress.
- Add an absolute EGGROLL stability gate that records an untrained baseline and bounded development checkpoints before alignment-weight selection or a full training run is accepted.
- Require alignment-weight search to report the EGGROLL method as unhealthy, with no recommendation, when its zero-weight control fails the absolute gate.
- Extend checkpoints and run identity with optimizer type, EGGROLL parameter scope, fitness-batch size, and example-consumption progress. Existing Adam-based EGGROLL checkpoints become incompatible with the new training contract.

## Capabilities

### New Capabilities

- `train/eggroll-stability-gate`: Defines bounded pre-training baselines, development checkpoints, absolute health thresholds, early stopping, and alignment-search eligibility.

### Modified Capabilities

- `train/eggroll-execution`: Changes fitness normalization, EGGROLL optimizer semantics, optimized-reference equivalence, parameter scope, and safe defaults.
- `train/stage0-training`: Changes EGGROLL from one problem per optimizer update to deterministic multi-problem fitness batches while preserving one consumption per epoch.
- `train/alternating-cycle`: Changes Eggroll optimizer ownership and defines example-count progress across batched Eggroll and single-example gradient updates.

## Impact

- Training code: `train/eggroll_updates.py`, `train/eggroll_trainer.py`, `train/training_state.py`, standalone and alternating runners, alignment-weight search, evaluation, and progress records.
- Persistence: Stage 0 run configuration, optimizer manifests, schedule state, checkpoint validation, and resume compatibility.
- Tests and benchmarks: official-reference normalization, SGD direction and magnitude, matrix-only updates, batched fitness, cursor boundaries, stability-gate behavior, alignment-search rejection, checkpoint resume, and CUDA equivalence/performance.
- Operations: full EGGROLL and alternating training must use fresh checkpoints after a bounded development gate passes.
