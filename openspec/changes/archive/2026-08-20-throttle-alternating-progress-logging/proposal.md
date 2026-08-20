## Why

The alternating training command currently emits one JSON training record per example, which makes long runs difficult to monitor. Progress output should remain useful without changing training, evaluation, checkpoint, or resume behavior.

## What Changes

- Change the default training progress interval from 10 steps to 50 steps.
- Emit training progress only at the configured interval.
- Emit evaluation and checkpoint progress at epoch boundaries.
- Keep phase-boundary evaluation and checkpoint operations unchanged but silent.
- Reject non-positive progress intervals before loading models or data.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `train/alternating-cycle`: Define throttled, configurable progress output without changing optimizer cycles or boundary operations.

## Impact

- Updates `train/run_alternating.py` progress filtering and argument validation.
- Updates focused alternating-command tests.
- Does not change training algorithms, phase scheduling, evaluation, checkpoint persistence, resume behavior, or standalone commands.
