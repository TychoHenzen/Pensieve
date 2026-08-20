## 1. Progress Interval

- [x] 1.1 Add command tests for the 50-step default, positive overrides, and rejection of non-positive values before production loading; then update interval validation and defaults.
  <!-- status: completed -->
  <!-- covers: train/alternating-cycle :: Throttled progress output :: Default training progress interval -->
  <!-- covers: train/alternating-cycle :: Throttled progress output :: Configured training progress interval -->
  <!-- covers: train/alternating-cycle :: Throttled progress output :: Resumed training progress interval -->
  <!-- covers: train/alternating-cycle :: Throttled progress output :: Final step outside the progress interval -->
  <!-- covers: train/alternating-cycle :: Throttled progress output :: Invalid training progress interval -->

## 2. Output Filtering

- [x] 2.1 Add schedule-output tests proving training records emit only at configured global-step multiples while retaining their existing metric and position fields; then apply the training-record filter.
  <!-- status: completed -->
  <!-- covers: train/alternating-cycle :: Throttled progress output :: Non-progress output remains independent -->

- [x] 2.2 Add boundary-output tests proving epoch evaluation and checkpoint records remain visible while phase-only records stay silent without suppressing their operations; then apply the boundary-record filter.
  <!-- status: completed -->
  <!-- covers: train/alternating-cycle :: Throttled progress output :: Epoch boundary progress records -->
  <!-- covers: train/alternating-cycle :: Throttled progress output :: Phase-only boundary progress records -->
  <!-- covers: train/alternating-cycle :: Throttled progress output :: Coincident phase and epoch boundary progress records -->
  <!-- covers: train/alternating-cycle :: Throttled progress output :: Partial final phase at an epoch boundary -->

## 3. Verification

- [x] 3.1 Run the focused alternating-command, scheduler, evaluation, and checkpoint tests, then run the complete pytest suite.
  <!-- status: completed -->
