## Context

See `proposal.md` for motivation. The repository currently has two independent trainers. Each constructs its own encoder, latent loop, tokenizer, optimizer, step result, metric calculation, and checkpoint format. Eggroll evaluates antithetic low-rank perturbations and writes an Adam pseudo-gradient. Gradient training backpropagates language-model and VICReg losses. Their logged variance values are not comparable because they are taken from different points in the forward pass with different aggregation.

The alternating run must process the same ordered GSM8K training list used today for every configured epoch. A phase boundary is based on the global count of completed training examples, not on epoch boundaries.

## Goals / Non-Goals

**Goals:**

- Give both update engines references to one encoder and one latent loop.
- Alternate update engines without resetting the dataset cursor or either optimizer.
- Make each run reproducible and resumable at phase and epoch boundaries.
- Produce metrics that show whether variance control corresponds to held-out task improvement.
- Keep the existing standalone commands usable.

**Non-Goals:**

- Switch methods dynamically from variance thresholds.
- Claim that a variance value is healthy before the experiment supplies evidence.
- Tune Eggroll population, perturbation rank, sigma, or either learning rate automatically.
- Change the frozen Pythia or MiniLM backbones.
- Make interrupted mid-phase steps resumable. Automatic checkpoints occur at completed phase and epoch boundaries.

## Decisions

### 1. Use a global fixed-step scheduler over nested epoch traversal

The orchestrator owns `epoch_index`, `example_index`, `global_step`, `phase`, and `phase_step`. It iterates through every training example for each epoch. After each update, it increments both `global_step` and `phase_step`. When `phase_step == phase_budget`, it evaluates, checkpoints, changes phase, and resets `phase_step` to zero. An epoch boundary records metrics and a checkpoint but does not change the phase.

This makes a 7,473-example epoch and a 500-step phase intentionally incommensurate. It samples each optimizer at different positions across later epochs and avoids repeatedly assigning the same examples to one optimizer.

Alternative considered: restart a 500-example block at every epoch. Rejected because that gives the start of every epoch to Eggroll and changes the requested continuous schedule.

### 2. Share modules while keeping update-engine state separate

Introduce a shared trainable-state object containing the encoder, workspace-facing latent loop, tokenizer, and named trainable tensors. Construct one gradient update engine and one Eggroll update engine against those same tensor objects. Each engine owns a separate Adam optimizer. Inactive optimizer state remains untouched.

Use a canonical ordered parameter registry. Both engines use the registry for common parameters. Any parameter that only one objective can update remains in the shared model and appears only in that engine's optimizer group. Checkpoint keys use parameter names rather than positional assumptions.

Alternative considered: copy state dictionaries between two complete trainers at every phase boundary. Rejected because copying can omit newly added trainable fields, duplicates frozen models in memory, and makes optimizer identity fragile.

### 3. Extract one forward result and one post-loop variance calculation

Both engines use a shared forward-result structure containing logits, answer targets, encoder slots, and post-loop slots. The common diversity metric is:

`post_loop_slots.var(dim=slot_axis, unbiased=False).mean()`

For batched Eggroll candidates, compute this per candidate and then aggregate candidates for the step record. For the unperturbed gradient and evaluation paths, compute the same formula on the unbatched post-loop slots. Keep VICReg's internal variance and covariance terms as objective diagnostics under distinct names. Do not label them as the shared variance.

Alternative considered: reuse `Workspace.collapse_stats()`. Rejected because the current call site measures a different tensor before the latent loop and uses a different estimator.

### 4. Preserve each method's objective while exposing language-model loss separately

Eggroll fitness remains `-lm_loss + variance_weight * log(shared_variance)`. Gradient training retains language-model loss plus VICReg regularization. Step records expose language-model loss separately from total objective and regularization terms. This prevents a lower combined objective from being mistaken for better answer prediction.

The experiment does not force both methods to use an identical objective. Its purpose is to test their complementary behavior. Objective alignment can be a later controlled experiment.

### 5. Evaluate a deterministic held-out GSM8K subset

Load the GSM8K test split once. Select a fixed prefix or seed-stable sample whose size is controlled by `--eval-problem-count`, default 128. Evaluation uses the unperturbed shared model in evaluation mode and without parameter updates. It reports mean teacher-forced language-model loss, mean shared variance, and generated-answer exact match.

Run evaluation after every completed phase, at every epoch boundary that does not coincide with a phase boundary, and once after a partial final phase. If boundaries coincide, emit one evaluation tagged with both boundary types.

Alternative considered: use training loss only. Rejected because it cannot distinguish useful alternation from objective gaming or memorization.

### 6. Save one versioned experiment checkpoint

Checkpoint payloads include:

- format version and complete run configuration
- named shared model state
- gradient optimizer state and Eggroll optimizer state
- epoch index, next example index, global step, active phase, phase step, and completed cycle count
- Python and PyTorch CPU/CUDA random-number-generator states
- held-out selection metadata and latest metrics

Write phase checkpoints as `phase-<global_step>.pt` and epoch checkpoints as `epoch-<epoch>.pt`. When both boundaries coincide, one payload may be written under both discoverable names. `--resume latest` selects the checkpoint with the largest global step, not the largest filename number.

Resume accepts non-schedule display settings such as logging frequency. It rejects changes to dataset selection, epoch target below completed progress, phase budget, model shape, optimizer hyperparameters, Eggroll population settings, or held-out selection.

Alternative considered: reuse either standalone checkpoint format. Rejected because neither contains phase position, both optimizer states, or full random state.

### 7. Add a dedicated command and keep standalone adapters

Add `python -m train.run_alternating` with defaults `--epochs 5` and `--phase-steps 500`. Reuse shared dataset, metric, model-state, and checkpoint helpers from the standalone commands where practical. Keep `train.run_training` and `train.run_eggroll` as single-method entry points backed by the refactored components.

The alternating command exposes both learning rates distinctly, for example `--gradient-lr` and `--eggroll-lr`, while retaining Eggroll population, sigma, rank, variance weight, evaluation batch size, and AMP controls.

## Risks / Trade-offs

- [Stale Adam moments may oppose the current shared parameters after an inactive phase] -> Log update norms at phase starts, retain separate state for the first experiment, and compare later with a reset-state ablation.
- [Eggroll and gradient objectives update different parameter subsets] -> Store a named parameter registry and report per-engine update coverage in run metadata.
- [Exact-match evaluation adds substantial runtime] -> Default to a fixed 128-problem subset and allow a smaller count for smoke tests.
- [A fixed dataset order can couple phase identity to example order] -> Keep current behavior for the first experiment and record every example position. A seeded-shuffle ablation remains possible later.
- [Checkpoint files are large and frequent] -> Save only at the specified boundaries and document retention as an operational choice.
- [The user's current edits in the Eggroll files overlap the likely implementation area] -> Preserve those edits and refactor against the resulting working tree during apply.

## Migration Plan

1. Add shared forward, metric, model-state, and result primitives with compatibility tests around both trainers.
2. Adapt the standalone trainers to accept shared state while preserving their command behavior.
3. Add and test the global phase scheduler independently with lightweight fake engines.
4. Add the alternating command, evaluation path, and versioned checkpoint resume.
5. Run deterministic short experiments before launching the five-epoch full run.

Rollback consists of removing the new alternating command and shared adapters. Existing standalone commands remain the operational fallback throughout implementation.
