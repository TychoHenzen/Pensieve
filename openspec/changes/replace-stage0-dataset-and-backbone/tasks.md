## 1. Shared identity and data contracts

- [x] 1.1 Add tests for canonical JSON Stage 0 identity, pinned revisions, Git-blob versus raw-SHA-256 manifest semantics, exact prompt messages and template arguments, and model and dataset load errors.
  <!-- status: completed -->
  <!-- covers: eval/generators/calc-mawps :: Filtered Calc-MAWPS split binding :: Default filtered splits load -->
  <!-- covers: eval/generators/calc-mawps :: Filtered Calc-MAWPS split binding :: Dataset revision unavailable -->
  <!-- covers: train/stage0-training :: Stage 0 identity accompanies training artifacts :: New checkpoint identity -->
- [x] 1.2 Implement the shared Stage 0 identity and official Qwen math-prompt module, replacing duplicated Stage 0 constants without changing runtime entry points yet.
  <!-- status: completed -->
- [x] 1.3 Add parser and scorer tests for the final delimiter, multiple numbers, signed integers, valid and invalid commas, exact fractions, equivalent decimals, inclusive tolerance boundaries, zero, division by zero, length limits, missing values, NaN, infinity, percentages, and scientific notation.
  <!-- status: completed -->
  <!-- covers: eval/stage0-gate :: Numerical answer equivalence :: Equivalent fraction and decimal -->
  <!-- covers: eval/stage0-gate :: Numerical answer equivalence :: Repeating decimal tolerance -->
  <!-- covers: eval/stage0-gate :: Numerical answer equivalence :: Non-finite answer -->
- [x] 1.4 Extend the existing answer scorer with finite numerical normalization, exact rational comparison, and `1e-6` relative and absolute tolerance.
  <!-- status: completed -->
- [x] 1.5 Add injected-row tests for exact Unicode and whitespace normalization, source identifiers, the one pinned validation exclusion, all other cross-split duplicate rejection, `Decimal(str(result_float))` target agreement, pinned split counts, omitted-count normalization, deterministic selection identities and limits, and v1 truth isolation.
  <!-- status: completed -->
  <!-- covers: eval/generators/calc-mawps :: Canonical math problem records :: Valid row normalization -->
  <!-- covers: eval/generators/calc-mawps :: Canonical math problem records :: Malformed row rejected -->
  <!-- covers: eval/generators/calc-mawps :: Calc-MAWPS stream follows the v1 schema :: Observe and probe pair -->
  <!-- covers: eval/generators/calc-mawps :: Calc-MAWPS stream follows the v1 schema :: Truth remains isolated -->
  <!-- covers: eval/generators/calc-mawps :: Selection identity is deterministic :: Repeated selection -->
  <!-- covers: eval/generators/calc-mawps :: Selection identity is deterministic :: Selection input changes -->
- [x] 1.6 Implement the shared Calc-MAWPS record adapter, register its stream generator, and preserve the unrelated GSM8K generator.
  <!-- status: completed -->

## 2. Frozen Qwen backbone and 896-dimensional workspace

- [x] 2.1 Add fake-model tests for revision reuse across model and tokenizer assets, pinned manifest verification, disabled remote code, safetensors, evaluation mode, complete freezing, public input embeddings, width 896, tuple tap 12, shared ownership, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization.
  <!-- status: completed -->
  <!-- covers: core/latent-loop :: Latent loop feeds hidden state back as input :: selected tap layer unavailable -->
  <!-- covers: core/latent-loop :: Model-neutral token embedding access :: Qwen context embedding -->
  <!-- covers: train/stage0-training :: Shared frozen Qwen backbone :: Backbone parameters remain frozen -->
  <!-- covers: train/stage0-training :: Shared frozen Qwen backbone :: Shared model instance -->
- [ ] 2.2 Implement the model-neutral frozen Qwen loader and route subject and training state construction through it.
- [ ] 2.3 Update workspace, encoder, decoder, and narration shape tests from 768 to 896, including explicit decoder mismatch errors and unchanged slot-count ablations.
  <!-- covers: workspace/concept-slots :: Workspace holds a configurable number of concept slots :: default slot count -->
  <!-- covers: workspace/concept-slots :: Workspace holds a configurable number of concept slots :: configurable slot count -->
  <!-- covers: workspace/concept-slots :: Workspace holds a configurable number of concept slots :: ablation sweep values accepted -->
  <!-- covers: codecs/encoder :: Encoder maps text to workspace slots :: text encoded to slots -->
  <!-- covers: codecs/encoder :: Encoder maps text to workspace slots :: variable-length input accepted -->
  <!-- covers: codecs/decoder :: Decoder maps workspace slots to text :: slots decoded to text -->
  <!-- covers: codecs/decoder :: Decoder maps workspace slots to text :: different slot states produce different text -->
  <!-- covers: codecs/decoder :: Decoder maps workspace slots to text :: incompatible slot width -->
- [ ] 2.4 Change the workspace and connected projections to 896 dimensions, remove Pythia-specific shape assumptions, and derive collapse-stat normalization from actual width.
- [ ] 2.5 Add latent-loop tests proving one unpadded `[chat_context, slots]` batch, complete hidden-state shape validation, the exact `outputs.hidden_states[12][0, -slot_count:, :]` slice, feedback through the declared projection and LayerNorm equation, no token sampling, and clear incompatible-shape failure.
  <!-- covers: core/latent-loop :: Latent loop feeds hidden state back as input :: hidden state feedback -->
  <!-- covers: core/latent-loop :: Latent loop feeds hidden state back as input :: no intermediate tokens -->
- [ ] 2.6 Update the latent loop to tap Qwen layer 12 and feed each normalized 896-dimensional workspace update directly into the next step.

## 3. Stage 0 training integration

- [ ] 3.1 Add contract tests proving all trainers reuse the complete seed-0 train order each epoch while held-out evaluation uses only the first 128 persisted seed-0 validation identifiers.
  <!-- covers: train/stage0-training :: Shared Stage 0 dataset contract :: Training modes share examples -->
  <!-- covers: train/stage0-training :: Shared Stage 0 dataset contract :: Held-out split isolation -->
- [ ] 3.2 Route all three training entry points and held-out evaluation through the shared dataset adapter and Stage 0 identity.
- [ ] 3.3 Extend decoder-aligned objective tests with canonical integer, decimal, fraction, and negative targets. For `K` answer tokens, assert `[slots, all K answer tokens]` inputs and equal-length labels containing `slot_count - 1` ignored entries followed by all `K` tokens and EOS.
  <!-- covers: train/stage0-training :: Decoder-aligned numerical objective :: Multi-token numerical target -->
  <!-- covers: train/stage0-training :: Decoder-aligned numerical objective :: Training prompt contract -->
- [ ] 3.4 Adapt gradient and Eggroll training to the shared Qwen tokenizer and context contract while preserving the existing answer-objective repairs and frozen-backbone policy.
- [ ] 3.5 Update alternating-schedule tests for 1,089 examples per epoch, 5,445 default steps, uninterrupted phase budgets, and invalid epoch rejection.
  <!-- covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Phase boundary within an epoch -->
  <!-- covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Epoch boundary within a phase -->
  <!-- covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Full default run -->
  <!-- covers: train/alternating-cycle :: Complete multi-epoch dataset traversal :: Invalid epoch count -->
- [ ] 3.6 Update human-readable descriptions and defaults in the standalone and alternating CLIs to name filtered Calc-MAWPS and frozen Qwen.

## 4. Checkpoint and result compatibility

- [ ] 4.1 Add checkpoint tests for a `ZIP_STORED` `.ckpt` containing exactly bounded `metadata.json` and `tensors.safetensors` members, with no pickle or `torch.load` path. Cover duplicate, compressed, traversing, missing, extra, oversized, malformed, and unknown-version members.
  <!-- covers: train/alternating-cycle :: Resumable experiment checkpoints :: Resume within an epoch -->
  <!-- covers: train/alternating-cycle :: Resumable experiment checkpoints :: Resume at an epoch boundary -->
  <!-- covers: train/alternating-cycle :: Resumable experiment checkpoints :: Reject incompatible resume configuration -->
  <!-- covers: train/stage0-training :: Stage 0 identity accompanies training artifacts :: Legacy artifact rejected -->
- [ ] 4.2 Add schema tests for version-2 metadata, gradient, Eggroll, and alternating modes, model and optimizer tensor manifests, every RNG source, complete alternating schedule state, metrics, and ordered train and held-out selections.
- [ ] 4.3 Implement the version-2 JSON plus safetensors checkpoint writer with canonical metadata, allowed tensor names, bounded members, and atomic replacement.
- [ ] 4.4 Implement CPU checkpoint loading that validates ZIP structure and metadata before reading safetensors, then validates every identity, name, shape, dtype, role, schedule, selection, and RNG field before applying any state.
- [ ] 4.5 Add compatible-resume tests and side-effect-free rejection tests for legacy checkpoints, mismatched identities, unexpected tensors, incomplete optimizer state, and incompatible schedules.

## 5. Comparable Stage 0 gate

- [ ] 5.1 Add token-baseline tests for exact Qwen messages, template arguments, greedy generation settings, all 520 ordered test identifiers, non-gating bounded development selection, and deterministic CUDA runtime identity.
  <!-- covers: eval/stage0-gate :: Frozen Qwen token baseline :: Full token baseline -->
  <!-- covers: eval/stage0-gate :: Frozen Qwen token baseline :: Official prompt format -->
- [ ] 5.2 Route the token baseline through the shared identity, prompt, dataset records, and numerical scorer, and include per-item identities in its JSON.
- [ ] 5.3 Add latent-evaluation tests proving full and limited runs consume the token path's persisted ordered identifiers, emit the shared `{item_id, prediction, target, correct}` item schema, and bind identity to the canonical ordered `{item_id, input_ids}` digest.
  <!-- covers: eval/stage0-gate :: Comparable test selection :: Full evaluation identity -->
  <!-- covers: eval/stage0-gate :: Comparable test selection :: Limited evaluation identity -->
- [ ] 5.4 Route latent evaluation through the shared records and scoring contract without per-seed resampling of selected test items.
- [ ] 5.5 Add gate-cache tests for typed schema validation, checkpoint digest, `rendered_inputs_sha256`, complete inference and runtime identity, exact compatible reuse, per-item coverage, recomputed aggregates, atomic writes, and rejection of absent or changed fields.
  <!-- covers: eval/stage0-gate :: Result compatibility identity :: Matching cached baseline -->
  <!-- covers: eval/stage0-gate :: Result compatibility identity :: Stale cached baseline -->
- [ ] 5.6 Add gate-report tests for the exact 26-of-520 floor, exactly seeds `[0,1,2,3,4]`, RNG initialization, missing or extra seed invalidation, and latent-mean comparison.
  <!-- covers: eval/stage0-gate :: Stage 0 pass condition :: Baseline below floor -->
  <!-- covers: eval/stage0-gate :: Stage 0 pass condition :: Too few latent seeds -->
  <!-- covers: eval/stage0-gate :: Stage 0 pass condition :: Latent result meets baseline -->
- [ ] 5.7 Update `run_gate` and the report builder to use a Calc-MAWPS/Qwen result location, enforce identity before reuse, and preserve the existing pass rule.

## 6. Verification and preflight

- [ ] 6.1 Run targeted dataset, scoring, backbone, shape, training, checkpoint, latent, baseline, and gate tests through `.\.venv\Scripts\python.exe -m pytest`.
- [ ] 6.2 Run `dod-guard cover replace-stage0-dataset-and-backbone` and bind every declared scenario to its named test.
- [ ] 6.3 Run the complete pytest suite and distinguish new failures from unrelated existing worktree changes or storage errors.
- [ ] 6.4 Check available disk space, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization, then run a one-problem integration smoke test against both pinned Hugging Face revisions on CUDA.
- [ ] 6.5 Run the frozen Qwen token baseline across all 520 test items and record whether it clears the unchanged 5% validity floor before any new multi-epoch latent training.
