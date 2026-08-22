## 1. Reference Harness

- [x] 1.1 Add test-only materialized perturbation evaluation and pair-loop update helpers that pin the official EGGROLL factor row order and optimizer direction without changing production behavior.
<!-- covers: train/eggroll-execution :: Optimized execution preserves EGGROLL training semantics :: One optimized step matches the reference step -->
<!-- status: completed -->

- [x] 1.2 Add fixtures that snapshot and restore trainable tensors, Adam state, workspace state, and Python, NumPy, CPU, and CUDA RNG states around one EGGROLL step.
<!-- covers: train/eggroll-execution :: Optimized execution preserves EGGROLL training semantics :: Resume stays deterministic -->
<!-- status: completed -->

## 2. Factorized EGGROLL Computation

- [x] 2.1 Add seeded matrix-factor and dense-vector perturbation representations, then prove reconstruction, antithetic signs, scale, and the official `B`-then-`A` generator order against the reference sampler.
<!-- covers: train/eggroll-execution :: Matrix perturbations remain factorized during candidate evaluation :: Antithetic factors share one seed -->
<!-- covers: train/eggroll-execution :: Matrix perturbations remain factorized during candidate evaluation :: Vector perturbation remains compatible -->
<!-- status: completed -->

- [x] 2.2 Implement shared-base factorized linear evaluation for encoder and latent-loop matrices, with output-equivalence and forbidden-allocation shape tests.
<!-- covers: train/eggroll-execution :: Matrix perturbations remain factorized during candidate evaluation :: Matrix candidate batch uses low-rank residuals -->
<!-- status: completed -->

- [x] 2.3 Implement batched factor contraction for matrix pseudo-gradients and weighted reduction for vector pseudo-gradients, then compare gradients and Adam updates with the reference pair loop.
<!-- covers: train/eggroll-execution :: Pseudo-gradient assembly aggregates low-rank factors :: Factorized matrix update matches reference update -->
<!-- covers: train/eggroll-execution :: Pseudo-gradient assembly aggregates low-rank factors :: Assembly has one dense matrix result -->
<!-- covers: train/eggroll-execution :: Pseudo-gradient assembly aggregates low-rank factors :: Update follows fitness ascent -->
<!-- status: completed -->

## 3. Layer-12 Qwen Execution

- [x] 3.1 Add a shared Qwen tap adapter that validates the pinned hidden width and tap layer, returns the selected layer-12 slot state, and preserves slot autograd.
<!-- covers: core/latent-loop :: Latent loop feeds hidden state back as input :: hidden state feedback -->
<!-- covers: core/latent-loop :: Latent loop feeds hidden state back as input :: selected tap layer unavailable -->
<!-- status: completed -->

- [x] 3.2 Add immutable context-prefix preparation and fresh candidate cache views with the reference attention mask and position identifiers.
<!-- covers: core/latent-loop :: Latent loop feeds hidden state back as input :: Cached context is immutable -->
<!-- status: completed -->

- [x] 3.3 Stop latent-only execution after decoder block 11 and bypass the final model work and vocabulary head, with call-spy tests for omitted layers and logits.
<!-- covers: core/latent-loop :: Latent loop feeds hidden state back as input :: Latent execution omits unused model work -->
<!-- status: completed -->

- [x] 3.4 Compare cached and full-model layer-12 slot states and slot-input gradients on the pinned Qwen assets under the declared tolerance.
<!-- covers: core/latent-loop :: Latent loop feeds hidden state back as input :: Cached latent state matches full execution -->
<!-- status: completed -->

- [ ] 3.5 Route ordinary and batched latent runs through the shared adapter while preserving the projection, residual, normalization, shape checks, and no-token behavior.
<!-- covers: core/latent-loop :: Latent loop feeds hidden state back as input :: no intermediate tokens -->

## 4. Trainer Integration and Determinism

- [ ] 4.1 Route EGGROLL candidate evaluation and pseudo-gradient assembly through the factorized components and run candidate evaluation under inference mode.
<!-- covers: train/eggroll-execution :: Optimized execution preserves EGGROLL training semantics :: One optimized step matches the reference step -->

- [ ] 4.2 Add a complete-step equivalence test covering candidate fitness, language-model loss, variance, normalized fitness, emitted fields, tensors, Adam state, and final RNG state.
<!-- covers: train/eggroll-execution :: Optimized execution preserves EGGROLL training semantics :: One optimized step matches the reference step -->

- [ ] 4.3 Add alternating checkpoint-resume equivalence and CLI wiring tests, then make the optimized path the only production path for both training commands.
<!-- covers: train/eggroll-execution :: Optimized execution preserves EGGROLL training semantics :: Resume stays deterministic -->
<!-- covers: train/eggroll-execution :: Optimized execution preserves EGGROLL training semantics :: Existing commands select the optimized path -->

## 5. CUDA Performance Gate

- [ ] 5.1 Add `python -m train.benchmark_eggroll` with pinned configuration, state restoration, two warm-ups, five alternating measurements, CUDA synchronization, peak-memory measurement, and one JSON result.
<!-- covers: train/eggroll-execution :: CUDA benchmark proves a material speed improvement :: Performance gate passes -->

- [ ] 5.2 Make the benchmark reject unavailable CUDA and block performance reporting when numerical equivalence fails.
<!-- covers: train/eggroll-execution :: CUDA benchmark proves a material speed improvement :: Equivalence failure blocks the benchmark result -->
<!-- covers: train/eggroll-execution :: CUDA benchmark proves a material speed improvement :: CUDA is unavailable -->

- [ ] 5.3 Run the benchmark on the target CUDA device, retain its JSON evidence, and continue profiling within this change until median speedup is at least 3x with no peak allocated-memory increase.
<!-- covers: train/eggroll-execution :: CUDA benchmark proves a material speed improvement :: Performance gate passes -->

## 6. Verification

- [ ] 6.1 Run the focused CPU test suite for perturbations, latent execution, trainer integration, checkpoint resume, command wiring, and benchmark error paths.

- [ ] 6.2 Run the pinned-Qwen equivalence tests, the full project test suite, strict OpenSpec validation, and `dod-guard cover implement-hardware-efficient-eggroll`; record any environment-limited check as unverified rather than passed.
