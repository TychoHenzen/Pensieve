## Context

See `proposal.md` for motivation. `EggrollTrainer` currently generates each matrix perturbation as a dense `A @ B.T`, stacks the dense deltas for a candidate batch, and constructs one perturbed matrix per candidate. Gradient assembly regenerates those dense deltas and adds them in a Python loop. With population 128 and evaluation batch size 8, one training example executes 16 candidate batches. Each batch performs four latent Qwen passes and one decoder-loss pass.

The frozen Qwen context is identical for every candidate. Decoder-only causal attention also means context-token states cannot depend on the later slot positions. The selected state is the output after decoder block 11. Later decoder blocks and the vocabulary head cannot affect that selected state.

The official EGGROLL implementation at `ESHyperscale/HyperscaleES` commit `b77f7d6f91238fd575313e946b9cad21e0a74b32` defines matrix-factor draw order and update direction. The repository's current sampler draws the same tensor shape but assigns the leading rows to `A`; official EGGROLL assigns the leading `in_features` rows to `B`. This change corrects that deterministic mapping while remaining compatible with checkpoints and with the standalone and alternating commands.

## Goals / Non-Goals

**Goals:**

- Remove candidate-sized dense matrix perturbations from evaluation.
- Replace per-pair dense update accumulation with one factorized contraction per matrix parameter.
- Compute the frozen context prefix once per prepared example.
- Compute only the first 12 Qwen decoder blocks for latent states.
- Prove optimized and reference steps equivalent before selecting the optimized path.
- Measure complete EGGROLL step time and peak CUDA allocation under one reproducible benchmark.

**Non-Goals:**

- Change the EGGROLL fitness function, variance regularizer, sigma, rank, population, learning rate, or optimizer.
- Change the dataset or numerical-answer prompt.
- Optimize the decoder loss pass, which still requires vocabulary logits.
- Add a production switch between two trainers.
- Claim that a particular population or fitness policy learns better.

## Decisions

### 1. Represent perturbations as factors until the optimizer boundary

For a matrix `W` with shape `(out, in)`, generate one CPU noise tensor with shape `(in + out, rank)`. Following official EGGROLL, split the first `in` rows into `B` with shape `(in, rank)` and the remaining `out` rows into `A` with shape `(out, rank)`. Preserve the scale `sigma / sqrt(rank)` and store sign separately.

For input `x`, evaluate a candidate as:

```text
base = x @ W.T
residual = sign * sigma / sqrt(rank) * (x @ B) @ A.T
candidate = base + residual
```

The base projection is computed once for a candidate batch. Batched factor contractions add the candidate residuals. Dense vector perturbations remain stacked because they are already linear in parameter size and small relative to the matrices.

Alternative considered: materialize dense deltas and increase `eval_batch_size`. Rejected because memory still scales with `candidate_batch * out * in`, which prevents useful batch growth and contradicts the factorized EGGROLL computation.

### 2. Contract all matrix update factors into one dense optimizer gradient

After fitness normalization, compute one scalar coefficient per positive antithetic direction. Stack the positive-direction `A` and `B` factors. A single einsum-style contraction computes the weighted sum of `A_i @ B_i.T`, including the existing sigma and population scales. Because PyTorch Adam minimizes its supplied gradient, supply the negative fitness-gradient estimate so `optimizer.step()` moves the parameter toward the higher-fitness antithetic member. This creates only the final dense gradient required by Adam.

Vector parameters use one weighted reduction over their stacked seeded noise. Factor generation retains CPU generator order so a fixed seed produces the same noise as the reference path. The parameter registry and `optimizer.step()` order remain unchanged.

Alternative considered: keep the existing per-pair loop only for update assembly. Rejected because it regenerates and transfers a full matrix 64 times for each matrix parameter.

### 3. Introduce one layer-12 Qwen execution adapter

Add a shared adapter around the frozen Qwen model body. It exposes two operations:

- Prepare an immutable context prefix through decoder blocks 0 through 11.
- Run a batch of slot embeddings through those blocks using the prepared prefix and return only the final slot states.

The adapter uses the model's input embeddings, rotary-position behavior, attention mask rules, normalization, and decoder blocks. Slot positions begin at `context_length`. Each slot batch gets a fresh cache view containing the immutable context keys and values. Candidate keys and values must never be appended to the stored prefix object.

The adapter returns the state that the full model exposes at `hidden_states[12]`. It does not execute decoder blocks 12 through 23, the final model normalization when that occurs after the selected tap, or `lm_head`. The full causal-language-model wrapper remains available to `decoder_aligned_answer_loss`.

Alternative considered: call the full model with `output_hidden_states=True` and select tuple item 12. Rejected because that computes 12 unused decoder blocks and 151,936-way logits for every latent pass.

Alternative considered: use the public mutable `DynamicCache` instance directly across candidates. Rejected because candidate execution can append keys and values, contaminating later candidates. The stored prefix must be immutable.

### 4. Use the same adapter in ordinary and batched latent execution

`LatentLoop` and `EggrollTrainer` call the same layer-12 adapter. Ordinary gradient training may retain autograd through slot inputs while the frozen context prefix remains detached. EGGROLL candidate evaluation runs under inference mode. This prevents separate implementations from drifting on position identifiers, masks, layer selection, or shape checks.

The materialized full-model implementation moves behind a reference interface used by equivalence tests and the benchmark. Normal training commands cannot select it.

Alternative considered: optimize only the EGGROLL call site. Rejected because two latent execution implementations would continue to encode the same contract differently.

### 5. Prove equivalence in layers

Tests compare the smallest meaningful units before comparing a complete step:

1. Seeded factor reconstruction and row order against the pinned official EGGROLL implementation.
2. Factorized linear output against a materialized perturbed linear layer.
3. Factorized pseudo-gradient against the materialized pair loop, including an asymmetric-fitness case that proves the optimizer moves toward the higher-fitness perturbation.
4. Cached layer-12 state against full Qwen `hidden_states[12]`.
5. Candidate fitness, language-model loss, and shared variance.
6. Complete parameter and Adam-state updates.
7. Save, resume, and next-step behavior.

All comparisons use the spec tolerance. Allocation tests instrument tensor shapes so a numerical pass cannot hide a candidate-sized dense matrix allocation.

### 6. Benchmark a complete state-restored training step

Add `python -m train.benchmark_eggroll`. It loads the pinned assets once and prepares one persisted training example. Each warm-up and measured run restores identical trainable tensors, Adam state, workspace, and RNG state before timing a complete `train_step`. Model loading is outside the measurement. Prompt preparation, prefix preparation, candidate evaluation, pseudo-gradient assembly, and optimizer update are inside it.

The benchmark alternates reference and optimized measurements. It synchronizes CUDA around each measurement and resets peak allocated-memory statistics after state restoration. It emits one JSON record and exits nonzero when equivalence, speed, memory, or CUDA requirements fail.

Alternative considered: benchmark only projection kernels. Rejected because the user-visible problem is full example time, and isolated kernels would omit the frozen-backbone and decoder costs that bound total speedup.

### 7. Keep checkpoint and log identities stable

The optimization does not add a run configuration field. The mathematical algorithm, selected model state, optimizer state, and schedule remain the same. Existing version-2 checkpoints therefore remain loadable, and newly written checkpoints retain the current schema. The benchmark JSON is a separate verification artifact and is not embedded in training checkpoints.

## Risks / Trade-offs

- [Transformers changes Qwen attention or cache internals] -> Keep the adapter narrow, validate the pinned model shape, and fail equivalence tests on dependency upgrades.
- [A shared cache is mutated by one candidate] -> Store detached prefix tensors and construct a fresh non-owning cache view for every candidate batch and latent step.
- [Changed contraction order creates floating-point drift] -> Apply the declared tolerance at every layer and block production selection if a complete step exceeds it.
- [The decoder loss pass limits total speedup below 3x] -> Keep the gate blocking. Profile the measured optimized step before expanding scope.
- [Benchmark noise changes the median] -> Warm both paths, alternate their order, synchronize CUDA, use at least five measurements, and report every duration.
- [Full reference execution exceeds available memory beside optimized state] -> Restore one shared state between paths rather than retaining two Qwen models or two candidate populations.
- [Prefix caching changes gradient flow in gradient training] -> Compare slot-input gradients between cached and full execution before routing gradient training through the adapter.

## Migration Plan

1. Add canonical materialized reference helpers that follow the pinned official factor row order and optimizer direction.
2. Add factor objects, factorized linear evaluation, and factorized update assembly behind unit tests.
3. Add the layer-12 adapter and prove forward and slot-gradient equivalence.
4. Route EGGROLL candidate evaluation through the optimized components.
5. Run complete-step and resume equivalence tests.
6. Run the CUDA benchmark and record its JSON output.
7. Remove production reachability of the reference path after the gate passes.

Rollback is a normal code revert. Checkpoint migration is unnecessary because this change does not alter checkpoint contents.
