## Context

See `proposal.md` for motivation. Stage 0 currently repeats GSM8K and Pythia identifiers across generators, three training commands, held-out evaluation, subjects, and gate runners. `Workspace` is fixed at 768 dimensions, and `LatentLoop.embed_tokens` reaches through Pythia's private `gpt_neox.embed_in` path. Gate cache files do not prove that their dataset, model, prompt, scorer, and item selection match the requested run.

The worktree already contains uncommitted decoder-aligned answer-objective and numerical-scoring repairs. Implementation must preserve and extend those edits. The machine has an 8 GB RTX 4060 Laptop GPU, so Qwen remains frozen. Checkpoints contain only trainable state, optimizer state, schedule state, selection state, RNG state, and identity metadata in a bounded JSON plus safetensors container.

## Goals / Non-Goals

**Goals:**

- Give every Stage 0 path one authoritative dataset, backbone, prompt, scoring, and identity contract.
- Feed 896-dimensional workspace slots directly to Qwen without a new input adapter.
- Reject incompatible checkpoints and caches before expensive model construction or training.
- Keep small injected tests independent from model and dataset downloads.

**Non-Goals:**

- Fine-tune, quantize, or otherwise modify Qwen parameters.
- Migrate Pythia/GSM8K checkpoints or cached results.
- Remove GSM8K from the general stream registry.
- Change the sentence-transformer backbone.
- Run the full five-epoch training and five-seed gate during implementation verification.

## Decisions

### 1. One immutable Stage 0 identity

Introduce one immutable identity value used by loaders, trainers, checkpoints, and gate results. It contains:

- dataset `MU-NLPC/Calc-mawps`
- configuration `default`
- dataset revision `38c10053efeafd20ab6ff4e08c3ec17de26c19b7`
- model `Qwen/Qwen2.5-0.5B-Instruct`
- model revision `7ae557604adf67be50417f59c2c2f167def9a775`
- workspace dimension 896
- latent tap layer 12
- prompt contract version
- numerical scorer contract version

The identity serializes as UTF-8 canonical JSON with sorted keys and compact separators. Checkpoint and result readers compare the complete typed identity before reuse. A missing, malformed, or unknown-version field is incompatible. This replaces filename-based assumptions and scattered string constants.

Pinned artifact manifests distinguish two digest forms. Forty-character Hugging Face manifest values are Git blob object identifiers and are verified with Git blob framing. Sixty-four-character safetensors and Parquet values are raw-file SHA-256 digests.

Alternative considered: allow callers to mix arbitrary datasets and backbones. That would broaden this repair into a general experiment-configuration system and permit unsupported model shapes.

### 2. Shared Calc-MAWPS record adapter

Add a dataset adapter that returns an immutable record containing item id, split, normalized question, canonical target, and source metadata. All training and evaluation paths consume these records instead of importing private generator helpers.

The adapter loads only the three pinned `default` Parquet files and verifies their SHA-256 digests. It checks raw cardinalities of 1,089 train, 1,040 validation, and 520 test. A live audit found one exact validation/test duplicate, so the adapter excludes validation id `mawps__qA0gWJatQEeMzvOw`, leaving 1,039 usable validation records, and rejects any other cross-split id or question-target overlap. It uses source `id` and `result`, checks `result_float`, applies the exact NFC and whitespace algorithm, and validates every row before model construction.

Selection starts in source order, uses `random.Random(seed).shuffle`, and applies the optional prefix limit. An omitted problem count normalizes to the usable split cardinality before selection identity is serialized or hashed. The selection identity contains its schema version, dataset coordinates, split, seed, normalized problem count, ordered item identifiers, and a record-content digest. Training persists the complete seed-0 train selection and reuses it each epoch. Held-out evaluation persists the first 128 identifiers from the seed-0 validation selection. The gate persists the complete seed-0 test selection. The stream generator wraps the same records as `Observe` and `Probe` pairs but passes only public question text to model-facing code.

`result_float` enters exact decimal validation through `Decimal(str(result_float))`. This preserves the source value's published decimal spelling instead of treating its in-memory binary expansion as the declared target.

Alternative considered: patch each GSM8K import to load Calc-MAWPS directly. That would retain duplicated parsing, selection, and identity behavior.

### 3. Numerical targets use exact values when available

Normalize integer, decimal, and fractional target forms through `fractions.Fraction` and `decimal.Decimal`. Prediction extraction uses this ASCII-only bounded grammar: `(?<![A-Za-z0-9_.,])[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)/(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)|(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:[.][0-9]+)?)(?![A-Za-z0-9_.,/%])`. It takes the first numeric literal after the final `####` delimiter, or the final literal when no delimiter exists. It rejects non-ASCII digits, scientific notation, percentages, invalid comma grouping, outputs over 4,096 characters, numeric components over 256 digits, NaN, infinity, and zero denominators. It never calls `eval` or another expression evaluator.

Exact values compare first. Otherwise comparison uses the inclusive formula `abs(a-b) <= max(1e-6, 1e-6 * max(abs(a), abs(b)))`. Canonical target serialization removes grouping and leading plus signs, reduces fractions, makes denominators positive, removes decimal trailing zeros, and emits integer forms for negative zero and denominator-one values.

Result JSON stores the original target text and canonical normalized value. This keeps failures inspectable without using equation strings as truth.

Alternative considered: filter to integer answers. That would silently change the declared filtered Calc-MAWPS test population.

### 4. Qwen is shared, frozen, and accessed through public interfaces

Create a single backbone loader for model, tokenizer, config, chat template, and generation config at the pinned revision. It passes `trust_remote_code=False`, requires safetensors, verifies the resolved commit and pinned manifests using their declared digest semantics, checks `hidden_size == 896`, calls `eval()`, freezes every parameter, and exposes token embeddings through `model.get_input_embeddings()`.

Deterministic CUDA execution requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` before any CUDA initialization. Runtime identity records this value so a result cache cannot cross deterministic-runtime contracts.

`TrainingState` owns the shared Qwen instance. Gradient and Eggroll trainers reuse it. `LatentCoreSubject`, held-out evaluation, and gate checkpoint loading use the same loader and identity. Tests inject fake models and tokenizers rather than downloading Qwen.

Alternative considered: keep Pythia-specific access branches. Explicit legacy rejection makes those branches unnecessary in the Stage 0 runtime.

### 5. The workspace adopts Qwen's 896-dimensional width

Change the workspace constant, sentence-to-slot projection, slot queries, normalization layers, and shape checks to 896. The latent hidden-state projection remains trainable but becomes 896-to-896. Each latent iteration sends one unpadded batch shaped `[chat_context, slots]`. After verifying `outputs.hidden_states[12]` has shape `(1, context_length + slot_count, 896)`, it defines `h = outputs.hidden_states[12][0, -slot_count:, :]`. This is the output after zero-based block 11. Slots enter Qwen directly as `inputs_embeds`, so no 768-to-896 input adapter is introduced.

The decoder and narration paths obtain their expected input width from the shared backbone contract and reject mismatches before generation. Collapse statistics derive their normalization denominator from the actual slot width instead of a stale literal.

Alternative considered: retain 768-dimensional slots and train an input adapter. That preserves an arbitrary Pythia dimension and creates an additional learned failure surface.

### 6. Prompt handling is explicit by role

One prompt module owns this exact versioned user content: `Solve the following math word problem. Show your reasoning, then end your response with exactly "#### <answer>", where <answer> is a finite integer, decimal, or fraction.\n\nProblem:\n{question}`. It creates one user message, no system message, and applies Qwen's official template with tokenization and `add_generation_prompt=True`.

The token baseline uses deterministic greedy generation with a 512-token context limit and 64 generated tokens. Latent training and evaluation use the same chat-formatted problem as Qwen context. Each latent pass receives `[chat_context, slots]`, and decoding receives `[slots, generated_answer_prefix]`. For `K` answer tokens, teacher forcing uses `[slots, all K answer tokens]`, with total length `slot_count + K`. Its labels have the same length: the first `slot_count - 1` entries are `-100`, followed by all `K` answer tokens and EOS. MiniLM encodes normalized problem text without Qwen control tokens. The target remains canonical numerical text plus end-of-sequence.

Result identity includes `rendered_inputs_sha256`. The digest covers canonical JSON for the ordered list of `{item_id, input_ids}`. This binds cache reuse to the exact official chat-template output rather than only a prompt version label.

Alternative considered: retain the legacy plain-text Pythia prompt. That discards the selected instruction model's declared interface and weakens the baseline preflight.

### 7. Compatibility is checked before expensive work

Version-2 checkpoints use a `.ckpt` ZIP container with `ZIP_STORED`. The archive contains exactly `metadata.json`, bounded to 1 MiB, and `tensors.safetensors`, bounded to 64 MiB. The loader never calls `torch.load` and never accepts a pickle payload. It rejects extra members, duplicate members, compressed members, invalid paths, oversized declared members, and oversized actual members before decoding either payload.

Metadata contains exactly the versioned identity, mode, tensor manifest, optimizer manifests, schedule, selections, metrics, and RNG state. Model tensor names use `model.<allowed_trainable_parameter_path>`. Optimizer tensor names use `optimizer.<method>.<parameter_path>.<state_name>`. Tensor manifests declare every name, shape, dtype, and role. Alternating schedule metadata records the active phase, completed phase steps, phase size, global step, epoch, and next dataset position. Selection metadata records the complete train and held-out identities and ordered identifiers. RNG metadata covers Python, NumPy, PyTorch CPU, and CUDA state.

The loader parses and validates bounded JSON before it reads safetensors. It then loads tensors on CPU through safetensors, validates the complete manifest and identity, and applies no state until every check passes. The format supports gradient, Eggroll, and alternating modes. Legacy pickle checkpoints and unknown versions are rejected explicitly.

Token and latent result files contain a versioned complete inference identity, exact ordered test identifiers, and per-item `{item_id, prediction, target, correct}` records. Latent identity also contains the checkpoint SHA-256 and exactly seeds `[0, 1, 2, 3, 4]`. Readers validate schema and coverage and recompute aggregates. Writers replace results atomically. `run_gate` uses a new Calc-MAWPS/Qwen results location and never interprets old identity-free JSON as current.

Alternative considered: restricted pickle loading or inferred legacy identity. Pickle still constructs an object graph before application schema validation. Filename and tensor-shape inference cannot prove prompt, scorer, dataset revision, or item selection.

### 8. Verification separates code checks from the costly experiment

Unit tests bind each specification scenario to injected records, fake tokenizers, and tiny fake causal models. A bounded integration smoke test loads the pinned Calc-MAWPS revision and Qwen model, formats one problem, performs one frozen forward/generation path, and verifies the 896-dimensional contract. The full 520-item token preflight is the first experiment command after implementation, not part of the unit suite.

The project has no configured linter or formatter. Verification uses `.\.venv\Scripts\python.exe -m pytest`, targeted tests first, followed by the complete suite when disk space permits temporary checkpoint files.

## Risks / Trade-offs

- [Qwen generation still scores below 5% on filtered Calc-MAWPS] -> Run the full 520-item frozen baseline before any multi-epoch latent training. The gate remains invalid rather than lowering the threshold.
- [Two gigabytes of free disk is consumed by model cache and test artifacts] -> Pin one model revision, avoid duplicate full-model checkpoints, and check free space before the integration smoke test.
- [Float and fraction text has ambiguous extraction] -> Use the exact delimiter and fallback grammar, cap text and digit lengths, and test signs, commas, fractions, multiple numbers, tolerance boundaries, malformed values, and non-finite inputs.
- [Changing slot width breaks every old state tensor] -> Reject legacy artifacts explicitly and assert all model and workspace dimensions at construction.
- [Official chat formatting diverges between baseline and latent context] -> Centralize the exact message and template arguments and test the tokenized context used by each path.
- [Untrusted checkpoint loading executes code or exhausts memory] -> Reject pickle entirely, bound both ZIP members, validate JSON before safetensors, load tensors on CPU, and apply no state before every compatibility check passes.
- [A deterministic CUDA run changes across environments] -> Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization and include it in runtime identity.
- [Full test selection is accidentally shuffled or truncated differently] -> Persist one ordered identifier list and require exact equality before comparison.
- [The existing dirty answer-objective edits are overwritten] -> Treat them as the starting implementation and make narrow patches around their current behavior.

## Migration Plan

1. Add the identity, prompt, dataset-record, and numerical-target contracts without changing runtime defaults.
2. Add Calc-MAWPS loading and generator tests against injected rows.
3. Make workspace and backbone code support the pinned 896-dimensional Qwen contract, then switch Stage 0 defaults.
4. Route all trainers and held-out evaluation through the shared contracts.
5. Add the version-2 JSON plus safetensors checkpoint container and explicit legacy rejection.
6. Route token, latent, and report paths through one ordered test selection and compatibility check.
7. Run targeted and full tests, then run the one-problem pinned-model smoke test.
8. Run the full 520-item token baseline before starting new latent training.

Rollback consists of reverting this change's code and selecting an old checkout for legacy Pythia/GSM8K experiments. New checkpoints are not backward-compatible and are not rewritten during rollback.

## Phase 1 review

**Verdict:** REVISE after round 3. The interview workflow's three-round review cap is exhausted, so implementation requires an explicit user override or a new planning pass.

- Security: 2 findings. Restricted pickle loading can still expand an unbounded primitive object graph before schema validation; bounded JSON parsing also needs construction-time limits.
- Assumptions: 6 findings. Remaining gaps include ASCII regex semantics, omitted-count identity encoding, context-row slicing, binary-float conversion, and manifest digest semantics.
- Testability: 6 findings. Selection, result identity, nested checkpoint schemas, and context-row slicing still lack complete independent fixtures or schemas.
- Consistency: 5 findings. Teacher-forcing lengths conflict, latent context rows are not sliced, latent item shape is incomplete, and the exact seed scenario still says "at least five".
- Implementability: 5 findings. The regex is double-escaped, deterministic CUDA lacks `CUBLAS_WORKSPACE_CONFIG`, teacher-forcing lengths conflict, the latent hidden slice is absent, and prompt token-id identity is ambiguous.

### Resolution after artifact update

The user selected a planning revision instead of an implementation override. The revised delta specs and this design address the recorded findings as follows:

- The numeric grammar is an exact ASCII regex, and non-ASCII digits are unsupported.
- Omitted selection counts normalize to usable split cardinality before identity hashing.
- `result_float` validation uses `Decimal(str(result_float))`.
- Artifact manifests distinguish Git blob object identifiers from raw SHA-256 digests.
- The latent loop validates the complete hidden-state shape and slices only the final slot rows.
- Teacher-forcing inputs and labels now have one exact, equal-length layout.
- Token and latent results share the same per-item schema and bind identity to rendered input token ids.
- Deterministic CUDA requires and records `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Version-2 checkpoints use bounded JSON plus safetensors and reject pickle.
- The latent gate requires exactly seeds `[0, 1, 2, 3, 4]`.

This section records how the findings were incorporated. It does not replace the historical `REVISE` verdict or claim a new adversarial review.
