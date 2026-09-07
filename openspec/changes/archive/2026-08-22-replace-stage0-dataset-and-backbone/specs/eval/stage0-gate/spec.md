## Purpose

Defines a comparable and cache-safe Stage 0 evaluation for frozen token reasoning and trained latent reasoning on filtered Calc-MAWPS.

## ADDED Requirements

### Requirement: Frozen Qwen token baseline
The token baseline MUST use frozen revision `7ae557604adf67be50417f59c2c2f167def9a775` of `Qwen/Qwen2.5-0.5B-Instruct` for model, tokenizer, model config, chat template, and generation config. Its asset manifest MUST contain Git blob object ids `config.json=0dbb161213629a23f0fc00ef286e6b1e366d180f`, `generation_config.json=dfc11073787daf1b0f9c0f1499487ab5f4c93738`, `merges.txt=20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0`, `tokenizer.json=443909a61d429dff23010e5bddd28ff530edda00`, `tokenizer_config.json=07bfe0640cb5a0037f9322287fbfc682806cf672`, `vocab.json=4783fe10ac3adce15ac8f358ef5462739852c569`, and raw-file SHA-256 `model.safetensors=fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`. A Git blob id MUST recompute as SHA-1 over `b"blob " + ascii(len(content)) + b"\\0" + content`; a raw SHA-256 MUST recompute directly over file bytes. No unlisted model asset or executable code may load. Before any CUDA initialization, the runtime MUST require `CUBLAS_WORKSPACE_CONFIG=:4096:8`; a missing or different value is incompatible. The loader MUST disable remote code, require safetensors, verify the resolved revision and manifest, load float32 with eager attention on one declared device, keep the model in evaluation mode, enable deterministic PyTorch algorithms, and disable TF32 and cuDNN benchmarking. The sole message MUST have role `user` and content `Solve the following math word problem. Show your reasoning, then end your response with exactly "#### <answer>", where <answer> is a finite integer, decimal, or fraction.\n\nProblem:\n{question}`. It MUST call `apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=False, truncation=False)`. A rendered input over 512 token ids MUST be rejected, never truncated. Generation MUST use batch size 1, `do_sample=False`, `num_beams=1`, `max_new_tokens=64`, `use_cache=True`, the pinned EOS token as EOS and padding, and stop at EOS or 64 new tokens. The baseline MUST evaluate the persisted seed-0 selection containing all 520 test identifiers.

#### Scenario: Full token baseline
- **WHEN** the Stage 0 token baseline runs without a development limit
- **THEN** it generates one answer for each of the same 520 ordered test item identifiers

#### Scenario: Official prompt format
- **WHEN** a test problem is prepared for token generation
- **THEN** the input uses Qwen's official chat template with a step-by-step math instruction

### Requirement: Numerical answer equivalence
The scorer MUST accept output text of at most 4,096 Unicode code points and a selected candidate containing at most 256 ASCII digits total. It MUST compile the bounded Python regular expression `(?<![A-Za-z0-9_.,])[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)/(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)|(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:[.][0-9]+)?)(?![A-Za-z0-9_.,/%])` exactly as printed and never use general expression evaluation. Thus `.5`, `5.`, whitespace around `/`, signed denominators, invalid grouping, percentages, scientific notation, and non-ASCII digits are unsupported. If `####` occurs, the candidate MUST be the first regex match after the final delimiter. Otherwise it MUST be the final match in the output. Denominator zero, missing candidates, excessive length, NaN, and infinity MUST be incorrect. Parsed values MUST reduce to exact rational values. Equality MUST pass when rationals are equal or when `abs(a-b) <= max(1e-6, 1e-6 * max(abs(a), abs(b)))`, inclusively.

#### Scenario: Equivalent fraction and decimal
- **WHEN** the prediction is `1/2` and the target is `0.500`
- **THEN** the scorer marks the answer correct

#### Scenario: Repeating decimal tolerance
- **WHEN** a prediction differs from its finite target by exactly or less than the inclusive declared formula, including near zero
- **THEN** the scorer marks the answer correct

#### Scenario: Non-finite answer
- **WHEN** a prediction is over limit, resolves to an unsupported or non-finite form, has a zero denominator, or contains no numerical answer
- **THEN** the scorer marks the answer incorrect without raising an unhandled exception

### Requirement: Comparable test selection
The token and latent paths MUST evaluate the same persisted ordered selection from the pinned test split. The full gate MUST use the complete seed-0 selection of 520 identifiers. A development limit MUST use the first N identifiers from that persisted selection, where N is an integer from 1 through 520, and reuse it in both paths. Limited results MUST be marked development-only and MUST NOT contain a passing gate decision. Training and held-out validation identifiers MUST be disjoint from the test selection by split.

#### Scenario: Full evaluation identity
- **WHEN** token and latent evaluation run on the full test split
- **THEN** both result records contain the same ordered 520 item identifiers

#### Scenario: Limited evaluation identity
- **WHEN** a development run limits the problem count
- **THEN** both paths consume the same persisted ordered subset

### Requirement: Result compatibility identity
Default files MUST be `gate_results/calc_mawps_qwen/token_cot.json`, `latent_eval.json`, and `gate_report.json`. Each file MUST be at most 16 MiB before parsing and use schema version 2 with maximum nesting depth 8, at most 3,000 collection items, duplicate JSON keys and non-finite JSON numbers rejected, and strings subject to their field limits. Token schema MUST contain `{schema_version, identity, items, correct, total}` where every item is `{item_id, prediction, target, correct}`. Latent schema MUST contain `{schema_version, identity, checkpoint_sha256, seeds, runs}` where every run is `{seed, items, correct, total}` and every latent item has the same `{item_id, prediction, target, correct}` schema as a token item. The identity MUST contain the complete dataset selection identity, model and tokenizer asset identities, exact prompt, `rendered_inputs_sha256`, scoring grammar/version, full token and latent generation settings, Python and dependency versions, device, float32 dtype, eager attention implementation, `CUBLAS_WORKSPACE_CONFIG`, deterministic-backend settings, slot count, latent-step count, and tap tuple index. `rendered_inputs_sha256` MUST hash canonical JSON over the ordered list of `{item_id, input_ids}` objects, where `input_ids` is the full ordered integer token-id list for that item. A cached result MUST be reused only when its typed identity matches the request, ids are unique and complete, and every stored `correct`, count, and aggregate recomputes from stored prediction text plus canonical target reloaded from the validated test selection. Latent identity MUST also match the checkpoint SHA-256 and exact seeds. Writers MUST fsync a temporary file in the destination directory and atomically replace the destination only after complete serialization; interrupted writes MUST leave the prior file unchanged. Gate commands MUST accept only local result paths.

#### Scenario: Matching cached baseline
- **WHEN** an existing token result contains identity fields equal to the requested gate configuration
- **THEN** the gate recomputes its aggregates and may reuse it only when schema, identity, and per-item coverage remain valid

#### Scenario: Stale cached baseline
- **WHEN** any dataset, backbone, prompt, scorer, or item identity field differs or is absent
- **THEN** the gate rejects the cached result and does not compare it with the requested latent run

### Requirement: Stage 0 pass condition
The gate MUST define token accuracy as the exact fraction `correct/520` and require at least 26 correct answers. A valid latent result MUST contain exactly evaluation seeds `[0, 1, 2, 3, 4]`; before constructing a fresh subject, each seed MUST initialize Python, NumPy, PyTorch CPU, and all CUDA RNGs. The same persisted item order and deterministic generation settings remain fixed, so zero spread is a valid reproducibility result and the seeds MUST NOT resample items. Each seed accuracy is the exact fraction `correct/520`; the latent mean is the unrounded arithmetic mean of those five fractions. A failed or incomplete seed invalidates the gate. The gate MUST pass only when that exact mean meets or exceeds exact token accuracy.

#### Scenario: Baseline below floor
- **WHEN** fewer than 26 of all 520 token answers are correct
- **THEN** the gate reports an invalid baseline and cannot pass

#### Scenario: Too few latent seeds
- **WHEN** the exact five default latent seed results are not all complete
- **THEN** the gate cannot pass

#### Scenario: Latent result meets baseline
- **WHEN** the token baseline is valid and exactly seeds `[0, 1, 2, 3, 4]` have a mean accuracy equal to or above it
- **THEN** the Stage 0 accuracy criterion passes
