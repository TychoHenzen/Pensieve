## Purpose

Binds every Stage 0 training mode to one filtered Calc-MAWPS data contract and one shared frozen Qwen language-model backbone.

## ADDED Requirements

### Requirement: Shared Stage 0 dataset contract
Standalone gradient, standalone Eggroll, and alternating training MUST load the same complete seed-0 selection of 1,089 normalized Calc-MAWPS training records. Every epoch MUST replay that persisted order without reshuffling, using one example per update. Held-out evaluation MUST use the first 128 identifiers from the persisted seed-0 selection over the 1,039-row decontaminated validation split, selected once and stored in the checkpoint. It MUST snapshot and restore wrapper mode, gradients, EMA state, workspace, optimizers, all RNGs, and schedule state so each is bitwise unchanged by evaluation. Training and phase evaluation MUST NOT load or receive test records.

#### Scenario: Training modes share examples
- **WHEN** each Stage 0 training command uses the same selection arguments
- **THEN** each receives the same ordered normalized training examples and item identifiers

#### Scenario: Held-out split isolation
- **WHEN** phase or epoch evaluation runs during training
- **THEN** it uses only the fixed validation selection and no test item

### Requirement: Shared frozen Qwen backbone
Every Stage 0 training mode MUST load Qwen assets through the gate's pinned safe-loader contract. It MUST also load `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with remote code disabled and Git blob object ids `1_Pooling/config.json=d1514c3162bbe87b343f565fadc62e6c06f04f03`, `config.json=72b987fd805cfa2b58c4c8c952b274a11bfd5a00`, `config_sentence_transformers.json=fd1b291129c607e5d49799f87cb219b27f98acdf`, `modules.json=952a9b81c0bfd99800fabf352f69c7ccd46c5e43`, `sentence_bert_config.json=59d594003bf59880a884c574bf88ef7555bb0202`, `tokenizer.json=cb202bfe2e3c98645018a6d12f182a434c9d3e02`, `tokenizer_config.json=c79f2b6a0cea6f4b564fed1938984bace9d30ff0`, `vocab.txt=fb140275c155a9c7c5a3b3e0e77a9e839594a938`, and raw-file SHA-256 `model.safetensors=53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`. Git blob and raw SHA-256 verification MUST use the algorithms declared by the Qwen safe-loader contract; no unlisted asset may load. The resolved identities and Qwen hidden size 896 MUST match the run identity. Both frozen models MUST remain in evaluation mode with every parameter `requires_grad=False`, absent from optimizer groups, and bitwise unchanged. Allowed trainable parameter paths MUST be exactly `encoder.projection.weight`, `encoder.projection.bias`, `encoder.slot_queries`, `encoder.attn_log_temp`, `latent_loop.projection.weight`, `latent_loop.projection.bias`, `latent_loop.proj_norm.weight`, `latent_loop.proj_norm.bias`, `latent_loop.layer_norm.weight`, and `latent_loop.layer_norm.bias`; EMA copies are buffers, not optimizer parameters. Construction MUST reject and list any other trainable path.

#### Scenario: Backbone parameters remain frozen
- **WHEN** any Stage 0 training step completes
- **THEN** every Qwen parameter remains unchanged and has no optimizer-owned state

#### Scenario: Shared model instance
- **WHEN** an alternating trainer constructs its gradient and Eggroll engines
- **THEN** both engines operate through the same frozen Qwen instance and shared trainable state

### Requirement: Decoder-aligned numerical objective
Canonical targets MUST remove grouping commas and a leading plus sign; integers MUST have no leading zeros; decimals MUST use fixed-point notation with no trailing fractional zeros or decimal point; fractions MUST be reduced with a positive denominator; negative zero and denominator-one fractions MUST serialize as an integer. Examples are `+001 -> 1`, `-0.000 -> 0`, `6.500 -> 6.5`, and `2/4 -> 1/2` after parsing. The tokenizer MUST encode canonical text with `add_special_tokens=False` and no added whitespace. Both the 512-token Qwen prompt limit and the frozen MiniLM tokenizer's 256-token limit MUST be checked without truncation; overlength rows fail before training. Qwen chat context MUST use the exact gate prompt and be embedded as `[chat_context]`; each latent pass MUST consume `[chat_context, slots]` with an all-ones attention mask and consecutive position ids beginning at zero. Decoder generation MUST consume `[slots, generated_answer_prefix]`, use an all-ones mask and consecutive positions, greedily append one token, stop before appending EOS or after 64 tokens, and decode with `skip_special_tokens=True`. For K answer tokens, teacher forcing MUST consume `[slots, ground_truth_answer_token_0, ..., ground_truth_answer_token_K-1]`, a sequence of length `slot_count + K`. Labels MUST have the same length: value `-100` for the first `slot_count - 1` positions followed by `[answer_token_0, ..., answer_token_K-1, eos_token]`. No chain-of-thought token is a training target.

#### Scenario: Multi-token numerical target
- **WHEN** a numerical answer tokenizes into multiple tokens
- **THEN** training supervises them in the same order consumed by autoregressive decoding and then supervises end-of-sequence

#### Scenario: Training prompt contract
- **WHEN** a problem is prepared for a training step
- **THEN** latent context uses the declared Qwen chat prompt while the target contains only the canonical numerical answer

### Requirement: Stage 0 identity accompanies training artifacts
Training initialization seed MUST default to 0, initialize Python, NumPy, PyTorch CPU, and CUDA before trainable module and optimizer construction, and be recorded in identity. Runtime identity MUST also record Python, NumPy, PyTorch, CUDA, Transformers, Datasets, SentenceTransformers, device topology, float32 dtype, eager attention, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`, disabled TF32, and disabled cuDNN benchmarking.

Checkpoints MUST use a version-2 `.ckpt` ZIP container with `ZIP_STORED` compression and exactly two entries: `metadata.json` of at most 1 MiB and `tensors.safetensors` of at most 64 MiB. No pickle content or `torch.load` call is permitted. `metadata.json` MUST reject duplicate keys and non-finite values and contain `{schema_version: 2, identity, mode, tensor_manifest, optimizer_manifests, schedule, selections, metrics, rng}`. Alternating metadata MUST additionally contain `run_config`; standalone metadata MUST NOT contain it. `run_config` MUST contain exactly `{dataset_selection, epochs, phase_steps, model_shape, gradient_optimizer, eggroll_optimizer, eggroll_population, held_out_selection, logging_frequency}` with typed nested fields matching the alternating command. `mode` is one of `gradient`, `eggroll`, or `alternating`. Each tensor-manifest item is `{name, shape, dtype, role}` and MUST describe one unique safetensors entry. Model tensor names MUST be `model.<allowed_trainable_parameter_path>`. Optimizer tensor names MUST be `optimizer.<method>.<parameter_path>.<state_name>`; their metadata MUST record optimizer type, ordered parameter names, group scalars, scalar state, and referenced tensor names. RNG metadata MUST encode Python's version, integer state list, and optional finite Gaussian cache; NumPy's bit-generator name, position, Gaussian flag, finite cache, and referenced state tensor; PyTorch CPU's referenced uint8 tensor; and an ordered list of CUDA device identities and referenced uint8 tensors. `schedule` MUST be empty for standalone modes and, for alternating mode, contain typed fields `active_phase`, `completed_phase_steps`, `phase_steps`, `global_step`, `epoch`, and `next_dataset_position`. `selections` MUST contain typed train and held-out selection identities plus ordered item-id lists. `metrics` MUST contain only bounded strings, finite numbers, booleans, and lists of those values.

Loading MUST first validate ZIP entry names and sizes, then parse and validate all metadata, identities, bounds, tensor names, shapes, dtypes, byte totals, optimizer references, schedule invariants, selections, and RNG topology before reading `tensors.safetensors`. It MUST then use safetensors safe loading and revalidate every tensor against the manifest before model construction, state application, optimizer or RNG restoration, cursor movement, metric writes, or updates. Rejection MUST leave all state unchanged and report every comparable missing, malformed, and unequal canonical field path.

#### Scenario: New checkpoint identity
- **WHEN** a Stage 0 checkpoint is saved
- **THEN** it contains every declared training identity field and the fixed held-out item identifiers

#### Scenario: Legacy artifact rejected
- **WHEN** a GSM8K/Pythia checkpoint or an artifact without complete identity is supplied to a Stage 0 training command
- **THEN** the command rejects it before applying a training update and names the incompatible or missing field
