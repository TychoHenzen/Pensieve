# core/latent-loop Specification

## Purpose
Implements COCONUT-style latent reasoning by feeding a selected hidden state from the frozen Qwen backbone back as the next input for a configurable number of steps, without decoding intermediate tokens.
## Requirements
### Requirement: Latent loop feeds hidden state back as input

The latent loop MUST use the shared frozen Qwen2.5-0.5B-Instruct backbone and take the state at `outputs.hidden_states[12]`, the output after zero-based decoder block 11 and the human-numbered layer 12. The reference input layout MUST be one unpadded batch containing `[chat_context, slots]`; after validating shape `(batch_size, context_length + slot_count, 896)`, it MUST define `h = outputs.hidden_states[12][:, -slot_count:, :]`. Production execution MAY precompute the causally independent chat-context state, reuse its immutable attention cache, and stop after producing layer 12. Cached execution MUST use the same attention mask and consecutive position identifiers as the reference sequence. During latent-only candidate evaluation it MUST NOT execute later decoder blocks or the language-model vocabulary head. Cached and reference `h` tensors MUST match within `rtol=1e-4` and `atol=1e-4`.

For tensors `s,h` with final dimensions `(slot_count,896)`, the trainable projection has weight shape `(896,896)` and bias `(896)`, and the update MUST be `u = LayerNorm_projection(W h + b)` followed by `s_next = LayerNorm_output(u + 0.5 * s)`. Both LayerNorm modules MUST operate independently on each slot's last axis, use epsilon `1e-5`, retain trainable affine parameters, initialize each weight to `0.7 / sqrt(896)`, and initialize each bias to zero. The next forward pass MUST receive `s_next` directly as its slot input embeddings. No token decoding occurs between latent steps.

#### Scenario: hidden state feedback

- **WHEN** the latent loop runs one step
- **THEN** the selected layer-12 state is transformed by the declared projection, residual, and normalization equation into the next step's slot embeddings

#### Scenario: no intermediate tokens

- **WHEN** the latent loop completes N steps
- **THEN** no tokens are produced or consumed during those N steps

#### Scenario: selected tap layer unavailable

- **WHEN** the loaded backbone does not expose hidden layer 12 or a hidden width of 896
- **THEN** latent-loop construction fails with an error that names the expected and actual model shape

#### Scenario: Cached latent state matches full execution

- **WHEN** cached and reference execution receive the same context embeddings and slot embeddings
- **THEN** their selected layer-12 slot states match within `rtol=1e-4` and `atol=1e-4`

#### Scenario: Cached context is immutable

- **WHEN** multiple candidates and latent steps reuse one prepared chat context
- **THEN** they read one detached context cache and no candidate or step mutates it

#### Scenario: Latent execution omits unused model work

- **WHEN** the optimized path obtains the selected layer-12 slot state
- **THEN** it does not execute decoder blocks after that state or construct vocabulary logits

### Requirement: Model-neutral token embedding access
The latent loop MUST obtain context token embeddings through the backbone's public input-embedding interface and MUST NOT depend on a Pythia-specific module path.

#### Scenario: Qwen context embedding
- **WHEN** chat-formatted Qwen token identifiers condition a latent run
- **THEN** their embeddings are obtained from Qwen's declared input-embedding layer

### Requirement: Latent step count is configurable

The number of latent reasoning steps per `observe`/`answer` cycle MUST be configurable. The default count MUST be specified in the Stage 0 plan. The count is fixed (not adaptive) in Stage 0.

#### Scenario: configured step count honored

- **WHEN** the latent loop is configured with N=8 steps
- **THEN** exactly 8 forward passes run per cycle

#### Scenario: different step counts produce different compute

- **WHEN** two runs use N=4 and N=16 respectively
- **THEN** the N=16 run reports approximately 4x the step count in `cost()`

### Requirement: Latent steps map to cost counters

Each latent reasoning step MUST increment the subject's `cost().steps` counter by 1. The `cost().flops` counter MUST reflect the actual floating-point operations of each forward pass through the base model.

#### Scenario: steps counter increments per latent step

- **WHEN** the latent loop runs N steps
- **THEN** `cost().steps` increases by N

#### Scenario: flops counter reflects forward passes

- **WHEN** the latent loop runs N steps
- **THEN** `cost().flops` increases by approximately N times the flops of one base-model forward pass

### Requirement: Latent loop reads and writes workspace slots

The latent loop MUST read from and write to the workspace concept slots during each step. The workspace state after the loop reflects the cumulative effect of all latent steps.

#### Scenario: workspace modified by latent steps

- **WHEN** the latent loop runs on a workspace with initial slot values
- **THEN** the slot values after the loop differ from the initial values

### Requirement: Subject implements the v1 subject protocol

The Stage 0 subject MUST implement all six methods of the v1 subject protocol: `observe`, `answer`, `idle`, `snapshot`, `restore`, `cost`. The subject wraps the latent loop inside `observe` (encode, run latent steps) and `answer` (run latent steps, decode).

#### Scenario: all protocol methods present

- **WHEN** the Stage 0 subject is inspected
- **THEN** it implements `observe`, `answer`, `idle`, `snapshot`, `restore`, and `cost`

#### Scenario: observe runs encoder then latent loop

- **WHEN** `observe(event)` is called
- **THEN** the event is encoded into workspace slots, then the latent loop runs

#### Scenario: answer runs latent loop then decoder

- **WHEN** `answer(probe)` is called
- **THEN** the latent loop runs on the current workspace, then the decoder produces a text response
