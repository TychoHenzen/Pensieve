## MODIFIED Requirements

### Requirement: Latent loop feeds hidden state back as input

The latent loop MUST run the shared frozen Qwen2.5-0.5B-Instruct backbone with `output_hidden_states=True` and take `outputs.hidden_states[12]`, the output after zero-based decoder block 11 and the human-numbered layer 12. The input layout MUST be one unpadded batch containing `[chat_context, slots]`; after validating shape `(1, context_length + slot_count, 896)`, it MUST define `h = outputs.hidden_states[12][0, -slot_count:, :]`. For tensors `s,h` of shape `(slot_count,896)`, the trainable projection has weight shape `(896,896)` and bias `(896)`, and the update MUST be `u = LayerNorm_projection(W h + b)` followed by `s_next = LayerNorm_output(u + 0.5 * s)`. Both LayerNorm modules MUST operate independently on each slot's last axis, use epsilon `1e-5`, retain trainable affine parameters, initialize each weight to `0.7 / sqrt(896)`, and initialize each bias to zero. The next forward pass MUST receive `s_next` directly as its slot input embeddings. No token decoding occurs between latent steps.

#### Scenario: hidden state feedback

- **WHEN** the latent loop runs one step
- **THEN** `outputs.hidden_states[12]` is transformed by the declared projection, residual, and normalization equation into the next step's slot embeddings

#### Scenario: no intermediate tokens

- **WHEN** the latent loop completes N steps
- **THEN** no tokens are produced or consumed during those N steps

#### Scenario: selected tap layer unavailable

- **WHEN** the loaded backbone does not expose hidden layer 12 or a hidden width of 896
- **THEN** latent-loop construction fails with an error that names the expected and actual model shape

## ADDED Requirements

### Requirement: Model-neutral token embedding access
The latent loop MUST obtain context token embeddings through the backbone's public input-embedding interface and MUST NOT depend on a Pythia-specific module path.

#### Scenario: Qwen context embedding
- **WHEN** chat-formatted Qwen token identifiers condition a latent run
- **THEN** their embeddings are obtained from Qwen's declared input-embedding layer
