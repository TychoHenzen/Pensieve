# codecs/encoder Specification

## Purpose
Maps variable-length text input into a fixed number of workspace concept slots, serving as the input edge of the latent reasoning system.
## Requirements
### Requirement: Encoder maps text to workspace slots

The encoder MUST accept a text string and produce a set of vectors that fill the workspace slots. The output dimension of each vector MUST match the workspace slot dimension (768).

#### Scenario: text encoded to slots

- **WHEN** the encoder receives a text input
- **THEN** it produces exactly as many vectors as there are workspace slots, each of dimension 768

#### Scenario: variable-length input accepted

- **WHEN** the encoder receives inputs of different lengths (one word vs. a paragraph)
- **THEN** both produce the same number of output vectors (one per slot)

### Requirement: Encoder is frozen at first

The encoder weights MUST be frozen during initial Stage 0 experiments. The encoder MUST use a pretrained sentence-embedding model. Freezing means no gradient flows back through the encoder during training.

#### Scenario: frozen encoder weights

- **WHEN** a training step runs with the encoder in the forward pass
- **THEN** the encoder's parameters have `requires_grad=False` and receive no gradient updates

#### Scenario: pretrained weights loaded

- **WHEN** the encoder is constructed
- **THEN** it loads weights from a pretrained sentence-embedding model, not random initialization

### Requirement: Encoder integrates with observe

The subject's `observe()` method MUST use the encoder to write input into workspace slots. After `observe()` returns, the workspace slots reflect the encoded input.

#### Scenario: observe writes encoded input to workspace

- **WHEN** `observe(event)` is called with a text-bearing event
- **THEN** the workspace slots contain the encoder's output for that event's rendered text

