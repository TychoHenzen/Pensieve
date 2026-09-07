# codecs/decoder Specification

## Purpose
Maps workspace concept slots back to text, serving as the output edge of the latent reasoning system. Invoked by `answer()` to produce probe responses.
## Requirements
### Requirement: Decoder maps workspace slots to text

The decoder MUST accept the current 896-dimensional workspace slot vectors and produce a text string through the shared frozen Qwen backbone. The decoder MUST consume slots directly as model input embeddings without a separate slot-to-model adapter.

#### Scenario: slots decoded to text

- **WHEN** the decoder receives 896-dimensional workspace slot vectors
- **THEN** it produces a text string through the shared frozen Qwen model

#### Scenario: different slot states produce different text

- **WHEN** a deterministic fake backbone receives two fixture workspace states defined to produce different next-token logits
- **THEN** the decoder forwards each supplied state and exposes the corresponding different greedy next token

#### Scenario: incompatible slot width

- **WHEN** the decoder receives slots whose width differs from Qwen's 896-dimensional input embeddings
- **THEN** it rejects the input before generation with an error that names both dimensions

### Requirement: answer invokes the decoder

The subject's `answer(probe)` method MUST use the decoder to produce its response from the current workspace state. The decoder reads workspace slots without modifying them.

#### Scenario: answer returns decoded text

- **WHEN** `answer(probe)` is called
- **THEN** the returned string is the decoder's output given the current workspace slots

#### Scenario: answer does not mutate workspace

- **WHEN** `answer(probe)` is called inside probe isolation
- **THEN** the workspace slots are unchanged after the call (before restore)

### Requirement: Decoder is frozen at first

The decoder weights MUST be frozen during initial Stage 0 experiments. Freezing means no gradient flows back through the decoder during training.

#### Scenario: frozen decoder weights

- **WHEN** a training step runs
- **THEN** the decoder's parameters have `requires_grad=False` and receive no gradient updates
