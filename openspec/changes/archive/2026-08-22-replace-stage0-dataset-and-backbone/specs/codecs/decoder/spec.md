## MODIFIED Requirements

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
