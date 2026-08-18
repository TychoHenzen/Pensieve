## Purpose

Maps workspace concept slots back to text, serving as the output edge of the latent reasoning system. Invoked by `answer()` to produce probe responses.

## ADDED Requirements

### Requirement: Decoder maps workspace slots to text

The decoder MUST accept the current workspace slot vectors and produce a text string. The input dimension MUST match the workspace slot dimension (768).

#### Scenario: slots decoded to text

- **WHEN** the decoder receives workspace slot vectors
- **THEN** it produces a text string

#### Scenario: different slot states produce different text

- **WHEN** the decoder is called with two distinct workspace states
- **THEN** the two outputs differ (the decoder is sensitive to slot content)

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
