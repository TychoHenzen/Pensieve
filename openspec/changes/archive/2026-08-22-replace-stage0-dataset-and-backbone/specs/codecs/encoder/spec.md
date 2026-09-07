## MODIFIED Requirements

### Requirement: Encoder maps text to workspace slots

The encoder MUST accept a text string and produce a set of vectors that fill the workspace slots. The output dimension of each vector MUST match the 896-dimensional workspace slot size.

#### Scenario: text encoded to slots

- **WHEN** the encoder receives a text input
- **THEN** it produces exactly as many vectors as there are workspace slots, each of dimension 896

#### Scenario: variable-length input accepted

- **WHEN** the encoder receives inputs of different lengths, such as one word and one paragraph
- **THEN** both produce the same number of 896-dimensional output vectors, one per slot
