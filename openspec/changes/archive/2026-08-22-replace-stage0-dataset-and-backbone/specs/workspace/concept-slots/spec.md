## MODIFIED Requirements

### Requirement: Workspace holds a configurable number of concept slots

The workspace MUST hold a set of concept slots, where each slot is a dense vector of the base model's hidden dimension, 896 for Qwen2.5-0.5B-Instruct. The default slot count MUST be 16. The slot count MUST be configurable to support ablation over {1, 4, 8, 16, 32, 64}.

#### Scenario: default slot count

- **WHEN** a workspace is constructed with no explicit slot count
- **THEN** it contains 16 slots, each of dimension 896

#### Scenario: configurable slot count

- **WHEN** a workspace is constructed with slot count 8
- **THEN** it contains 8 slots, each of dimension 896

#### Scenario: ablation sweep values accepted

- **WHEN** a workspace is constructed with slot count set to any of {1, 4, 8, 16, 32, 64}
- **THEN** construction succeeds and the workspace holds that many slots
