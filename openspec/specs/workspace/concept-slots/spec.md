# workspace/concept-slots Specification

## Purpose
Manages a fixed-size set of concept vectors that serve as the persistent workspace for latent reasoning. All subject state lives in these slots.
## Requirements
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

### Requirement: Workspace state round-trips exactly through snapshot and restore

`snapshot()` MUST capture the full workspace state, including all slot values. `restore()` MUST return the workspace to the exact state that `snapshot()` captured. The restored state MUST be bitwise identical to the snapshotted state for all slot vectors.

#### Scenario: snapshot-restore round-trip

- **WHEN** a workspace with modified slot values is snapshotted, then arbitrary operations are performed, then the snapshot is restored
- **THEN** all slot vectors are bitwise identical to their values at snapshot time

#### Scenario: isolation integrity

- **WHEN** the harness runs the probe isolation test (snapshot, feed noise, restore)
- **THEN** the workspace-backed subject produces the same outputs as a subject that never saw the noise

### Requirement: Workspace reports collapse detection fields

The workspace MUST report the variance and covariance of its slot vectors through the instrumentation schema. These fields feed the latent collapse detector that Stage -1 shipped.

#### Scenario: healthy slot distribution

- **WHEN** slots hold diverse vectors with high variance
- **THEN** the collapse detection fields report variance above the configured healthy threshold

#### Scenario: collapsed slot distribution

- **WHEN** all slots converge to the same vector
- **THEN** the collapse detection fields report near-zero variance

### Requirement: Single-vector comparison mode

The workspace MUST support a single-vector mode (slot count 1) so the multi-slot version can be measured against it directly. The single-vector mode MUST use the same interface as the multi-slot mode.

#### Scenario: single slot mode

- **WHEN** a workspace is constructed with slot count 1
- **THEN** it holds one slot and uses the same read/write interface as the multi-slot workspace
