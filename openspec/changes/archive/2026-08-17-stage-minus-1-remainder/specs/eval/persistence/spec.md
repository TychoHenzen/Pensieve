## Purpose

Defines checkpoint-resume, crash recovery, and run-record management for harness runs that last hours or days.

## ADDED Requirements

### Requirement: Run record directory

Each run MUST produce a single directory containing the resolved config, the code revision, the stream hash, the environment snapshot, the probe log, the metric summary, and any checkpoints. [REQUIRED]

#### Scenario: directory contents after a completed run

- **GIVEN** a completed run
- **WHEN** the run directory is listed
- **THEN** it contains files for config, revision, stream hash, environment, probe log, and metric summary

### Requirement: Config resolution and hashing

The config MUST be resolved from a file plus overrides, and the resolved config MUST be hashed into the run id. No values MUST be read from the environment at runtime except filesystem paths. [REQUIRED]

#### Scenario: same config produces same run id

- **GIVEN** two runs with identical config
- **WHEN** the run ids are compared
- **THEN** they are equal

### Requirement: Checkpoint by position and by wall clock

Checkpoints MUST be writable on two schedules: every N stream positions and every M wall-clock seconds. Both schedules are configurable. [REQUIRED]

#### Scenario: checkpoint at configured interval

- **GIVEN** a checkpoint interval of 100 events
- **WHEN** 250 events have been processed
- **THEN** checkpoints exist at positions 100 and 200

### Requirement: Atomic checkpoint writes

A checkpoint MUST be written atomically: write to a temporary file, then rename. A crash during a write MUST NOT corrupt the most recent complete checkpoint. [REQUIRED]

#### Scenario: incomplete checkpoint does not overwrite last good

- **GIVEN** a checkpoint at position 100 exists
- **WHEN** a write at position 200 is interrupted (simulated)
- **THEN** the checkpoint at position 100 is still intact and loadable

### Requirement: Resume produces same probe log as unbroken run

A run that is killed and resumed MUST produce the same probe log as an unbroken run with the same config and seed. Resume MUST continue the same stream at the same position with the same RNG state. [REQUIRED]

#### Scenario: resumed run matches unbroken run

- **GIVEN** an unbroken run producing probe log L
- **WHEN** the same run is killed at position 50 and resumed from checkpoint
- **THEN** the resumed run's probe log equals L

### Requirement: Checkpoint retention policy

The config MUST specify a retention policy for checkpoints. The default is to keep the last checkpoint plus every Nth. Older checkpoints outside the policy MUST be deleted. [REQUIRED]

#### Scenario: default retention keeps last plus every Nth

- **GIVEN** a retention policy of `keep_every=10` and a run of 55 events with checkpoints every 5
- **WHEN** the run completes
- **THEN** checkpoints at positions 10, 20, 30, 40, 50, and 55 are retained; others are deleted

### Requirement: Seeded reproducibility

Seeded generators, seeded model init, and seeded data order MUST produce identical results across runs. The harness MUST record whether deterministic CUDA kernels were active. A number from the non-deterministic mode MUST NOT be used to pass a gate. [REQUIRED]

#### Scenario: two runs with same seed produce same probe log

- **GIVEN** two runs with the same config and seed, both using deterministic kernels
- **WHEN** their probe logs are compared
- **THEN** they are identical
