## Purpose

Defines five reference oracle implementations used as baselines and test fixtures: PerfectMemory (lossless recall), Forgetful (capacity-1), Chance (random), TaskWiper (boundary-aware wipe), and Cheater (probe-exploiting adversary).

## Requirements

### Requirement: PerfectMemoryOracle recalls all observations

PerfectMemoryOracle MUST return the correct value for every key it has observed. For keys it has not observed, it MUST return "". [REQUIRED]

#### Scenario: recalls everything

- **GIVEN** a PerfectMemoryOracle that has observed multiple key-value pairs
- **WHEN** probed for each key
- **THEN** every answer matches the observed value

#### Scenario: unknown key returns empty

- **GIVEN** a PerfectMemoryOracle
- **WHEN** probed for a key that was never observed
- **THEN** the answer is ""

### Requirement: ForgetfulOracle retains only the most recent observation

ForgetfulOracle MUST answer correctly for the most recently observed key-value pair. For all other keys, it MUST return "". [REQUIRED]

#### Scenario: immediate recall correct

- **GIVEN** a single observation followed immediately by a probe
- **WHEN** probed for that key
- **THEN** the answer is correct

#### Scenario: earlier keys forgotten after intervening observation

- **GIVEN** observations of keys A then B
- **WHEN** probed for key A
- **THEN** the answer is ""

### Requirement: ChanceOracle answers randomly from VOCAB

ChanceOracle MUST draw answers uniformly at random from VOCAB. Over a large sample, accuracy MUST be statistically consistent with `1/len(VOCAB)`. [REQUIRED]

#### Scenario: accuracy near chance rate

- **GIVEN** 2000 probes with known truth
- **WHEN** ChanceOracle answers each
- **THEN** accuracy is within 3 standard deviations of `1/len(VOCAB)`

### Requirement: TaskWiperOracle forgets on task boundaries

TaskWiperOracle MUST recall all observations within the current task. After a `TASK_SWITCH` boundary, it MUST forget all prior observations. [REQUIRED]

#### Scenario: current task correct

- **GIVEN** observations within one task
- **WHEN** probed for those keys
- **THEN** every answer is correct

#### Scenario: drops on boundary

- **GIVEN** observations before a `TASK_SWITCH` boundary
- **WHEN** probed for those keys after the boundary
- **THEN** every answer is ""

### Requirement: CheaterOracle exploits probe truth map

CheaterOracle MUST return "" when no truth is available for a key. When a truth map entry exists, it MUST learn the answer from probe side effects and return it on subsequent probes. [REQUIRED]

#### Scenario: unknown without truth map

- **GIVEN** a CheaterOracle with an empty truth map
- **WHEN** probed for any key
- **THEN** the answer is ""

#### Scenario: learns from probes

- **GIVEN** a CheaterOracle with truth map containing a key
- **WHEN** probed twice for that key (without isolation)
- **THEN** first answer is "", second answer is the truth value
