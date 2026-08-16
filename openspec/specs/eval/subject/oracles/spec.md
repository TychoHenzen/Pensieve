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

#### Scenario: retains across many observations

- **GIVEN** a PerfectMemoryOracle that has observed 100 key-value pairs
- **WHEN** probed for the first key observed
- **THEN** the answer is correct

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

#### Scenario: most recent key always answerable

- **GIVEN** observations of keys A, B, C in sequence
- **WHEN** probed for key C
- **THEN** the answer is correct

### Requirement: ChanceOracle answers randomly from VOCAB

ChanceOracle MUST draw answers uniformly at random from VOCAB. Over a large sample, accuracy MUST be statistically consistent with `1/len(VOCAB)`. [REQUIRED]

#### Scenario: accuracy near chance rate

- **GIVEN** 2000 probes with known truth
- **WHEN** ChanceOracle answers each
- **THEN** accuracy is within 3 standard deviations of `1/len(VOCAB)`

#### Scenario: answers are from VOCAB

- **GIVEN** 100 probes
- **WHEN** ChanceOracle answers each
- **THEN** every answer is a member of `VOCAB`

#### Scenario: answers vary across probes

- **GIVEN** 100 probes
- **WHEN** ChanceOracle answers each
- **THEN** at least 2 distinct answers appear

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

#### Scenario: new observations after boundary are retained

- **GIVEN** observations before and after a `TASK_SWITCH` boundary
- **WHEN** probed for keys observed after the boundary
- **THEN** every answer is correct

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

#### Scenario: no truth map entry returns empty

- **GIVEN** a CheaterOracle with a truth map that does not contain key X
- **WHEN** probed for key X
- **THEN** the answer is ""
