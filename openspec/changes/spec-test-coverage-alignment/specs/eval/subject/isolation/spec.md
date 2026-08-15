## Purpose

Defines the `isolated_answer` wrapper that prevents a subject from retaining side effects of answering a probe, and the CheaterOracle that serves as the canonical adversarial test case for it.

## ADDED Requirements

### Requirement: isolated_answer prevents probe-based learning

`isolated_answer` MUST snapshot the subject before answering and restore it after, so no side effects of the answer call persist. [REQUIRED]

#### Scenario: cheater detected by isolation

- **GIVEN** a CheaterOracle (which learns truth from probe side effects)
- **WHEN** probed twice for the same key, once with and once without `isolated_answer`
- **THEN** the isolated answers differ from the non-isolated answers (isolation prevents learning)

#### Scenario: honest subject unaffected

- **GIVEN** a PerfectMemoryOracle (which does not exploit probes)
- **WHEN** probed with and without `isolated_answer`
- **THEN** the answers are identical

### Requirement: CheaterOracle learns from probes when unprotected

Without isolation, the CheaterOracle MUST learn truth from probe side effects. This behavior exists so isolation can be tested against a known adversary. [REQUIRED]

#### Scenario: learns on second probe

- **GIVEN** a CheaterOracle with truth map containing a key
- **WHEN** probed for that key twice without isolation
- **THEN** the first answer is "" and the second answer is the truth

#### Scenario: cannot learn when isolated

- **GIVEN** a CheaterOracle with truth map containing a key
- **WHEN** probed for that key twice with `isolated_answer`
- **THEN** both answers are "" (state is rolled back after each probe)
