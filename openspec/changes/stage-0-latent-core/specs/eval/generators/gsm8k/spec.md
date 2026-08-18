## Purpose

Wraps the GSM8K dataset (grade-school math word problems) as a stream generator, providing the evaluation surface for the Stage 0 gate.

## ADDED Requirements

### Requirement: GSM8K generator yields a stream of math word problems

The generator MUST yield a deterministic, seeded stream of `Observe` and `Probe` events drawn from the GSM8K dataset. Each `Observe` event presents a word problem. Each `Probe` event asks for a numerical answer.

#### Scenario: deterministic stream

- **WHEN** the generator runs twice with the same config and seed
- **THEN** it produces the same sequence of events

#### Scenario: observe events contain word problems

- **WHEN** an `Observe` event is rendered
- **THEN** it contains a natural-language math word problem from GSM8K

#### Scenario: probe events ask for numerical answers

- **WHEN** a `Probe` event is rendered
- **THEN** it asks for a numerical answer to a previously observed word problem

### Requirement: GSM8K generator reports a chance rate

The generator MUST report a computable chance rate for its probe class, consistent with the v1 generator protocol. The chance rate for free-form numerical answers is effectively zero (1 over the answer space).

#### Scenario: chance rate reported

- **WHEN** the generator's chance rate is queried
- **THEN** it returns a value consistent with the answer space size

### Requirement: GSM8K generator follows the v1 stream schema

The generator MUST produce `StreamItem`s that conform to the v1 stream schema. Truth is carried on the side channel. Rendering MUST NOT leak the answer, probe id, or task id into the subject's view.

#### Scenario: truth on side channel only

- **WHEN** a `Probe` event is rendered for the subject
- **THEN** the rendered text contains no answer, probe id, or task id

#### Scenario: stream items are valid

- **WHEN** the generator's output is validated against the v1 stream schema
- **THEN** all items pass validation

### Requirement: GSM8K generator supports a configurable subset

The generator MUST support running on a subset of GSM8K (configurable number of problems) to enable fast iteration during development. The default MUST use the full dataset.

#### Scenario: subset mode

- **WHEN** the generator is configured with a problem count of 100
- **THEN** it yields a stream derived from 100 GSM8K problems

#### Scenario: full dataset default

- **WHEN** the generator is configured with default settings
- **THEN** it uses all available GSM8K problems
