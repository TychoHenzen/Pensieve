# core/latent-loop Specification

## Purpose
Implements COCONUT-style latent reasoning by feeding Pythia-160M's last hidden state back as the next input for a configurable number of steps, without decoding intermediate tokens.
## Requirements
### Requirement: Latent loop feeds hidden state back as input

The latent loop MUST take the base model's (Pythia-160M) last hidden state from one forward pass and feed it back as the input embedding for the next forward pass. No token decoding occurs between latent steps.

#### Scenario: hidden state feedback

- **WHEN** the latent loop runs one step
- **THEN** the output hidden state of that step becomes the input embedding of the next step, with no argmax or softmax decoding in between

#### Scenario: no intermediate tokens

- **WHEN** the latent loop completes N steps
- **THEN** no tokens are produced or consumed during those N steps

### Requirement: Latent step count is configurable

The number of latent reasoning steps per `observe`/`answer` cycle MUST be configurable. The default count MUST be specified in the Stage 0 plan. The count is fixed (not adaptive) in Stage 0.

#### Scenario: configured step count honored

- **WHEN** the latent loop is configured with N=8 steps
- **THEN** exactly 8 forward passes run per cycle

#### Scenario: different step counts produce different compute

- **WHEN** two runs use N=4 and N=16 respectively
- **THEN** the N=16 run reports approximately 4x the step count in `cost()`

### Requirement: Latent steps map to cost counters

Each latent reasoning step MUST increment the subject's `cost().steps` counter by 1. The `cost().flops` counter MUST reflect the actual floating-point operations of each forward pass through the base model.

#### Scenario: steps counter increments per latent step

- **WHEN** the latent loop runs N steps
- **THEN** `cost().steps` increases by N

#### Scenario: flops counter reflects forward passes

- **WHEN** the latent loop runs N steps
- **THEN** `cost().flops` increases by approximately N times the flops of one base-model forward pass

### Requirement: Latent loop reads and writes workspace slots

The latent loop MUST read from and write to the workspace concept slots during each step. The workspace state after the loop reflects the cumulative effect of all latent steps.

#### Scenario: workspace modified by latent steps

- **WHEN** the latent loop runs on a workspace with initial slot values
- **THEN** the slot values after the loop differ from the initial values

### Requirement: Subject implements the v1 subject protocol

The Stage 0 subject MUST implement all six methods of the v1 subject protocol: `observe`, `answer`, `idle`, `snapshot`, `restore`, `cost`. The subject wraps the latent loop inside `observe` (encode, run latent steps) and `answer` (run latent steps, decode).

#### Scenario: all protocol methods present

- **WHEN** the Stage 0 subject is inspected
- **THEN** it implements `observe`, `answer`, `idle`, `snapshot`, `restore`, and `cost`

#### Scenario: observe runs encoder then latent loop

- **WHEN** `observe(event)` is called
- **THEN** the event is encoded into workspace slots, then the latent loop runs

#### Scenario: answer runs latent loop then decoder

- **WHEN** `answer(probe)` is called
- **THEN** the latent loop runs on the current workspace, then the decoder produces a text response

