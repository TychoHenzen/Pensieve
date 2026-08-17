## Purpose

Defines the reproduction gate that exits Stage -1: class-incremental Split-MNIST, three methods compared against published numbers, with a multi-seed sweep and a written pass condition.

## ADDED Requirements

### Requirement: Gate target

The gate MUST reproduce class-incremental Split-MNIST results from van de Ven, Siegelmann and Tolias, "Brain-inspired replay for continual learning with artificial neural networks," Nature Communications 11:4069, 2020. The three methods are naive sequential, EWC, and generative replay. [REQUIRED]

#### Scenario: all three methods run on the same stream

- **GIVEN** a class-incremental Split-MNIST stream
- **WHEN** the gate runner executes
- **THEN** naive sequential, EWC, and generative replay all run on identical streams

### Requirement: Pass condition written before first run

The tolerance for each method's accuracy MUST be written down before the first gate run. The pass condition MUST include: each method within a stated tolerance of the paper's number, the ordering of the three methods matches the paper, and two runs with the same config and seed produce the same probe log. [REQUIRED]

#### Scenario: tolerance is declared

- **GIVEN** the gate configuration
- **WHEN** it is inspected before any run
- **THEN** a numerical tolerance per method is specified

### Requirement: Multi-seed sweep

The gate MUST run at least five seeds and report the spread. Single-seed comparisons MUST be labeled as anecdotes and MUST NOT pass the gate. [REQUIRED]

#### Scenario: five seeds

- **GIVEN** a gate run
- **WHEN** it completes
- **THEN** at least five seed results are present with mean and standard deviation

### Requirement: Method ordering matches paper

In the class-incremental setting, naive sequential and EWC MUST land near chance while generative replay MUST stay high. The ordering MUST match the paper. [REQUIRED]

#### Scenario: ordering check

- **GIVEN** gate results
- **WHEN** the three methods' accuracies are compared
- **THEN** replay > EWC and replay > naive, matching the paper's ordering

### Requirement: Diagnostic split for failure

If the gate does not reproduce, the harness MUST support running the paper's own released code as a subject through the same harness. This separates instrument bugs from method-rewrite bugs. [REQUIRED]

#### Scenario: paper code as subject

- **GIVEN** gate failure
- **WHEN** the paper's released code is wrapped as a subject
- **THEN** it runs through the harness and produces a probe log for comparison

### Requirement: Reproducibility check

Two runs with the same config and seed MUST produce the same probe log. The harness MUST record whether deterministic CUDA kernels were active. [REQUIRED]

#### Scenario: exact repeat

- **GIVEN** two gate runs with the same seed and deterministic mode
- **WHEN** their probe logs are compared
- **THEN** they are identical
