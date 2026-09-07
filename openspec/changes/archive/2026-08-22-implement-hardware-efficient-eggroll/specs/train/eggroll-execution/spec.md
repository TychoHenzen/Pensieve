## Purpose

Defines the execution, equivalence, and performance contracts for running EGGROLL candidate evaluation and updates efficiently on one CUDA device.

## ADDED Requirements

### Requirement: Matrix perturbations remain factorized during candidate evaluation

For every trainable matrix `W` with shape `(out_features, in_features)`, EGGROLL MUST reproduce the seeded perturbation `delta = sigma / sqrt(rank) * A @ B.T`, where `B` is drawn from the first `in_features` rows of one `(in_features + out_features, rank)` standard-normal sample and `A` is drawn from the remaining `out_features` rows. Candidate evaluation MUST compute the perturbed projection as the shared base projection plus the candidate-specific low-rank residual. It MUST NOT allocate a tensor with shape `(candidate_batch, out_features, in_features)`. Non-matrix perturbations MUST retain their existing seeded dense-noise definition. Positive and negative members of each antithetic pair MUST use the same sampled factors with opposite signs. This row order MUST match `ESHyperscale/HyperscaleES` commit `b77f7d6f91238fd575313e946b9cad21e0a74b32`.

#### Scenario: Matrix candidate batch uses low-rank residuals

- **WHEN** a batch of matrix-parameter candidates is evaluated
- **THEN** its outputs match direct evaluation of the corresponding materialized perturbed matrices within `rtol=1e-4` and `atol=1e-4`, without allocating a candidate batch of full matrices

#### Scenario: Antithetic factors share one seed

- **WHEN** the positive and negative candidates for one pair are constructed
- **THEN** they use identical sampled factors and sigma scaling with opposite signs

#### Scenario: Vector perturbation remains compatible

- **WHEN** a bias, query, normalization, or temperature parameter is perturbed
- **THEN** its generated noise matches the existing seeded dense perturbation for the same seed and sign

### Requirement: Pseudo-gradient assembly aggregates low-rank factors

EGGROLL MUST assemble each matrix pseudo-gradient from the score-weighted low-rank factors in one batched contraction. It MUST NOT materialize one full matrix delta per candidate or loop over candidates while adding full matrix deltas. It MAY materialize the one final dense pseudo-gradient required by the optimizer for each matrix parameter. The population normalization, fitness normalization, antithetic score difference, and optimizer call order MUST remain unchanged. The tensor supplied to a minimizing optimizer MUST be the negative fitness-gradient estimate, so the resulting parameter update moves toward the higher-fitness antithetic member.

#### Scenario: Factorized matrix update matches reference update

- **WHEN** reference and factorized update assembly receive the same parameters, base seed, population, and candidate fitnesses
- **THEN** every pseudo-gradient, updated parameter, and optimizer tensor matches within `rtol=1e-4` and `atol=1e-4`

#### Scenario: Assembly has one dense matrix result

- **WHEN** a matrix pseudo-gradient is assembled for a complete population
- **THEN** no candidate-sized collection of full matrix deltas exists and at most one full dense gradient exists for that parameter

#### Scenario: Update follows fitness ascent

- **WHEN** the positive antithetic member has higher normalized fitness than the negative member
- **THEN** the optimizer update moves the parameter toward the positive perturbation rather than away from it

### Requirement: Optimized execution preserves EGGROLL training semantics

The optimized and materialized reference paths MUST accept the same prepared example and run configuration. For a fixed initial state and RNG state, they MUST preserve candidate order, candidate seeds, official EGGROLL factor row order, fitness definition, language-model loss, shared variance, normalized fitness, parameter registry order, optimizer state transitions, emitted training fields, and checkpoint contents. Floating-point tensors and scalar metrics MAY differ only within `rtol=1e-4` and `atol=1e-4`. The materialized path MUST be available only to tests and the benchmark command, not as a second production training mode.

#### Scenario: One optimized step matches the reference step

- **WHEN** both paths run one EGGROLL step from identical model, optimizer, and RNG states
- **THEN** their candidate results, step result, updated trainable tensors, optimizer tensors, and final RNG states match under the declared tolerances

#### Scenario: Resume stays deterministic

- **WHEN** an optimized alternating run saves and resumes at a supported boundary
- **THEN** its next EGGROLL step matches an uninterrupted optimized run under the declared tolerances

#### Scenario: Existing commands select the optimized path

- **WHEN** standalone EGGROLL or alternating training starts without a test-only benchmark override
- **THEN** it uses the optimized path while retaining the existing command arguments and defaults

### Requirement: CUDA benchmark proves a material speed improvement

The repository MUST provide a benchmark command that compares the materialized reference and optimized paths in one process on the same CUDA device, prepared example, frozen assets, initial trainable state, population 128, evaluation batch size 8, perturbation rank 4, and float32 execution. It MUST perform at least two unmeasured warm-up steps per path followed by at least five measured steps per path. Each measurement MUST synchronize CUDA before starting and after completing. Measurement order MUST alternate between paths. The command MUST report one JSON object containing the device identity, software identity, asset identity, example identity, full EGGROLL configuration, individual durations, median duration for each path, speedup ratio, and peak allocated CUDA bytes for each path.

#### Scenario: Performance gate passes

- **WHEN** the benchmark completes on CUDA with equivalent path outputs
- **THEN** optimized median duration is at most one third of reference median duration and optimized peak allocated CUDA memory does not exceed the reference peak

#### Scenario: Equivalence failure blocks the benchmark result

- **WHEN** any compared output exceeds the declared numerical tolerance
- **THEN** the command exits nonzero and does not report the performance gate as passed

#### Scenario: CUDA is unavailable

- **WHEN** the benchmark command cannot access a CUDA device
- **THEN** it exits nonzero with a message that states the CUDA benchmark was not run
