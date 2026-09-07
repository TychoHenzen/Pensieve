## MODIFIED Requirements

### Requirement: Matrix perturbations remain factorized during candidate evaluation

For every EGGROLL-trainable matrix `W` with shape `(out_features, in_features)`, EGGROLL MUST reproduce the seeded perturbation `delta = sigma / sqrt(rank) * A @ B.T`, where `B` is drawn from the first `in_features` rows of one `(in_features + out_features, rank)` standard-normal sample and `A` is drawn from the remaining `out_features` rows. Candidate evaluation MUST compute the perturbed projection as the shared base projection plus the candidate-specific low-rank residual. It MUST NOT allocate a tensor with shape `(candidate_batch, out_features, in_features)`. Positive and negative members of each antithetic pair MUST use the same sampled factors with opposite signs. This row order MUST match `ESHyperscale/HyperscaleES` commit `b77f7d6f91238fd575313e946b9cad21e0a74b32`. Production EGGROLL candidate generation and optimizer groups MUST exclude every non-matrix parameter.

#### Scenario: Matrix candidate batch uses low-rank residuals

- **WHEN** a batch of matrix-parameter candidates is evaluated
- **THEN** its outputs match direct evaluation of the corresponding materialized perturbed matrices within `rtol=1e-4` and `atol=1e-4`, without allocating a candidate batch of full matrices

#### Scenario: Antithetic factors share one seed

- **WHEN** the positive and negative candidates for one pair are constructed
- **THEN** they use identical sampled factors and sigma scaling with opposite signs

#### Scenario: Vector perturbation remains compatible

- **WHEN** EGGROLL evaluates candidates or applies an update
- **THEN** every bias, normalization vector, and scalar temperature retains its base value and has no EGGROLL optimizer state

### Requirement: Pseudo-gradient assembly aggregates low-rank factors

EGGROLL MUST normalize a population of `N` finite fitness values as `z = (fitness - mean(fitness)) / sqrt(population_variance(fitness) + 1e-5)`, where population variance uses divisor `N`. For each antithetic pair, the descent score MUST be `z_negative - z_positive`. Each matrix pseudo-gradient MUST be `sqrt(N) / N` times the sum of its descent-score-weighted positive low-rank deltas. EGGROLL MUST assemble that sum in one batched contraction, MUST NOT materialize one full matrix delta per candidate, and MAY materialize only the final dense pseudo-gradient required for each matrix. The tensor supplied to the minimizing optimizer MUST therefore move an SGD update toward the higher-fitness antithetic member.

#### Scenario: Factorized matrix update matches reference update

- **WHEN** reference and factorized update assembly receive the same parameters, base seed, population, finite fitnesses, sigma, rank, and SGD learning rate
- **THEN** normalized fitnesses, pair scores, pseudo-gradients, updated parameters, and optimizer state match the official formula within `rtol=1e-4` and `atol=1e-4`

#### Scenario: Assembly has one dense matrix result

- **WHEN** a matrix pseudo-gradient is assembled for a complete population
- **THEN** no candidate-sized collection of full matrix deltas exists and at most one full dense gradient exists for that parameter

#### Scenario: Update follows fitness ascent

- **WHEN** the positive antithetic member has higher normalized fitness than the negative member
- **THEN** the SGD update moves the parameter toward the positive perturbation rather than away from it

#### Scenario: Non-finite candidate fitness

- **WHEN** any candidate fitness is NaN or infinite
- **THEN** the update is rejected before assigning a gradient or changing optimizer or parameter state

### Requirement: Optimized execution preserves EGGROLL training semantics

The optimized and materialized reference paths MUST accept the same deterministic fitness batch and run configuration. For a fixed initial state and RNG state, they MUST preserve problem order, per-problem candidate order, candidate seeds, official EGGROLL factor row order, per-problem fitness components, mean candidate fitness, language-model loss, shared variance, normalized fitness, matrix-only parameter registry order, SGD state transitions, emitted training fields, and checkpoint contents. Floating-point tensors and scalar metrics MAY differ only within `rtol=1e-4` and `atol=1e-4`. The materialized path MUST be available only to tests and the benchmark command, not as a second production training mode.

#### Scenario: One optimized step matches the reference step

- **WHEN** both paths run one EGGROLL fitness-batch update from identical model, optimizer, and RNG states
- **THEN** their candidate results, aggregated step result, updated matrix tensors, optimizer state, untouched non-matrix tensors, and final RNG states match under the declared tolerances

#### Scenario: Resume stays deterministic

- **WHEN** an optimized alternating run saves and resumes at a supported boundary
- **THEN** its next EGGROLL fitness-batch update matches an uninterrupted optimized run under the declared tolerances

#### Scenario: Existing commands select the optimized path

- **WHEN** standalone EGGROLL or alternating training starts without a test-only benchmark override
- **THEN** it uses factorized candidates, official fitness normalization, matrix-only momentum-free SGD, and the configured deterministic fitness-batch size

## ADDED Requirements

### Requirement: EGGROLL uses safe Stage 0 defaults

Stage 0 EGGROLL commands MUST default to population `128`, evaluation batch size `8`, perturbation rank `4`, perturbation sigma `0.001`, fitness-batch size `8`, and momentum-free SGD learning rate `0.1`. Gradient commands MUST retain their existing Adam defaults. Every EGGROLL value MUST be recorded in run configuration and validated before model or dataset loading.

#### Scenario: Default stabilized configuration

- **WHEN** a Stage 0 EGGROLL command runs without population, rank, sigma, fitness-batch, or EGGROLL learning-rate overrides
- **THEN** it uses the declared stabilized defaults and reports SGD as the EGGROLL optimizer type

#### Scenario: Invalid stabilized configuration

- **WHEN** population is odd or below two, rank is below one, sigma or learning rate is non-finite or non-positive, or fitness-batch size is below two
- **THEN** the command rejects the configuration before model construction, dataset access, checkpoint loading, or progress output
