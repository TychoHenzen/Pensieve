# Reproduction gate: pass condition

Written before the first gate run, per the "Pass condition written before
first run" requirement in `specs/eval/reproduction-gate/spec.md`. This file
is the sole source of truth for gate pass/fail. A gate run that has not been
checked against every criterion below has not been evaluated.

## Reference numbers

Source: van de Ven, G.M. and Tolias, A.S. (2019), "Three scenarios for
continual learning," arXiv:1904.07734, Table 4, class-incremental
Split-MNIST. See `investigation-phase7.md` for the full derivation and the
architecture/hyperparameters used to reproduce them (2x400 ReLU MLP, Adam
lr=0.001, 2000 iterations/task, batch 128, 5 tasks; EWC lambda=1e7, Fisher
samples=1000; replay: symmetric VAE, latent dim 100).

| Method  | Paper mean accuracy | Paper spread |
|---------|---------------------|--------------|
| Naive   | 19.90%              | +/- 0.02     |
| EWC     | 20.01%              | +/- 0.06     |
| Replay  | 90.79%              | +/- 0.41     |
| Chance  | 10.00%              | (10 classes) |

## Criterion 1: per-method tolerance

Each method's gate-run mean accuracy (across all seeds, see Criterion 4)
MUST fall inside the stated band. Bands are wider than the paper's own
seed spread because this is an independent reimplementation, not a rerun
of the released code.

| Method  | Paper mean | Tolerance | Pass band       |
|---------|------------|-----------|------------------|
| Naive   | 19.90%     | +/- 5 pp  | 14.90% - 24.90%  |
| EWC     | 20.01%     | +/- 5 pp  | 15.01% - 25.01%  |
| Replay  | 90.79%     | +/- 10 pp | 80.79% - 100.00% |

A method whose mean accuracy falls outside its pass band fails the gate,
regardless of the other criteria.

## Criterion 2: near-chance / high-accuracy bands

Independent of the paper-relative tolerance above, each method's mean
accuracy MUST also fall in its qualitative band:

- Naive: below 25.00%
- EWC: below 25.00%
- Replay: above 80.00%

This restates the intent of Criterion 1 as an explicit, simpler check
(near chance vs. high) so a gate result cannot pass on a technicality
where the paper-relative band and the qualitative band disagree.

## Criterion 3: ordering

The three methods' mean accuracies MUST satisfy both:

- `replay_mean > ewc_mean`
- `replay_mean > naive_mean`

matching the paper's ordering (replay far ahead of naive and EWC, which
sit near each other close to chance). No minimum margin is required beyond
the strict inequality, but the pass bands in Criterion 1 already force a
gap of at least `80.79% - 25.01%` = 55.78 percentage points between replay
and either of the other two, so any borderline ordering failure implies a
Criterion 1 failure as well.

## Criterion 4: seed count and reporting

Each method MUST be run for at least 5 seeds. The gate report MUST include,
per method: the list of seeds used, the per-seed accuracy, the mean, and
the standard deviation. A run with fewer than 5 seeds is an anecdote and
MUST NOT be used to pass or fail the gate. The "mean accuracy" used in
Criteria 1-3 is the mean across these seeds.

## Criterion 5: reproducibility

Two gate runs with identical config (method, architecture, hyperparameters,
task order) and identical seed MUST produce identical probe logs. The
harness MUST record whether deterministic CUDA kernels were active for the
run; if determinism was not enabled, the run MUST be labeled
non-deterministic and MUST NOT be used to satisfy this criterion. Only runs
executed with deterministic mode active may be compared for exact-repeat
equality.

## Overall pass/fail

The gate PASSES only if all of the following hold:

1. All three methods (naive, EWC, replay) satisfy Criterion 1 (per-method
   tolerance).
2. All three methods satisfy Criterion 2 (qualitative band).
3. The ordering in Criterion 3 holds.
4. Each method used at least 5 seeds with mean and std reported
   (Criterion 4).
5. The reproducibility check in Criterion 5 passes for at least one
   deterministic-mode config/seed pair per method.

Failure of any single criterion, for any single method, fails the gate. On
failure, the diagnostic split requirement in the spec (running the paper's
released code as a subject through the same harness) applies to separate
instrument bugs from method-rewrite bugs.
