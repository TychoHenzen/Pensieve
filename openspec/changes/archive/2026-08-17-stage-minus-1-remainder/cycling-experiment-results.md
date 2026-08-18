# Cycling experiment results

Date: 2026-08-17 (overnight run, ~7 hours on CUDA)

Script: `scripts/show_gate.py`

## Question

Does re-exposing models to the same 5-task sequence (curriculum cycling)
improve retention? Each config keeps total examples per task fixed at
2000. The variable is how those 2000 examples split across cycles.
`train_iterations` scales inversely with cycle count, floored at 50.

## Configs

| Name      | examples/cycle | cycles | train_iter/boundary |
|-----------|---------------|--------|---------------------|
| seq       | 2000          | 1      | 2000                |
| 2-cycle   | 1000          | 2      | 1000                |
| 10-cycle  | 200           | 10     | 200                 |
| 50-cycle  | 40            | 50     | 50 (floor)          |
| 200-cycle | 10            | 200    | 50 (floor)          |

3 seeds per config, 4 methods (naive, ewc, replay, joint).

## Results: MNIST

| config    | naive  | ewc   | replay | joint  |
|-----------|--------|-------|--------|--------|
| seq       | 19.5%  | 19.5% | 88.7%  | 95.1%  |
| 2-cycle   | 19.8%  | 19.9% | 92.0%  | 95.6%  |
| 10-cycle  | 25.2%  | 19.9% | 94.1%  | 95.7%  |
| 50-cycle  | 89.7%  | 43.5% | 94.3%  | 96.7%  |
| 200-cycle | 89.1%  | 56.3% | 92.5%  | 96.2%  |

All values are mean accuracy across 3 seeds.

## Results: FashionMNIST

| config    | naive  | ewc   | replay | joint  |
|-----------|--------|-------|--------|--------|
| seq       | 19.8%  | 19.8% | 77.0%  | 84.5%  |
| 2-cycle   | 20.0%  | 22.7% | 76.2%  | 85.2%  |
| 10-cycle  | 22.1%  | 31.1% | 82.6%  | 84.6%  |
| 50-cycle  | 42.4%  | 21.6% | 78.2%  | 86.3%  |
| 200-cycle | 63.4%  | 35.5% | 74.5%  | 84.7%  |

## Findings

### 1. Replay peaks at 10 cycles, then diverges by dataset difficulty

On MNIST, replay rises from 88.7% (seq) to 94.1% (10-cycle), then
plateaus through 200-cycle (92.5%). On FashionMNIST, replay rises from
77.0% to 82.6% at 10-cycle, then drops to 74.5% at 200-cycle. The
harder dataset reveals that heavy cycling degrades the VAE: each cycle
has too few examples for the generator to learn good reconstructions.

The sweet spot on both datasets is 10 cycles. That closes roughly half
the gap between sequential replay and the joint oracle.

### 2. Naive converges toward interleaved training at high cycle counts

At 200 cycles of 10 examples each, naive SGD on MNIST reaches 89.1%,
which matches sequential replay (88.7%). Catastrophic forgetting
requires enough consecutive same-task training to overwrite prior
weights. With only 10 examples per task per cycle, there is not enough
gradient pressure to forget.

FashionMNIST naive reaches 63.4% at 200-cycle. The lower ceiling
reflects harder inter-class boundaries: even with interleaved-like
presentation, the model cannot separate all 10 classes as cleanly.

### 3. EWC is erratic and non-monotonic under cycling

On MNIST, EWC improves from 19.5% (seq) to 56.3% (200-cycle), but with
high variance (3.1% std). On FashionMNIST, EWC peaks at 31.1%
(10-cycle), crashes to 21.6% (50-cycle), then partially recovers to
35.5% (200-cycle). The 50-cycle FashionMNIST std is 6.6%.

Fisher information matrices accumulate across every TASK_TRAINED
boundary. More cycles means more Fisher consolidation steps. The
quadratic penalty grows and eventually prevents learning new tasks,
but the effect is unstable across seeds.

### 4. Joint is stable

Joint accuracy varies less than 2 points across all configs on both
datasets (95.1-96.7% MNIST, 84.5-86.3% FashionMNIST). The oracle
ceiling is indifferent to curriculum order because it retrains on all
accumulated data at every boundary.

### 5. FashionMNIST is a harder test with more room for improvement

Sequential replay on FashionMNIST (77.0%) leaves a 7.5 point gap to the
oracle (84.5%). Sequential replay on MNIST (88.7%) leaves a 6.4 point
gap to its oracle (95.1%). FashionMNIST has fuzzier class boundaries
(T-shirt vs. shirt, pullover vs. coat), which stress the VAE's
reconstruction quality and the solver's decision boundaries.

## Implications for Stage 0

The Stage 0 subject should be evaluated on both MNIST and FashionMNIST.
FashionMNIST is the more informative test: it has more headroom between
baselines and oracle, and it discriminates between methods that MNIST
conflates (replay looks good on MNIST at all cycle counts, but
FashionMNIST shows that heavy cycling actually hurts replay).

The 10-cycle sweet spot suggests that a latent-reasoning subject
benefits from moderate re-exposure to old tasks. The Stage 0 gate
evaluation should include a cycling sweep to see whether the
architecture handles re-exposure better than replay does.

The split-classify generator already supports FashionMNIST via the
`data_source: "fashion-mnist"` config option. No generator changes
are needed for Stage 0 to use it.

## Confounds

The 50-cycle and 200-cycle configs hit the 50-iteration floor. Their
total SGD budget (50 * 250 or 50 * 1000 boundaries = 12,500 or 50,000
iterations) exceeds the sequential baseline's 10,000 iterations. The
high-cycle naive results partly reflect extra compute, not just better
curriculum ordering.

## Reproduction

```
python scripts/show_gate.py > gate.log 2>&1
```

Full output (50,000 lines): `gate.log` in the repository root.
Raw seed-level data is in the log under each `GRAND SUMMARY` header.
