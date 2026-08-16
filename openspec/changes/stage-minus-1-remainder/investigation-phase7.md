# Phase 7 Investigation: van de Ven et al. baseline reference numbers

## Source

van de Ven, G.M. and Tolias, A.S. (2019). "Three scenarios for continual
learning." arXiv:1904.07734. Table 4, class-incremental Split-MNIST.

Released code: github.com/GMvandeVen/continual-learning (Python 3.10,
PyTorch 1.11).

## Architecture

- 2-hidden-layer MLP, 400 units per layer, ReLU activations
- Output: 10 units (all digits, single head for class-incremental)
- Optimizer: Adam (beta1=0.9, beta2=0.999), lr=0.001
- 2000 iterations per task, batch size 128
- 5 tasks: digits 0-1, 2-3, 4-5, 6-7, 8-9

## Class-incremental Split-MNIST accuracy (Table 4)

| Method              | Accuracy (%)     |
|---------------------|------------------|
| None (naive SGD)    | 19.90 (+/- 0.02) |
| EWC                 | 20.01 (+/- 0.06) |
| Online EWC          | 19.96 (+/- 0.07) |
| SI                  | 19.99 (+/- 0.06) |
| LwF                 | 23.85 (+/- 0.44) |
| DGR (replay)        | 90.79 (+/- 0.41) |
| DGR+distill         | 91.79 (+/- 0.32) |

Chance is 10% (10 classes). Naive and EWC both land near 20%, which means
the network learns the last task's two classes and guesses randomly on the
other eight. Replay methods reach 90%+.

## EWC details

- Quadratic penalty: lambda * sum((theta - theta_star)^2 * F_diag)
- Fisher diagonal estimated from task data after each task boundary
- Lambda (regularization strength): 1e7 in the released code for Split-MNIST
- Fisher computed over 1000 samples per task

## Generative replay details

- Generator: symmetric VAE with same hidden-layer sizes as the solver
- Latent dimension: 100
- Replay: after each task, generate pseudo-examples from the VAE and mix
  with real examples from the current task during training
- The solver and generator are trained jointly

## Wrapping the released code

Not feasible as a Subject. The code uses a monolithic training loop spread
across multiple scripts, with no event-driven interface. Reimplementing
the five baselines from scratch using the paper's architecture and
hyperparameters is simpler and produces code that directly implements the
Subject protocol.

## Decisions for implementation

1. Use the paper's architecture: 2x400 ReLU MLP, Adam lr=0.001.
2. Use iterations (not epochs): 2000 iterations per task, batch size 128.
3. EWC lambda: 1e7, Fisher samples: 1000.
4. Replay generator: symmetric VAE, latent dim 100.
5. Joint baseline: train on all tasks simultaneously (offline upper bound).
6. Frozen baseline: random init, no training at all.
