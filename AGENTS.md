# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pensive** is a staged research implementation of a brain-inspired continual learning architecture. The system learns and reasons in latent vector space rather than tokens, with a fast-weight hippocampus for online learning and offline replay-based consolidation. The design is staged and each stage must pass a measured gate before the next begins.

Current status: Stage 0 complete (latent core, code only). Stage -1 harness is production-ready with 1365+ passing tests. GPU training runs not yet executed.

**Key references:**
- `docs/PLAN.md` - staged implementation roadmap (Stage -1 through Stage 5)
- `docs/Reference.md` - neuroscience grounding and architecture justification
- `docs/STAGE-MINUS-1.md` - Stage -1 contract and evaluation harness design
- `docs/STAGE-MINUS-1-CONTRACT.md` - exact reproducibility requirements for the gate

## Repository Structure

```
pensive/
  core/              Latent loop, Qwen backbone tap, residual learning
  train/             Training runners (stage0, eggroll, alternating)
                     Trainers, checkpoint management, dataset loading
  workspace/         Concept slots (persistent latent state container)
  codecs/            Text encoder/decoder at system edges
  eval/              Streaming benchmark harness, metrics, baselines, gates
  runtime/           (Stage 1+) always-on loop, output gate
  memory/            (Stage 2+) fast-weight hippocampus
  consolidate/       (Stage 3+) generative replay, idle consolidation
  tests/             Comprehensive pytest suite (1365+ tests)
  scripts/           Utility scripts (gate runner, stream visualization)
  openspec/          Formal spec documents and change proposals
  docs/              Architecture, plan, contracts
```

## Common Development Commands

### Testing

```bash
# Full test suite (runtime: ~8 min on CPU, ~10 min with GPU training tests)
pytest

# Single test file
pytest tests/test_latent_loop.py

# Single test
pytest tests/test_latent_loop.py::test_latent_loop_step_shape

# Tests matching a pattern
pytest -k "eggroll" 

# Run with verbose output (useful for debugging test setup)
pytest -vv tests/test_filename.py

# Run with print statements visible
pytest -s tests/test_filename.py

# Stop at first failure
pytest -x

# Exit on second failure
pytest --maxfail=2
```

### Code Quality

```bash
# Check for ruff violations (lint + style)
ruff check .

# Auto-fix safe violations (imports, formatting, obvious errors)
ruff check --fix .

# Format code (applies ruff format rules)
ruff format .

# Run both check and format
ruff check --fix . && ruff format .
```

### Training and Evaluation

```bash
# Train Stage 0 latent core (default 10 epochs, ~2-5 min per epoch on GPU)
python -m train.run_training --epochs 10

# Train with custom hyperparameters
python -m train.run_training --epochs 10 --lr 0.01 --slot-count 16 --device cuda

# Run Stage 0 reproducibility gate (requires full 5-seed run)
python scripts/run_gate.py

# Visualize stream for a generator
python scripts/show_stream.py --generator assoc --seed 42

# Plot training curves
python -m train.plot_training --checkpoint-dir checkpoints/
```

## Architecture Patterns and Key Design Decisions

### Latent Representation (Stage 0+)

Thought happens in a **workspace** - a fixed set of continuous concept slot vectors, typically 8 to 64 slots. No tokens exist in the core reasoning loop. This design:
- Decouples internal compute from token I/O (tokens only at encoding/decoding edges)
- Allows reasoning over abstract concepts rather than surface tokens
- Enables fast weight updates during use

Key files: `workspace/concept_slots.py`, `core/latent_loop.py`

### Two-Speed Learning (Stage 2+)

The system will have two learning systems:
- **Fast (hippocampus)**: Weights update during inference on every step, capturing surprising events strongly. Capacity-limited, decays naturally.
- **Slow (cortex)**: Frozen during inference, trained only during idle consolidation on replayed episodes.

Current Stage 0 has only the slow core. This split prevents catastrophic forgetting and enables learning from a single example.

### Checkpoint and Resume Mechanics

Training runs are long (hours to days). Stage 0 uses `train.standalone_checkpoint` to:
- Save complete checkpoint state at regular intervals
- Resume from any checkpoint, bit-identical to stopped state
- Validate checkpoint compatibility before resume
- Freeze randomness (`configure_deterministic_runtime`) to make resumed runs deterministic

Resume is verified with paired tests (`test_*_resume_equivalence.py`). Always checkpoint; do not rely on "remember state." The stream runner restores state through `eval/run/runner.py` and `eval/run/checkpoint.py`. Stage 0 trainer checkpoints use `train/standalone_checkpoint.py` and `train/alternating_checkpoint.py`.

### Streaming Evaluation (Stage -1)

The harness in `eval/` is built for **non-stationary streams**, not static test sets. This matters:
- Probes (evaluation points) are embedded in a stream and measure what is still remembered
- `Observe` events may be learned from; `Probe` events are measured but isolated from learning
- Generators are seeded and deterministic; order is part of the benchmark
- Baseline runs put a matched conventional model on the same stream, same seed

Key files: `eval/gate/`, `eval/run/`, and `eval/stream/generators/`

### Spec-Driven Development (Stage -1+)

Formal requirements live in `openspec/specs/`. This is not documentation; it is executable ground truth:
- Specs use MUST (verified by tests) and SHOULD (compliance-desired)
- Run `openspec validate --all --strict` to check compliance
- Changing a spec is a breaking change; audit impact via `openspec diff` before editing

Current status: all Stage -1 specs pass (`39 passed, 0 failed`).

## Critical Platform Issues and Workarounds

### Windows Transformers Access Violation

On Windows, transformers may crash with an access violation in `torch/storage.py __getitem__` during weight materialization from safetensors, even with async disabled.

**Mitigation in place**: `conftest.py` sets `HF_DEACTIVATE_ASYNC_LOAD=1` to force serial loading. This trades speed for stability; verified in transformers source.

**Symptom**: pytest crashes at ~80-95% through the full suite, different test each run, stderr shows the torch storage error.

**Action if it recurs**: The crash is in the memory-mapped safetensors layer itself, not in our code. Suspect transformers version mismatch, stale cached weights, or disk corruption. Rebuild the venv if the issue appears in a fresh clone.

### Shell Dialect on Windows

PowerShell is the user's shell. When copying Bash commands from docs:
- Use `;` not `&&` for sequential execution (PowerShell does not short-circuit on failure like Bash)
- Avoid bash-isms: `$(cmd)` works, but `${VAR}` does not (use `$var`)
- Backward slashes in paths: PowerShell handles both `/` and `\`, but quoting varies with context

## Version Locks and Reproducibility

**Python**: 3.13+

**Key locked packages** (from pyproject.toml):
- `transformers`: handles Qwen backbone loading; async load behavior is version-sensitive
- `torch`: 2.11.0+cpu pinned for torchvision==0.26.0+cpu compatibility
- `sentence-transformers`: frozen encoder for Stage 0; lock version if replacing

**Why it matters**: Continual learning experiments are fragile. A minor library update can change numerical behavior enough to fail a reproducibility gate. Pin early, test before updating, save reproduction runs against the old version before rolling forward.

## Testing Notes

### High-Cost Tests

Some tests run Stage 0 training for verification:
- `test_stage0_checkpoint_resume.py` - trains 2 epochs
- `test_stage0_trainability_*` - trains full dataset, full epochs
- `test_run_training.py`, `test_gradient_training.py` - stage0 trainers

These take 2-10 minutes each on GPU. Use `-k` filtering if iterating on the latent loop itself.

### Probe Isolation

The `eval/` harness requires that probes do not leak into training. `test_probe_isolation.py` verifies this boundary. If you add a new stream event type or oracle, add an isolation test.

### Spec-Test Alignment

Tests in `tests/test_*_contract.py` are alignment checks: they verify the code does what the spec says. If a test documents a spec deviation, mark it with `# SPEC: <issue>` and note it in `docs/STAGE-MINUS-1-CONTRACT.md` under deviations.

## Code Style and Conventions

- **Imports**: Use `from __future__ import annotations` at the top of every file (enables PEP 563 deferred evaluation, required for forward references in Stage 0 type hints).
- **Type hints**: All public functions and class methods must have type hints. Use `torch.Tensor` not `Tensor` (clarity).
- **Line length**: 120 characters (set in pyproject.toml).
- **Comments**: Avoid; self-documenting names are preferred. Comment only the WHY, not the WHAT. Spec-referencing comments (`# SPEC: <ref>`) are encouraged.
- **Naming**: Use descriptive names over abbreviations. `hidden_state` not `h_state`. Exception: `slot_queries`, `workspace`, `tap_layer` are architectural terms in the spec and should be exact.

## Assumptions and Known Issues

### Latent Collapse

The workspace (concept slots) can collapse to trivial constant representations without regularization. This is managed at the encoder level (VICReg stop-gradient, EMA target in later stages), but if you see all slot values converging, check:
- Encoder gradient flow (should be frozen in Stage 0)
- Projection layer initialization (uses gamma-init, see `core/latent_loop.py`)
- Whether an experiment accidentally enabled encoder training

### Stage 0 Gate Unreliability

The reproducibility gate is sensitive to:
- Seed ordering (five distinct seeds must show convergence as a group)
- Numerical precision (float32 vs float64 mismatches break resumption)
- Qwen model variant (code assumes Qwen2.5-0.5B-Instruct; version changes break)

Do not assume a single run proves correctness. The gate requires five paired runs.

## Development Workflow

### When Starting a Task

1. **Read the relevant spec** if one exists (`openspec/specs/` or `STAGE-MINUS-1-CONTRACT.md`).
2. **Check CLEANUP_LEDGER.md** to see what has been tried and what blockers remain.
3. **Run the full test suite** to establish baseline. Note the runtime (it varies with machine load and torch caching).
4. **Make your change. Run tests again.** If you add behavior, add tests in a `test_*.py` file in the `tests/` directory.

### Debugging a Test Failure

- Run the specific test with `-vv -s` to see print output and full assertion details.
- Check whether the test is environment-dependent (GPU availability, torch compiled versions).
- If the failure is intermittent (esp. Windows), collect 3-5 runs to see if it is flaky.
- If you cannot reproduce it locally but it fails in CI, add debug logging and re-run.

### When Committing

- Run `ruff check --fix . && ruff format .` before staging.
- Run `pytest` against the full suite (or a targeted subset if time-constrained).
- Use `git status --porcelain` and account for every untracked file. Files generated by skills, scripts, or CI must be in `.gitignore` (add the ignore entry same commit).
- Commit message should reference which stage/contract it affects and the gate (if any) it relates to.

## Useful Debug Patterns

### Inspecting Checkpoint State

```python
import torch

ckpt = torch.load("checkpoints/stage0_epoch_5.pt", weights_only=False)
print(ckpt.keys())  # top-level keys
print(ckpt["trainer_state"].keys())  # trainer internals
print(ckpt["model"].latent_loop.projection.weight.shape)  # model state
```

### Visualizing Stream Output

```bash
python scripts/show_stream.py --generator split-classify --seed 42 --limit 10
```

Prints first 10 stream events (and ground truth for probes). Use this to debug payload rendering or spot dataset issues.

### Instrumenting Training Loops

The `eval.instrumentation` module has cost counters and metrics collectors. Add to your trainer:

```python
from eval.instrumentation import CostCounters

costs = CostCounters()
# ... in loop:
costs.forward_pass(flops=model.flops)
costs.backward_pass(grad_norm=grad.norm())
print(f"FLOPs/step: {costs.mean_forward_flops}")
```

## When Things Break

### Test Flakiness on Windows

If the full suite occasionally crashes (not fails, but crashes) with a torch storage error:
- This is the known Windows transformers issue.
- Run the suite again (usually passes on second try).
- Escalate only if it becomes regular (same test, same error, every run).

### Ruff Violations in openspec/ or .venv/

Ignore. These are generated/vendored. Run `ruff check` from the project root; it will skip ignored paths.

### Import Errors on Resume

If you get `ModuleNotFoundError` after resuming from an old checkpoint:
- A file was renamed or moved since the checkpoint was saved.
- Checkpoint was saved with a different Python path setup (venv mismatch).
- Delete the checkpoint and retrain from scratch (checkpoints are not archival; they are session-local).

## Next Steps for Future Stages

When you move to Stage 1 (persistent state, adaptive halting):
- Expect to wire `runtime/` and introduce the `runtime_clock`.
- The `memory/` and `consolidate/` modules will remain stubs until Stage 2+.
- The gate moves from "match baseline accuracy" to "performance holds across an artificially long stream."

Read `PLAN.md` Stage 1 and the corresponding change proposal under `openspec/changes/stage-1-persistent-state/proposal.md` before starting.

## dod-guard GitHub workflow

Use `/add-backlog-idea` to capture ideas in Backlog. Use `/refine-backlog-item` to research an idea and move it to Todo as a PBI. Use `/next-ticket` to implement and push the selected PBI branch. Use `/submit-draft-pr` to create or update its draft pull request. Keep review read-only until the user explicitly accepts findings. Use `/complete-pr` for ready, merge, linked-issue confirmation, and remote branch deletion.
