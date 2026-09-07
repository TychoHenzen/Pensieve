# Conventions

- Put `from __future__ import annotations` at the top of every Python source file.
- Type all public functions and class methods. Spell tensor types as `torch.Tensor`.
- Prefer descriptive names. Keep architecture terms exact: `slot_queries`, `workspace`, and `tap_layer`.
- Prefer self-documenting code. Comments explain why, not what. Use `# SPEC: <ref>` for explicit contract links.
- Ruff owns formatting. Configured line length is 120.
- Keep stream generation seeded and deterministic. Event order is benchmark data.
- Preserve the Observe versus Probe isolation boundary.
- Treat OpenSpec MUST requirements as executable ground truth. Audit impact before changing a spec.
- Preserve checkpoint schema, complete trainer state, randomness state, and resume equivalence when touching training code.
- For learning work, require decoded or behavioral evidence against the protected baseline. Structural tests and loss curves are insufficient.