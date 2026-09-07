## Context

See `proposal.md` for motivation. The scheduler currently writes every training result and every boundary result directly to the JSON progress stream. The `--log-every` option is stored in checkpoint configuration but does not control output. Phase evaluation and checkpoint operations must remain unchanged.

The repository defines no lint or formatting command in `pyproject.toml`, so no pre-existing lint or format violation count is available. Pytest is the project check.

## Goals / Non-Goals

**Goals:**

- Make one command-side output policy decide which existing records reach the JSON stream.
- Apply `--log-every` to global training steps with a default of 50.
- Keep epoch-boundary evaluation and checkpoint records visible.
- Validate the interval before production loading.

**Non-Goals:**

- Change when training, evaluation, or checkpoint operations execute.
- Change record schemas or checkpoint payloads.
- Change standalone command output.

## Decisions

### Filter output after operations produce their records

The scheduler continues to train, evaluate, and save at the same boundaries. The command checks record position and boundary labels before writing JSON. This separates operational behavior from console visibility.

Alternative considered: skip phase evaluation and checkpoint calls. Rejected because the confirmed requirement changes logging only.

### Use global-step multiples for training progress

Training records emit when `global_step > 0` and `global_step % log_every == 0`. The restored global step participates in the same calculation after resume. Global position remains stable across phase and epoch boundaries, so output does not bunch or reset. A final non-matching step does not produce an extra training record.

Alternative considered: reset the interval per phase or epoch. Rejected because that would make output frequency depend on phase and dataset sizes.

### Keep epoch boundaries visible

Evaluation and checkpoint records emit when the boundary set contains `epoch`. A coincident phase and epoch boundary emits one evaluation record carrying both labels and one checkpoint record carrying both paths through the existing deduplication path. Silence applies only to structured JSON progress records, not warnings, errors, or third-party diagnostics.

Alternative considered: emit all boundary records. Rejected because phase cycles are the source of the reported console noise.

## Risks / Trade-offs

- [A run ends before the first interval] -> Epoch-boundary records still provide final visible statistics.
- [A phase-only failure has less nearby console context] -> Phase checkpoints and evaluations remain available on disk and in checkpoint state.
- [A non-positive interval causes division errors] -> Validate it before model or data loading.

## Migration Plan

No data migration is required. Existing invocations inherit the new default of 50. Users can choose another positive interval with `--log-every`.

## Phase 1 review

Verdict: `REVISE` after the third and final review round.

- Security: 1 minor finding about checkpoint paths in console output.
- Assumptions: 10 major and 2 minor findings, mainly malformed internal records, failure paths, and resume edge cases.
- Testability: 4 major and 4 minor findings about operation invariants and exact record contracts.
- Consistency: 2 major and 1 minor findings about boundary terminology and coincident checkpoint behavior.
- Implementability: 0 findings; existing command and scheduler hooks can implement the confirmed change.

The workflow stopped after round 3 because the major-finding threshold remained above the `GO` limit. The user explicitly overrode the Phase 1 gate with `continue`; the change proceeds with the confirmed logging-only scope.
