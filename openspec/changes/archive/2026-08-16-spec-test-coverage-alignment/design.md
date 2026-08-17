## Context

See proposal.md for motivation. The project has 10 specs under `openspec/specs/eval/` and 18 test files under `tests/`. A coverage audit found two kinds of gaps: spec scenarios without tests, and test files without specs.

## Goals / Non-Goals

**Goals:**
- Every spec scenario under `eval/` has a corresponding test with a `# covers:` marker
- Every behavioral test file has a corresponding spec

**Non-Goals:**
- Changing production code (this is spec-and-test alignment only)
- Adding edge-case scenarios beyond what specs already define
- Measuring or improving code coverage percentages

## Decisions

**1. No delta specs for existing capabilities**

All 12 uncovered scenarios and 7 partial scenarios already exist in the specs. The gap is test coverage, not spec coverage. Writing delta specs that repeat existing text adds nothing. Tests will reference the existing spec scenarios via `# covers:` comments.

Alternative considered: writing MODIFIED deltas that mark scenarios as "now tested". Rejected because delta specs describe behavior changes, not test status changes.

**2. Subject specs go under `eval/subject/`**

The 5 new specs follow the existing `eval/` prefix. Subject-layer behavior (protocol, oracles, isolation) groups under `eval/subject/` with further nesting by concern.

Alternative considered: a top-level `subject/` outside `eval/`. Rejected because all current specs live under `eval/` and the subject layer is part of the eval harness.

**3. Tests go in existing test files where possible**

New tests for existing spec scenarios go into the test file that already covers that spec (e.g., events R4-S1 goes in `test_events.py`). New tests for new specs go into the test file that already exists for that concern (e.g., `test_oracles.py` already covers the oracles spec).

**4. `# covers:` markers on every test**

Each test gets a comment of the form `# covers: eval/<capability> :: <requirement> :: <scenario>` linking it to its spec scenario. Existing tests that already cover scenarios do not need markers retrofitted in this change.

## Risks / Trade-offs

- [OBSERVED-priority scenarios may not have implementation support] Some scenarios are marked OBSERVED in their spec, meaning the behavior is observed but not contractually required. Tests for these may break if the implementation changes. Mitigation: OBSERVED scenarios are lower priority and can be deferred.
- [split-classify R12 cluster separability is hard to test deterministically] The scenario says "class centers are spaced farther apart than the noise standard deviation". This requires computing actual cluster statistics from generated data. Mitigation: a statistical test with a generous threshold.
