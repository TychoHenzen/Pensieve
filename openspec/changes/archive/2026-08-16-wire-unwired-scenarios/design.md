## Context

See proposal.md. 246 scenarios across 10 eval capabilities have no test binding. The specs already contain the scenarios with GIVEN/WHEN/THEN. The test files already exist and follow pytest conventions with `# covers:` comments.

## Goals / Non-Goals

**Goals:**
- Write one test function per unwired scenario in the existing test files
- Each test carries a `# covers: <capability>::<requirement>::<scenario>` comment
- All new tests pass against the current implementation

**Non-Goals:**
- Changing production code to make tests pass (if a test would fail, the scenario contradicts the implementation and needs resolution, not a code change)
- Refactoring existing tests
- Achieving any coverage metric beyond binding the unwired scenarios

## Decisions

1. **Use `/dod-guard:spec-test` per capability** rather than hand-writing tests. The skill reads the spec, reads the implementation, and generates tests that bind to scenarios. It already produces `# covers:` comments in the right format.

   Alternative: hand-write each test. Rejected because 246 tests is too many to write manually in one session.

2. **Group by test file, not by requirement.** Each capability maps to one or two test files. Run spec-test once per capability, targeting the existing test file. This keeps related tests together and avoids creating new test modules.

3. **Order by unwired count ascending.** Start with the capabilities that have the fewest unwired scenarios (package: 2, serialize: 5) so early steps complete fast and build confidence before the larger batches (generators: 110).

## Risks / Trade-offs

- [Spec-test may generate tests that fail] The scenario's GIVEN/WHEN/THEN may describe behavior the implementation does not match. Mitigation: fix the scenario text in a follow-up change rather than changing production code.
- [110 unwired scenarios in eval/generators spans 3 generator specs] Mitigation: split generators into 3 sub-tasks (assoc, split-classify, difficulty-mix) matching the existing test files.
