## Why

A spec-test coverage audit found 12 spec scenarios with no test, 7 with partial coverage, and 5 test files that pin behavioral requirements with no corresponding spec. The specs and tests should be a closed loop: every scenario has a test, every test traces to a scenario.

## What Changes

- Add tests for 12 uncovered spec scenarios across events, config, vocab, corpus, serialize, assoc, split-classify, and difficulty-mix
- Strengthen 7 partially covered scenarios with missing assertions
- Write new specs for 5 existing test files that cover subject-layer and package-level behavior currently unspecified

## Capabilities

### New Capabilities
- `eval/package`: Stream package importability and STREAM_SCHEMA_VERSION constant
- `eval/subject/protocol`: Subject ABC shape, CostCounters frozen dataclass, cost monotonicity
- `eval/subject/snapshot`: Oracle snapshot/restore contract (capture, revert, erase intermediate state)
- `eval/subject/isolation`: isolated_answer probe-isolation wrapper, cheater detection
- `eval/subject/oracles`: Five oracle behaviors (PerfectMemory, Forgetful, Chance, TaskWiper, Cheater)

### Modified Capabilities

None. All 8 existing specs already define the scenarios that lack tests. The gap is test coverage, not spec coverage. No delta specs are needed for existing capabilities.

## Impact

- Test files: new tests added to existing files in `tests/stream/` and `tests/subject/`
- Specs: 5 new spec files under `openspec/specs/eval/`, 8 existing specs amended with new scenarios
- No production code changes - this is a spec-test alignment pass
