## Why

The split-compound-requirements change split 96 compound requirements into individual scenarios. 238 of those scenarios have no test binding (`dod-guard cover` reports them as `unwired`). Each unwired scenario is a stated obligation with no generated or hand-written test to verify it.

## What Changes

- Generate `# covers:` test functions for all 238 unwired scenarios across 13 capabilities
- Each test function names the scenario it covers in a `# covers:` comment so `dod-guard cover` reports it as `bound`
- No behavioral changes to production code - test-only additions

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. The specs already contain the scenarios. This change writes tests that bind to them.

## Impact

- Test files under `tests/stream/` and `tests/subject/` gain new test functions
- No production code changes
- `dod-guard cover --all` unwired count drops from 238 toward 0
