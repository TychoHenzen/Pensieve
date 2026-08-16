## Why

96 of 110 requirements across all eval specs are compound: they contain more obligation keywords (SHALL/MUST) than scenarios to verify them. The spec-test skill generates one test per scenario, so 140 obligations have no scenario and therefore no generated test. Splitting each compound requirement into one-obligation-per-scenario form gives spec-test a complete surface to generate against.

## What Changes

- Split compound requirements in all 15 eval specs so every obligation has its own scenario
- No behavioral changes to the implementation - this is a spec restructuring
- After the split, `check-spec-hygiene.mjs` should report 0 compound requirements
- The spec-test skill can then generate one test per scenario with full coverage

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `eval/events`: 8 compound requirements, mostly field-list obligations that need one scenario per field
- `eval/config`: 3 compound requirements
- `eval/vocab`: 2 compound requirements
- `eval/generator`: 5 compound requirements
- `eval/corpus`: 9 compound requirements
- `eval/render`: 14 compound requirements
- `eval/serialize`: 5 compound requirements
- `eval/generators/assoc`: 14 compound requirements
- `eval/generators/split-classify`: 11 compound requirements
- `eval/generators/difficulty-mix`: 12 compound requirements
- `eval/package`: 2 compound requirements
- `eval/subject/protocol`: 2 compound requirements
- `eval/subject/snapshot`: 0 compound requirements (clean)
- `eval/subject/isolation`: 0 compound requirements (clean)
- `eval/subject/oracles`: 5 compound requirements

## Impact

- All 15 spec files under `openspec/specs/eval/` are modified
- No code changes - only spec scenarios are restructured
- Existing `// covers:` markers in tests remain valid if the scenario title is preserved
- New scenarios enable spec-test to generate tests for the 140 currently uncovered obligations
