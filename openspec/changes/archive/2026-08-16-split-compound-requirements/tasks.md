## 1. Split compound requirements in eval/events

- [x] 1.1 Archive the events delta spec into `openspec/specs/eval/events/spec.md`, replacing all 8 compound requirements with their split scenarios
<!-- covers: eval/events :: Event immutability :: attribute assignment on a constructed Observe -->
<!-- covers: eval/events :: Event required fields :: Probe without probe_id -->
<!-- covers: eval/events :: Event optional-field defaults :: narration defaults to None -->
<!-- covers: eval/events :: BoundaryKind string values :: session_end string value -->
<!-- covers: eval/events :: Event type union :: isinstance dispatch covers all four types -->
<!-- covers: eval/events :: StreamItem truth pairing invariant :: probe without truth -->
<!-- covers: eval/events :: ProbeTruth immutability and optional difficulty :: ProbeTruth is frozen -->
<!-- covers: eval/events :: subject_view hides truth and hidden boundaries :: hidden boundary omitted -->
<!-- covers: eval/events :: harness_view yields everything unfiltered :: hidden boundary included for harness -->
- [x] 1.2 Run `check-spec-hygiene.mjs --root=. --strict` and confirm 0 compound requirements for eval/events

## 2. Split compound requirements in eval/config

- [x] 2.1 Archive the config delta spec into `openspec/specs/eval/config/spec.md`
<!-- covers: eval/config :: StreamConfig immutable params :: top-level params assignment rejected -->
<!-- covers: eval/config :: StreamConfig params are canonically hashable :: key order does not affect the hash -->
<!-- covers: eval/config :: StreamConfig is frozen :: frozen generator attribute -->
- [x] 2.2 Re-run hygiene check for eval/config

## 3. Split compound requirements in eval/render

- [x] 3.1 Archive the render delta spec into `openspec/specs/eval/render/spec.md`
<!-- covers: eval/render :: RENDER_VERSION is a non-empty string :: version is present -->
<!-- covers: eval/render :: Idle carries no text and render_event refuses it :: carries_text returns False for Idle -->
<!-- covers: eval/render :: Probe renders its query and hides harness fields :: probe renders query -->
<!-- covers: eval/render :: rendered_subject_view drops Idle and hidden Boundary :: Idle events omitted -->
<!-- covers: eval/render :: default_token_counter is deterministic and reports its name :: repeated counting is stable -->
<!-- covers: eval/render :: token distance is measured over rendered text between teaching and probe :: adjacent teaching and probe measure zero -->
- [x] 3.2 Re-run hygiene check for eval/render

## 4. Split compound requirements in eval/generators/assoc

- [x] 4.1 Archive the assoc delta spec into `openspec/specs/eval/generators/assoc/spec.md`
<!-- covers: eval/generators/assoc :: Generator identity and version :: name is assoc -->
<!-- covers: eval/generators/assoc :: Keys are distinct vocabulary words :: keys are pairwise distinct -->
<!-- covers: eval/generators/assoc :: Config validation :: distance below 1 -->
<!-- covers: eval/generators/assoc :: token_distance absent without counter, computed with counter :: no counter configured -->
<!-- covers: eval/generators/assoc :: Deterministic in (config, seed) and diverges on different seeds :: replay determinism -->
- [x] 4.2 Re-run hygiene check for eval/generators/assoc

## 5. Split compound requirements in eval/generators/split-classify

- [x] 5.1 Archive the split-classify delta spec into `openspec/specs/eval/generators/split-classify/spec.md`
<!-- covers: eval/generators/split-classify :: Generator identity and version :: name is split-classify -->
<!-- covers: eval/generators/split-classify :: Validates all four required params as >= 1 :: zero tasks rejected -->
<!-- covers: eval/generators/split-classify :: chance_rate equals 1/classes_per_task, computed from that single param alone :: chance rate with minimal config -->
<!-- covers: eval/generators/split-classify :: Deterministic in (config, seed) and diverges on different seeds :: replay determinism -->
- [x] 5.2 Re-run hygiene check for eval/generators/split-classify

## 6. Split compound requirements in eval/generators/difficulty-mix

- [x] 6.1 Archive the difficulty-mix delta spec into `openspec/specs/eval/generators/difficulty-mix/spec.md`
<!-- covers: eval/generators/difficulty-mix :: Generator identity and version :: name is difficulty-mix -->
<!-- covers: eval/generators/difficulty-mix :: Validates num_items, difficulty_levels, and probe_rate :: empty difficulty_levels rejected -->
<!-- covers: eval/generators/difficulty-mix :: Deterministic in (config, seed) and diverges on different seeds :: replay determinism -->
- [x] 6.2 Re-run hygiene check for eval/generators/difficulty-mix

## 7. Split compound requirements in eval/corpus

- [x] 7.1 Archive the corpus delta spec into `openspec/specs/eval/corpus/spec.md`
<!-- covers: eval/corpus :: resolve locates a snapshot under an overridable directory :: env var overrides the default directory -->
<!-- covers: eval/corpus :: corpus_id is the SHA-256 of the raw file bytes :: corpus_id equals SHA-256 hex digest -->
<!-- covers: eval/corpus :: a non-positive word count raises ValueError :: zero request -->
- [x] 7.2 Re-run hygiene check for eval/corpus

## 8. Split compound requirements in remaining specs

- [x] 8.1 Archive the vocab delta spec into `openspec/specs/eval/vocab/spec.md`
<!-- covers: eval/vocab :: VOCAB pool size, uniqueness, order, and shape :: vocabulary has at least 256 entries -->
<!-- covers: eval/vocab :: sample determinism, distinctness, and bounds checking :: reproducible draw -->
- [x] 8.2 Archive the generator delta spec into `openspec/specs/eval/generator/spec.md`
<!-- covers: eval/generator :: StreamGenerator protocol shape :: protocol conformance -->
<!-- covers: eval/generator :: Registry resolves a config's generator name :: unknown generator name rejected -->
<!-- covers: eval/generator :: Three generators are registered under fixed names :: assoc is registered -->
- [x] 8.3 Archive the serialize delta spec into `openspec/specs/eval/serialize/spec.md`
<!-- covers: eval/serialize :: canonical_json sorts keys and uses compact separators :: key order does not affect output -->
<!-- covers: eval/serialize :: canonical_json raises TypeError on sets :: a set cannot be canonicalized -->
- [x] 8.4 Archive the package delta spec into `openspec/specs/eval/package/spec.md`
<!-- covers: eval/package :: Package importability :: clean import -->
<!-- covers: eval/package :: STREAM_SCHEMA_VERSION constant :: version is a non-empty string -->
- [x] 8.5 Archive the subject/protocol delta spec into `openspec/specs/eval/subject/protocol/spec.md`
<!-- covers: eval/subject/protocol :: Subject is an abstract base class :: direct instantiation rejected -->
<!-- covers: eval/subject/protocol :: CostCounters is a frozen dataclass with zero defaults :: frozen after construction -->
<!-- covers: eval/subject/protocol :: Cost monotonicity :: steps increase after answer -->
- [x] 8.6 Archive the subject/oracles delta spec into `openspec/specs/eval/subject/oracles/spec.md`
<!-- covers: eval/subject/oracles :: PerfectMemoryOracle recalls all observations :: recalls everything -->
<!-- covers: eval/subject/oracles :: ChanceOracle answers randomly from VOCAB :: accuracy near chance rate -->
<!-- covers: eval/subject/oracles :: CheaterOracle exploits probe truth map :: unknown without truth map -->
- [x] 8.7 Re-run hygiene check for all remaining specs, confirm 0 compound

## 9. Validate and generate tests

- [x] 9.1 Run `check-spec-hygiene.mjs --root=. --strict` across all 15 specs, confirm 0 compound requirements total
- [ ] 9.2 Run spec-test per capability to generate `// covers:` tests for the newly split scenarios
<!-- status: skipped -->
- [x] 9.3 Run full pytest suite, fix any contradictions between spec and implementation
