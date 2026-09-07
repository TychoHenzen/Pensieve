# eval/generators/asdiv-a Specification

## Purpose

Provides deterministic, disjoint Stage 0 partitions over Calc-ASDiv_A with stable item identity and numerical truth.

## Requirements

### Requirement: Filtered Calc-ASDiv_A split binding
The generator MUST load revision `520a6910e097ee287ecd2bb9104f7f45805f9df9` of `MU-NLPC/Calc-asdiv_a` with configuration `default`. It MUST load only source split `test` and file `data/test-00000-of-00001-d118ad90f5719063.parquet`. The file SHA-256 MUST equal `f13920459f68f633b3bcff528f5084066e8e6f57b166769232f8c15a294c0570`. Loading MUST disable remote code and MUST NOT execute repository code. The source cardinality MUST be 1,218. The loader MUST shuffle the source rows once with `random.Random(0).shuffle`. Shuffled positions 0 through 519 MUST form the 520-item test partition used by the measured frozen-Qwen baseline. Positions 520 through 1,089 MUST form the 570-item training partition. Positions 1,090 through 1,217 MUST form the 128-item validation partition. A digest, cardinality, or partition mismatch MUST fail loading.

#### Scenario: Default filtered splits load
- **WHEN** the Stage 0 dataset loader requests train, validation, or test data
- **THEN** it returns exactly 570 train, 128 validation, or 520 test records from the declared seed-0 partition

#### Scenario: Dataset revision unavailable
- **WHEN** the pinned Calc-ASDiv_A revision cannot be loaded
- **THEN** loading fails with an error that names the dataset and revision

### Requirement: Canonical math problem records
Each loaded record MUST use the pinned source `id` string as its item identifier. Source ids MUST be at most 512 Unicode code points, non-empty, and unique across all partitions. Raw questions MUST be at most 16,384 code points. Question normalization MUST be exactly `" ".join(unicodedata.normalize("NFC", question).split())`, without changing punctuation or letter case. Source `result` MUST be at most 512 code points. The pinned source MAY use `_` only where the gate grammar permits a grouping comma. The loader MUST replace those underscores with commas before applying the gate's bounded integer, decimal, or fraction grammar and 256-digit limit. Parsing MUST use `int`, `Decimal`, and `Fraction` without expression or symbolic evaluation. The binary float source field `result_float` MUST first convert with `Decimal(str(result_float))`, MUST be finite, and MUST satisfy the gate's inclusive tolerance formula against parsed `result`. The loader MUST reject any cross-partition duplicate source id or canonical `{question, target}` fingerprint. A malformed row MUST fail with its partition, position, and source identifier when present.

#### Scenario: Valid row normalization
- **WHEN** a valid Calc-ASDiv_A row is loaded
- **THEN** the loader returns its source identifier, normalized question, canonical numerical target, and assigned partition

#### Scenario: Malformed row rejected
- **WHEN** a Calc-ASDiv_A row lacks a usable question or finite numerical target
- **THEN** the loader rejects it and names the offending row identity or position

### Requirement: Calc-ASDiv_A stream follows the v1 schema
The generator MUST present each selected problem as an `Observe` event whose public content is only the normalized question. A numerical `Probe` MUST follow with public query `What is the numerical answer?`. Truth and identifiers MUST remain in evaluator-owned side-channel fields. Model-facing prompt, tokenizer, encoder, latent-loop, and decoder arguments MUST NOT receive them.

#### Scenario: Observe and probe pair
- **WHEN** one selected Calc-ASDiv_A problem is generated
- **THEN** one word-problem observation is followed by one numerical probe with side-channel truth

#### Scenario: Truth remains isolated
- **WHEN** the observation and probe are rendered for a subject
- **THEN** model-facing inputs contain no answer, item identifier, probe identifier, or task identifier

### Requirement: Selection identity is deterministic
Selection MUST begin from the assigned partition order. It MUST shuffle identifiers with `random.Random(seed).shuffle` and take the first `problem_count` identifiers. An omitted count MUST normalize to the partition cardinality. A count MUST be an integer from 1 through that cardinality. Each normalized record object MUST be `{id, question, target}` in selected order. Selection identity MUST hash canonical JSON object `{schema_version: 1, dataset: "MU-NLPC/Calc-asdiv_a", configuration: "default", revision: "520a6910e097ee287ecd2bb9104f7f45805f9df9", split, seed, problem_count, ordered_item_ids, records_sha256}` with SHA-256. The invocation seed MUST remain in the identity.

#### Scenario: Repeated selection
- **WHEN** the generator runs twice with the same partition, selection arguments, and seed
- **THEN** both runs produce the same ordered item identifiers and selection identity

#### Scenario: Selection input changes
- **WHEN** the partition, revision, selected count, ordered records, or seed changes
- **THEN** the resulting selection identity differs
