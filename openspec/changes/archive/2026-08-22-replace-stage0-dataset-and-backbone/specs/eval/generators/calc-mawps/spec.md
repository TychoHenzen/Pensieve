## Purpose

Provides a deterministic Stage 0 stream over the filtered Calc-MAWPS arithmetic word problems with stable item identity and numerical truth.

## ADDED Requirements

### Requirement: Filtered Calc-MAWPS split binding
The generator MUST load revision `38c10053efeafd20ab6ff4e08c3ec17de26c19b7` of `MU-NLPC/Calc-mawps` with configuration `default`. It MUST download only `data/train-00000-of-00001-4bb1451333aad61c.parquet`, `data/validation-00000-of-00001-2ce28573971ca59f.parquet`, and `data/test-00000-of-00001-5a59f3fc4b0d9c98.parquet`, whose SHA-256 digests are respectively `7a8dd8f7680b5e5ddb908cdfa33867c298d9225d1d7595a17d4771c09e482140`, `9d2ca9e33d8efdbc2527a84f1685d6dd5ec7defb201a290a836660c071f5b607`, and `2ef6313cb811d5c5ebef422a909015f7db4750020f835fd95c2ac2ccbf343d82`. Loading MUST disable remote code and MUST NOT execute a repository script or substitute `original-splits`. Raw cardinalities MUST be 1,089 train, 1,040 validation, and 520 test. The loader MUST then exclude validation id `mawps__qA0gWJatQEeMzvOw`, the pinned duplicate of test id `mawps__yCG5jGSKjPM9koup`, leaving exactly 1,039 usable validation rows. It MUST apply no other row exclusion. A digest, cardinality, excluded-id, or duplicate-fingerprint mismatch MUST fail loading.

#### Scenario: Default filtered splits load
- **WHEN** the Stage 0 dataset loader requests train, validation, or test data
- **THEN** it verifies the raw split and returns 1,089 train, 1,039 decontaminated validation, or 520 test records as requested

#### Scenario: Dataset revision unavailable
- **WHEN** the pinned Calc-MAWPS revision cannot be loaded
- **THEN** loading fails with an error that names the dataset and revision

### Requirement: Canonical math problem records
Each loaded record MUST use the pinned source `id` string as its item identifier. Source ids MUST be at most 512 Unicode code points, non-empty, and unique across all three splits. Raw questions MUST be at most 16,384 code points. Question normalization MUST be exactly `" ".join(unicodedata.normalize("NFC", question).split())`, without changing punctuation or letter case. Source `result` MUST be at most 512 code points. The pinned source MAY use `_` only in positions where the gate grammar permits a grouping comma; the loader MUST replace those underscores with commas before applying the same bounded integer, decimal, or fraction grammar and 256-digit limit as gate predictions. All other underscore placements MUST be rejected. Parsing MUST use `int`, `Decimal`, and `Fraction` without expression or symbolic evaluation. The binary float source field `result_float` MUST first convert with `Decimal(str(result_float))`, MUST be finite, and MUST satisfy the gate's inclusive tolerance formula against parsed `result`. The loader MUST reject any cross-split duplicate of either source id or the SHA-256 fingerprint of canonical JSON `{question, target}`. A row with a missing field, duplicate, oversized value, empty normalized question, unparseable target, non-finite float, or disagreeing target fields MUST be rejected with an error that names the split, row position, and source identifier when present.

#### Scenario: Valid row normalization
- **WHEN** a valid Calc-MAWPS row is loaded
- **THEN** the loader returns its source identifier, exactly normalized question, and canonical numerical target

#### Scenario: Malformed row rejected
- **WHEN** a Calc-MAWPS row lacks a usable question or finite numerical target
- **THEN** the loader rejects it and names the offending row identity or position

### Requirement: Calc-MAWPS stream follows the v1 schema
The generator MUST present each selected problem as an `Observe` event whose public content is only the normalized question, followed by a numerical `Probe` whose public query is exactly `What is the numerical answer?`. Truth and identifiers MUST remain in evaluator-owned side-channel fields that are never passed to the subject, prompt builder, tokenizer, encoder, latent loop, or decoder. Tests MUST spy on all arguments at those model-facing boundaries and verify that only public strings, token tensors, attention masks, position identifiers, and slot tensors appear.

#### Scenario: Observe and probe pair
- **WHEN** one selected Calc-MAWPS problem is generated
- **THEN** one word-problem observation is followed by one numerical probe with side-channel truth

#### Scenario: Truth remains isolated
- **WHEN** the observation and probe are rendered for a subject
- **THEN** neither rendered text nor any model-facing argument contains the answer, item identifier, probe identifier, or task identifier

### Requirement: Selection identity is deterministic
Selection MUST begin from pinned source order, shuffle identifiers with Python `random.Random(seed).shuffle`, and then take the first `problem_count` identifiers. An omitted count MUST normalize to the usable split cardinality before selection and identity serialization. A count MUST be an integer from 1 through that cardinality; other values MUST be rejected. Canonical JSON MUST use `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")`; identity fields contain only strings, integers, lists, and objects, with no null or floating values. Each normalized record object MUST be `{id, question, target}` in selected order. Selection identity MUST be the SHA-256 digest of canonical JSON object `{schema_version: 1, dataset: "MU-NLPC/Calc-mawps", configuration: "default", revision: "38c10053efeafd20ab6ff4e08c3ec17de26c19b7", split, seed, problem_count, ordered_item_ids, records_sha256}`. The invocation seed MUST remain in the identity even when two seeds happen to produce the same effective order.

#### Scenario: Repeated selection
- **WHEN** the generator runs twice with the same split, selection arguments, and seed
- **THEN** both runs produce the same ordered item identifiers and selection identity

#### Scenario: Selection input changes
- **WHEN** the split, revision, selected count, ordered records, or seed changes
- **THEN** the resulting selection identity differs
