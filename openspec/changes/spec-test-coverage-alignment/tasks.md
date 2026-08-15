## 1. Tests for uncovered scenarios in existing specs

- [x] 1.1 Add `test_boundary_kind_value_strings` to `test_events.py` asserting `.value` on all three BoundaryKind members (events R4-S1)
- [x] 1.2 Add `test_probe_truth_difficulty_defaults_to_none` to `test_truth.py` (events R7-S1)
- [x] 1.3 Add `test_stream_config_attribute_assignment_raises` to `test_hashing.py` assigning to `.generator` (config R3-S1)
- [x] 1.4 Add `test_vocab_is_a_tuple` to `test_vocab.py` (vocab R1)
- [x] 1.5 Add `test_sample_negative_n_raises` to `test_vocab.py` (vocab R2-S3)
- [x] 1.6 Add `test_vocab_version_is_a_non_empty_string` to `test_vocab.py` (vocab R4)
- [x] 1.7 Add `test_load_corpus_rejects_wordless_snapshot` to `test_corpus.py` (corpus R10-S1)
- [x] 1.8 Add `test_canonical_json_rejects_nan` and `test_canonical_json_rejects_infinity` to `test_hashing.py` (serialize R7-S1)
- [x] 1.9 Add `test_recall_distance_zero_raises` to `test_assoc.py` (assoc R9-S1)
- [x] 1.10 Add `test_cluster_centers_are_separable` to `test_split_classify.py` (split-classify R12-S1)
- [x] 1.11 Add `test_empty_difficulty_levels_raises` to `test_difficulty_mix.py` (difficulty-mix R11-S1)

## 2. Strengthen partially covered scenarios

- [x] 2.1 Add `test_boundary_kind_value_is_literal_string` to `test_events.py` pinning all three `.value` strings (events R4 partial)
- [x] 2.2 Add `test_split_classify_and_difficulty_mix_in_registry` to `test_registry.py` (generator R5 partial)
- [x] 2.3 Add `test_default_token_counter_name_is_regex_whitespace_v1` to `test_render.py` (render R13 partial)
- [x] 2.4 Add `test_canonical_json_uses_compact_separators` to `test_hashing.py` (serialize R4 partial)
- [x] 2.5 Add `test_distance_exceeding_max_raises` and `test_negative_filler_density_raises` to `test_assoc.py` (assoc R9 partial)
- [x] 2.6 Add `test_num_items_zero_raises` and `test_difficulty_level_below_one_raises` to `test_difficulty_mix.py` (difficulty-mix R11 partial)

## 3. Archive new specs

- [x] 3.1 Archive `eval/package` spec into `openspec/specs/eval/package/spec.md`
- [x] 3.2 Archive `eval/subject/protocol` spec into `openspec/specs/eval/subject/protocol/spec.md`
- [x] 3.3 Archive `eval/subject/snapshot` spec into `openspec/specs/eval/subject/snapshot/spec.md`
- [x] 3.4 Archive `eval/subject/isolation` spec into `openspec/specs/eval/subject/isolation/spec.md`
- [x] 3.5 Archive `eval/subject/oracles` spec into `openspec/specs/eval/subject/oracles/spec.md`

## 4. Add `# covers:` markers to new tests

- [x] 4.1 Add `# covers:` comments to every test written in groups 1 and 2, linking each to its spec scenario

## 5. Verify

- [x] 5.1 Run `pytest tests/` and confirm all new and existing tests pass
- [x] 5.2 Run `openspec validate --change spec-test-coverage-alignment` and confirm no errors
