## 1. Small capabilities (2-6 unwired)

- [ ] 1.1 Run spec-test for eval/package (2 unwired) targeting tests/stream/test_package.py
- [ ] 1.2 Run spec-test for eval/serialize (5 unwired) targeting tests/stream/test_hashing.py
- [ ] 1.3 Run spec-test for eval/config (6 unwired) targeting tests/stream/test_hashing.py

## 2. Medium capabilities (9-12 unwired)

- [ ] 2.1 Run spec-test for eval/vocab (9 unwired) targeting tests/stream/test_vocab.py
- [ ] 2.2 Run spec-test for eval/generator (9 unwired) targeting tests/stream/test_generator.py and tests/stream/test_registry.py
- [ ] 2.3 Run spec-test for eval/corpus (12 unwired) targeting tests/stream/test_corpus.py

## 3. Large capabilities (29-33 unwired)

- [ ] 3.1 Run spec-test for eval/events (29 unwired) targeting tests/stream/test_events.py and tests/stream/test_truth.py
- [ ] 3.2 Run spec-test for eval/render (31 unwired) targeting tests/stream/test_render.py
- [ ] 3.3 Run spec-test for eval/subject (33 unwired) targeting tests/subject/test_protocol.py, tests/subject/test_oracles.py, tests/subject/test_isolation.py, and tests/subject/test_snapshot_restore.py

## 4. Generator sub-capabilities (110 unwired total)

- [ ] 4.1 Run spec-test for eval/generators/assoc targeting tests/stream/test_assoc.py
- [ ] 4.2 Run spec-test for eval/generators/split-classify targeting tests/stream/test_split_classify.py
- [ ] 4.3 Run spec-test for eval/generators/difficulty-mix targeting tests/stream/test_difficulty_mix.py

## 5. Validation

- [ ] 5.1 Run full pytest suite, confirm all new and existing tests pass
- [ ] 5.2 Run dod-guard cover --all, confirm unwired count dropped to 0
