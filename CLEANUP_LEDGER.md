# Cleanup Ledger

State file for the incremental cleanup autopilot (see CLEANUP_PROMPT.md).
Statuses: unchecked | scanned | in-progress | fixed | blocked | clean | conflict
Baselines recorded 2026-08-27 at cycle 0.

## Pytest baseline (cycle 1)

- Full suite: 1362 passed, 2 failed, 2 skipped, 474s. (Cycle 0 recorded 487
  passed / 127s; that run must have been partial. This is the real baseline.)
- torchvision failure RESOLVED cycle 1: installed torchvision==0.26.0+cpu
  pinned to torch 2.11.0+cpu; tests/stream/test_mnist_binding.py all pass.
- device_topology failures RESOLVED cycle 2: _thaw_value in
  train/eggroll_stability.py turned the frozen empty tuple back into {} instead
  of []; on CPU-only hosts device_topology is [] so report validation failed.
  Fixed the thaw, added regression test
  test_report_with_empty_device_topology_round_trips_as_array.
- Access-violation crash MITIGATED cycle 4: threaded safetensors
  materialization in transformers killed 2 of 3 full-suite runs (Windows
  access violation in torch/storage.py __getitem__, different tests each
  time). Root conftest.py now sets HF_DEACTIVATE_ASYNC_LOAD=1 so weights
  load serially (knob verified in transformers/core_model_loading.py).
- Full suite at cycle 7: 1365 passed, 2 skipped, 0 failed, 471s.
- Full suite at cycle 10: 1365 passed, 2 skipped, 0 failed, 579s.
- Full suite at cycle 11: 1365 passed, 2 skipped, 0 failed, 474s.
- Full suite at cycle 12: 1365 passed, 2 skipped, 0 failed, 493s.
- Full suite at cycle 13: 1365 passed, 2 skipped, 0 failed, 499s.
- Full suite at cycle 14: 1365 passed, 2 skipped, 0 failed, 494s.
- Full suite at cycle 18: 1365 passed, 2 skipped, 0 failed, 485s.
- Full suite at cycle 20: 1365 passed, 2 skipped, 0 failed, 465s.
- Full suite at cycle 20 (post-fixes): 1365 passed, 2 skipped, 0 failed, 493s.
- Full suite at cycle 21 (after 5 test-file manual fixes): 1365 passed, 2 skipped, 0 failed, 481s.
- Full suite at cycle 22 (after removing sentence_transformers stubs from 3 test files): 1365 passed, 2 skipped, 0 failed, 483s.
- Full suite at cycle 23 (after 5 train/ manual fixes): 1365 passed, 2 skipped, 0 failed, 489s.
- Full suite at cycle 24 (after 5 manual-fix items incl. safetensors import hoist): 1365 passed, 2 skipped, 0 failed, 495s.
- Full suite at cycle 26 (start of cycle, post cycle-25 code fixes): 1365 passed, 2 skipped, 0 failed, 737s.
- Last full-suite run: cycle 26.

## Ruff baseline (cycle 0)

- 558 findings across 111 files with the [tool.ruff] config in pyproject.toml.
- 174 safely autofixable. Top offender: tests/test_stage0_trainability_reports.py (210).

## Openspec validation baseline (cycle 0)

`openspec validate --all --strict --no-interactive`: 27 passed, 12 failed at cycle 0; 39 passed, 0 failed at cycle 19 (all resolved).
Failing at cycle 0 (any NEW failure beyond these is a regression to fix):
- spec/eval/corpus | fixed cycle 19 | SHOULD -> MUST in "load_corpus rejects a wordless snapshot"; strict validation now passes
- spec/eval/events | fixed cycle 19 | SHOULD -> MUST in "narration and hostile flags exist for future stages"
- spec/eval/generators/assoc | fixed cycle 19 | SHOULD -> MUST in "Interleaving puts other pairs' events between a teaching and its probe"
- spec/eval/generators/split-classify | fixed cycle 19 | SHOULD -> MUST in "Class clusters stay separable"
- spec/eval/serialize | fixed cycle 19 | SHOULD -> MUST in "canonical_json forbids NaN and Infinity" + "items_to_plain output shape"
- spec/eval/subject-protocol | fixed cycle 19 | added MUST to 4 oracle requirements (Perfect-memory, Forgetful, Task-wiper, Cheater)
- spec/eval/vocab | fixed cycle 19 | SHOULD -> MUST in "VOCAB_VERSION tracks vocabulary changes"
- change/stage-1-persistent-state | fixed cycle 19 | added skip_specs: true (planning-pass change, no deltas)
- change/stage-2-fast-weight-hippocampus | fixed cycle 19 | added skip_specs: true (planning-pass change, no deltas)
- change/stage-3-idle-consolidation | fixed cycle 19 | added skip_specs: true (planning-pass change, no deltas)
- change/stage-4-bottlenecked-dual | fixed cycle 19 | added skip_specs: true (planning-pass change, no deltas)
- change/stage-5-always-on-runtime | fixed cycle 19 | added skip_specs: true (planning-pass change, no deltas)

Fixing one of these baseline failures is valid work under priority 5: run
`openspec validate <name> --type spec` (or `--type change`) to see the details,
fix the spec/change file, and check the item off this list. The stage-1
through stage-5 changes are future-stage plans; fix their format, never
their intent.

## Delete candidates

Format: path | decision (proposed/approved/rejected) | reason
- training-hysteresis.jsonl | proposed | training run output at repo root
- training-hysteresis2.jsonl | proposed | training run output at repo root
- training-hysteresis.log | proposed | training run log at repo root
- training-hysteresis2.log | proposed | training run log at repo root
- training.log | proposed | training run log at repo root
- training2.log | proposed | training run log at repo root
- training3.log | proposed | training run log at repo root
- training3.log~ | approved | editor backup file
- training4.log | proposed | training run log at repo root
- eval.log | proposed | run log at repo root
- gate.log | proposed | run log at repo root
- checkpoints/alternating-hysteresis | proposed | old checkpoint dir, confirm unused
- checkpoints/alternating-hysteresis2 | proposed | old checkpoint dir, confirm unused
- checkpoints/epoch-1.pt | proposed | loose checkpoint, confirm unused

## Spec coverage

Format: spec dir | status | scenarios total / covered | notes
- openspec/specs/codecs | scanned | 17 / 10 | decoder+encoder+narration. Uncovered: "answer returns decoded text", "answer does not mutate workspace", "observe writes encoded input to workspace", "narration field populated", "narration field absent when disabled", "off by default", "enabled via config". Conflicts/gaps: answer() runs latent_loop.run which mutates workspace in place (contradicts "answer does not mutate workspace"; overlaps core "answer runs latent loop then decoder"); narration is only a LatentCoreSubject constructor arg, eval/run/runner.py discards observe's return and never writes event.narration, no run-config toggle exists
- openspec/specs/core | scanned | 15 / 10 | latent-loop. Uncovered: "different step counts produce different compute", "flops counter reflects forward passes", "all protocol methods present", "observe runs encoder then latent loop", "answer runs latent loop then decoder"
- openspec/specs/eval/baselines | scanned | 10 / 5 | five baselines + param matching + MNIST binding. Covered: "each baseline is a valid subject" (test_baselines_smoke.py protocol+snapshot-restore), "mismatched parameters rejected" + "report includes architecture metadata" (test_param_match.py), "real MNIST features in payload" + "hash changes with data source" (test_mnist_binding.py). Uncovered: "catastrophic forgetting visible" (naive), "high accuracy on all tasks" (joint), "accuracy at initialization level" (frozen), "EWC collapses in class-incremental setting", "replay retains old task accuracy" - no test asserts any baseline's accuracy/forgetting behavior
- openspec/specs/eval/config | scanned | 11 / 11 | StreamConfig immutability + canonical hashability + frozen dataclass. All covered by tests/stream/test_hashing.py (each test carries a "covers: eval/config::" comment)
- openspec/specs/eval/corpus | scanned | 22 / 22 | resolve/load_corpus/Corpus.walk/next_span contract. All covered by tests/stream/test_corpus.py (each test carries a "covers: eval/corpus::" comment). Note: spec's wordless-snapshot requirement still says "No test exercises this path" but test_load_corpus_rejects_wordless_snapshot does - stale spec observation, not a gap
- openspec/specs/eval/events | scanned | 41 / 41 | event immutability/required fields/defaults, BoundaryKind values, Event union, StreamItem truth pairing, ProbeTruth, subject_view/harness_view, narration/hostile hooks. All covered by tests/stream/test_events.py + tests/stream/test_truth.py (each test carries a "covers: eval/events::" comment)
- openspec/specs/eval/generator | scanned | 17 / 17 | derive independence, StreamGenerator protocol, chance_rate, registry build/names, cross-process determinism. All covered by tests/stream/test_generator.py + test_registry.py + test_chance.py (each test carries a "covers: eval/generator::" comment)
- openspec/specs/eval/generators/asdiv-a | scanned | 8 / 6 | Uncovered: "Truth remains isolated" (no test renders asdiv Observe/Probe and asserts model-facing text carries no answer/item id/probe id/task id), "Selection input changes" (partial: test_selection_is_repeatable_and_binds_every_identity_input changes only revision + records; split/seed/problem_count changes untested). Covered: "Default filtered splits load" (test_partition_counts_and_measured_gate_population_are_fixed + test_asdiv_stage0_contract), "Dataset revision unavailable" (test_stage0_identity.py), "Valid row normalization", "Malformed row rejected", "Observe and probe pair", "Repeated selection"
- openspec/specs/eval/generators/assoc | scanned | 40 / 40 | all scenarios covered by tests/stream/test_assoc.py (each test carries a "covers:" comment); chance_rate measurement also in test_chance.py, cross-process determinism in test_replay.py
- openspec/specs/eval/generators/difficulty-mix | scanned | 36 / 36 | all scenarios covered by tests/stream/test_difficulty_mix.py (each test carries a "covers:" comment); chance_rate measurement also in test_chance.py
- openspec/specs/eval/generators/gsm8k | scanned | 8 / 3 | Uncovered: "stream items are valid" (no v1 stream-schema validator exists to run), "full dataset default" (no test generates with default problem_count=None). Partial: "observe events contain word problems" (test asserts non-empty text, not GSM8K word-problem content), "probe events ask for numerical answers" (test asserts non-empty query, not the numerical-answer wording), "truth on side channel only" (tests assert no-answer leak but not probe_id/task_id). Covered: "deterministic stream", "chance rate reported", "subset mode" (test_each_problem_yields_one_observe_then_one_probe, problem_count=5 -> 10 items). Note: test_gsm8k.py covers comments reference requirement names (Generator identity, chance_rate reports 0.0, deterministic replay, truth isolation, event shape) that no longer exist in this spec - stale linkage
- openspec/specs/eval/generators/split-classify | conflict | 37 / 37 | every scenario has a matching test, but spec/impl conflict: spec pins version "2" ("version is pinned" + "version is 2" scenarios) while SplitClassifyGenerator.version = "3" and tests assert "3". Secondary: spec's "Boundaries between tasks" requirement describes only TASK_SWITCH, but the impl also emits a TASK_TRAINED Boundary after each task (test_task_trained_boundaries_one_per_task, test_task_trained_sits_between_examples_and_probes), so "exactly 2 Boundary events" for num_tasks=3 is false against the impl (5 emitted). Covered via tests/stream/test_split_classify.py + test_mnist_binding.py (MNIST data binding + stream-hash scenarios)
- openspec/specs/eval/instrumentation | conflict | 3 / 2 | "default construction" (CostCounters) covered by tests/subject/test_protocol.py (test_cost_counters_defaults/steps/flops/wall_seconds/frozen; note CostCounters lives in eval/subject/__init__.py, not eval/instrumentation.py); "positive correlation" covered by tests/instrumentation/test_difficulty_correlation.py but only asserts Spearman > 0.95, not the "statistically significant" half (no p-value/significance check). Uncovered: "fields present but empty in a Stage -1 run". CONFLICT: spec requires the run record to include the 4 optional instrumentation fields, but ProbeLogEntry (eval/metrics/__init__.py) carries only cost_counters; InstrumentationFields is defined but never attached to any run record (only referenced by workspace/concept_slots.py). test_defaults_are_none checks the dataclass defaults, not a run record.
- openspec/specs/eval/metrics | unchecked | - | -
- openspec/specs/eval/package | unchecked | - | -
- openspec/specs/eval/persistence | unchecked | - | -
- openspec/specs/eval/render | unchecked | - | -
- openspec/specs/eval/reproduction-gate | unchecked | - | -
- openspec/specs/eval/serialize | unchecked | - | -
- openspec/specs/eval/stage0-gate | unchecked | - | -
- openspec/specs/eval/subject-protocol | unchecked | - | -
- openspec/specs/eval/subject/isolation | scanned | 4 / 4 | isolated_answer + CheaterOracle. All scenarios covered by tests/subject/test_isolation.py
- openspec/specs/eval/subject/oracles | unchecked | - | -
- openspec/specs/eval/subject/protocol | unchecked | - | -
- openspec/specs/eval/subject/snapshot | scanned | 2 / 2 | snapshot/restore contract. Both scenarios covered by tests/subject/test_snapshot_restore.py
- openspec/specs/eval/vocab | unchecked | - | -
- openspec/specs/train/alternating-cycle | unchecked | - | -
- openspec/specs/train/eggroll-execution | unchecked | - | -
- openspec/specs/train/stage0-training | unchecked | - | -
- openspec/specs/workspace | scanned | 8 / 8 | concept-slots. All scenarios covered

## Archived changes audit

Format: change | status | notes
- 2026-08-16-spec-test-coverage-alignment | unchecked | -
- 2026-08-16-split-compound-requirements | unchecked | -
- 2026-08-16-wire-unwired-scenarios | unchecked | -
- 2026-08-17-stage-minus-1-remainder | unchecked | -
- 2026-08-18-stage-0-latent-core | unchecked | -
- 2026-08-20-add-fixed-budget-training-cycle | unchecked | -
- 2026-08-20-throttle-alternating-progress-logging | unchecked | -
- 2026-08-22-implement-hardware-efficient-eggroll | unchecked | -
- 2026-08-22-replace-stage0-dataset-and-backbone | unchecked | -

## Files

Format: path | status | ruff findings at cycle 0 | notes
- codecs_module/__init__.py | fixed | 1 | autofix cycle 3, ruff clean
- codecs_module/decoder.py | clean | 0 | -
- codecs_module/encoder.py | clean | 0 | -
- codecs_module/narration.py | clean | 0 | -
- core/__init__.py | clean | 0 | -
- core/latent_loop.py | fixed | 1 | autofix cycle 3, ruff clean
- core/qwen_tap.py | fixed | 2 | cycle 10 manual TRY004 x2 (ValueError -> TypeError), ruff clean
- eval/__init__.py | clean | 0 | -
- eval/baselines/__init__.py | clean | 0 | -
- eval/baselines/ewc.py | fixed | 2 | cycle 10 manual: PLC0206 (use items(), no value mutation) + PLW0127 (removed no-op self-assignment); ruff clean
- eval/baselines/frozen.py | clean | 0 | -
- eval/baselines/joint.py | fixed | 1 | cycle 10 manual: PLW0127 removed no-op self-assignment; ruff clean
- eval/baselines/model.py | clean | 0 | -
- eval/baselines/naive.py | fixed | 2 | autofix cycle 3 + PLW0127 removed cycle 10; ruff clean
- eval/baselines/param_match.py | clean | 0 | -
- eval/baselines/replay.py | clean | 0 | -
- eval/gate/__init__.py | clean | 0 | -
- eval/gate/answer_scoring.py | fixed | 1 | autofix cycle 3, ruff clean
- eval/gate/cycling_sweep.py | clean | 0 | -
- eval/gate/gate_report.py | fixed | 4 | autofix cycle 3; cycle 11 manual: PLC0415 x3 (deferred imports hoisted to module refs); ruff clean
- eval/gate/latent_eval.py | fixed | 9 | cycle 11 manual: TRY004 x7 (ValueError->TypeError on isinstance checks) + PLC0415 x2 (hoisted deferred imports to module refs result_cache.*/token_cot_baseline.*); ruff clean
- eval/gate/result_cache.py | fixed | 1 | autofix cycle 3, ruff clean
- eval/gate/run_gate.py | fixed | 3 | autofix cycle 5, ruff clean (stale in-function import removed)
- eval/gate/slot_ablation.py | fixed | 4 | autofix cycle 5; cycle 11 manual: TRY004 x3 (ValueError->TypeError on isinstance checks); ruff clean
- eval/gate/token_cot_baseline.py | fixed | 3 | autofix cycle 5; cycle 11 manual: TRY004 x1 + PLC0415 x1 (deferred import -> result_cache module ref); ruff clean
- eval/instrumentation.py | clean | 0 | -
- eval/metrics/__init__.py | fixed | 2 | cycle 12 manual: E402 x2 (imports hoisted above METRICS_VERSION); ruff clean
- eval/metrics/accuracy.py | clean | 0 | -
- eval/metrics/compute.py | clean | 0 | -
- eval/metrics/first_use.py | fixed | 1 | cycle 12 manual: PLW2901 (loop var distance renamed to raw_distance); ruff clean
- eval/metrics/retention.py | clean | 0 | -
- eval/metrics/transfer.py | clean | 0 | -
- eval/run/__init__.py | clean | 0 | -
- eval/run/checkpoint.py | clean | 0 | -
- eval/run/config.py | clean | 0 | -
- eval/run/retention_policy.py | clean | 0 | -
- eval/run/runner.py | fixed | 1 | autofix cycle 5, ruff clean
- eval/stage0_identity.py | fixed | 10 | autofix cycle 5; cycle 13 manual: PLC0415 x7 (hoisted huggingface_hub/transformers/sentence_transformers/datasets imports to top) + TRY301 x2 (digest-mismatch raise extracted to _require_digest_match); ruff clean
- eval/stream/__init__.py | clean | 0 | -
- eval/stream/config.py | clean | 0 | -
- eval/stream/corpus.py | fixed | 1 | cycle 12 manual: PLW2901 (loop var line renamed to raw_line); ruff clean
- eval/stream/corpus_walk.py | clean | 0 | -
- eval/stream/events.py | fixed | 1 | cycle 11 manual: UP007 (Union -> X | Y); ruff clean
- eval/stream/generator.py | fixed | 1 | autofix cycle 5, ruff clean
- eval/stream/generators/__init__.py | clean | 0 | -
- eval/stream/generators/asdiv_a.py | fixed | 4 | autofix cycle 5; cycle 13 manual: TRY301 (is_finite raise extracted to _require_finite_decimal) + TRY004 (removed redundant bool isinstance guard; Decimal(str(bool)) already raises InvalidOperation); ruff clean
- eval/stream/generators/assoc.py | clean | 0 | -
- eval/stream/generators/difficulty_mix.py | fixed | 1 | cycle 13 manual: B905 (zip strict=True; both sequences derive from num_items); ruff clean
- eval/stream/generators/gsm8k.py | fixed | 1 | cycle 13 manual: PLC0415 (hoisted datasets import to top); ruff clean
- eval/stream/generators/split_classify.py | blocked | 2 | autofix cycle 5; 1 left: PLC0415 (torchvision import intentionally lazy - keeps the synthetic stream path free of the optional `baselines` extra; documented in _MnistData docstring)
- eval/stream/hashing.py | clean | 0 | -
- eval/stream/registry.py | fixed | 1 | autofix cycle 5, ruff clean
- eval/stream/render.py | blocked | 1 | TRY004 conflicts with spec: openspec/specs/eval/render/spec.md:102 requires render_event to raise ValueError for Idle (tests assert it); TypeError would violate the spec
- eval/stream/serialize.py | clean | 0 | -
- eval/stream/truth.py | clean | 0 | -
- eval/stream/vocab.py | clean | 0 | -
- eval/subject/__init__.py | fixed | 3 | cycle 12 manual: E402 x3 (imports hoisted above SUBJECT_PROTOCOL_VERSION); ruff clean
- eval/subject/isolation.py | clean | 0 | -
- eval/subject/oracles/__init__.py | clean | 0 | -
- eval/subject/oracles/chance.py | clean | 0 | -
- eval/subject/oracles/cheater.py | clean | 0 | -
- eval/subject/oracles/forgetful.py | fixed | 1 | cycle 12 manual: PLC0415 (format_features import hoisted to top); ruff clean
- eval/subject/oracles/perfect_memory.py | fixed | 1 | cycle 14 manual: PLC0415 (format_features import hoisted to top); ruff clean
- eval/subject/oracles/task_wiper.py | fixed | 1 | cycle 14 manual: PLC0415 (format_features import hoisted to top); ruff clean
- eval/subject/oracles/variable_compute.py | clean | 0 | -
- eval/subjects/__init__.py | clean | 0 | -
- eval/subjects/latent_core.py | fixed | 1 | autofix cycle 6, ruff clean
- main.py | clean | 0 | -
- scripts/fetch_corpus.py | blocked | 2 | cycle 14 manual: PLW2901 fixed (no-tests); 1 left: PLC0415 (zstandard import intentionally lazy - optional [corpus] extra, preserves friendly SystemExit message)
- scripts/run_gate.py | fixed | 3 | cycle 14 manual: PLW2901 (loop var renamed) + UP031 (percent format -> f-string); ruff clean (no-tests: script drives eval.run.runner.run, exercised indirectly via tests/gate/test_gate_smoke.py)
- scripts/show_gate.py | fixed | 8 | autofix cycle 6, ruff clean (F541 x8)
- scripts/show_stream.py | clean | 0 | -
- tests/baselines/__init__.py | clean | 0 | -
- tests/baselines/test_baselines_smoke.py | fixed | 1 | cycle 8, ruff clean (SIM114+SIM101 merged by hand)
- tests/baselines/test_param_match.py | clean | 0 | -
- tests/eggroll_reference.py | fixed | 2 | cycle 15 manual: UP047 (PEP 695 type param) + SIM108 (ternary); ruff clean
- tests/eggroll_stability_fixtures.py | clean | 0 | -
- tests/gate/__init__.py | clean | 0 | -
- tests/gate/test_gate_smoke.py | clean | 0 | -
- tests/gate/test_numerical_answer_scoring.py | clean | 0 | -
- tests/gate/test_stage0_gate_report.py | fixed | 1 | autofix cycle 8, ruff clean
- tests/gate/test_stage0_latent_evaluation.py | fixed | 1 | autofix cycle 8, ruff clean
- tests/gate/test_stage0_result_cache.py | fixed | 1 | autofix cycle 8, ruff clean
- tests/gate/test_stage0_run_gate.py | fixed | 2 | autofix cycle 8, ruff clean
- tests/gate/test_stage0_slot_ablation.py | clean | 0 | -
- tests/gate/test_stage0_token_baseline.py | fixed | 1 | autofix cycle 8, ruff clean
- tests/instrumentation/__init__.py | clean | 0 | -
- tests/instrumentation/test_difficulty_correlation.py | fixed | 1 | cycle 16 manual: B905 zip strict=True (rx/ry same length); ruff clean
- tests/instrumentation/test_instrumentation_fields.py | clean | 0 | -
- tests/metrics/__init__.py | clean | 0 | -
- tests/metrics/test_accuracy.py | clean | 0 | -
- tests/metrics/test_first_use.py | clean | 0 | -
- tests/metrics/test_oracle_integration.py | fixed | 1 | cycle 16 manual: B905 zip strict=True (equal lengths); ruff clean
- tests/metrics/test_retention.py | clean | 0 | -
- tests/metrics/test_transfer.py | clean | 0 | -
- tests/run/test_checkpoint.py | fixed | 1 | cycle 16 manual: SIM117 combined nested with (patch + pytest.raises); ruff clean
- tests/run/test_config.py | fixed | 1 | autofix cycle 8, ruff clean
- tests/run/test_resume.py | fixed | 1 | autofix cycle 8, ruff clean
- tests/run/test_retention_policy.py | clean | 0 | -
- tests/run/test_runner.py | fixed | 1 | autofix cycle 8, ruff clean
- tests/stream/test_asdiv_a.py | fixed | 1 | autofix cycle 8, ruff clean
- tests/stream/test_assoc.py | fixed | 4 | cycle 16 manual: B905 zip strict=True, SIM102 combined nested if, ISC004 x2 parenthesized implicit concat; ruff clean
- tests/stream/test_chance.py | clean | 0 | -
- tests/stream/test_corpus.py | fixed | 1 | cycle 16 manual: RUF043 match=re.escape(...); ruff clean
- tests/stream/test_difficulty_mix.py | clean | 0 | -
- tests/stream/test_events.py | fixed | 1 | autofix cycle 9, ruff clean
- tests/stream/test_generator.py | clean | 0 | -
- tests/stream/test_gsm8k.py | clean | 0 | -
- tests/stream/test_hashing.py | fixed | 3 | autofix cycle 9; cycle 17 manual PLC0415 x2 (hoisted events/truth imports to top); ruff clean
- tests/stream/test_mnist_binding.py | clean | 0 | -
- tests/stream/test_package.py | clean | 0 | -
- tests/stream/test_registry.py | fixed | 1 | cycle 17 manual PLW1510 (explicit check=False; test asserts returncode itself); ruff clean
- tests/stream/test_render.py | clean | 0 | -
- tests/stream/test_replay.py | clean | 0 | -
- tests/stream/test_split_classify.py | fixed | 5 | autofix cycle 9; cycle 17 manual B905 x2 (zip strict=True) + PLC0415 x2 (hoisted math/_class_center/FEATURE_NOISE_STD imports to top); ruff clean
- tests/stream/test_truth.py | fixed | 7 | cycle 17 manual RUF007 (itertools.pairwise, also clears B905) + PLC0415 x5 (hoisted Path/StreamConfig/generator imports to top); ruff clean
- tests/stream/test_vocab.py | fixed | 2 | autofix cycle 9; cycle 17 manual PLC0415 (hoisted VOCAB_VERSION import to top); ruff clean
- tests/subject/test_isolation.py | clean | 0 | -
- tests/subject/test_oracles.py | fixed | 2 | cycle 18 manual: B007 (value->_value) + C416 (dict comprehension -> dict()); ruff clean
- tests/subject/test_protocol.py | fixed | 1 | cycle 18 manual: PLC0415 (import abc hoisted to top); ruff clean
- tests/subject/test_snapshot_restore.py | clean | 0 | -
- tests/test_alternating_checkpoint.py | fixed | 5 | cycle 21 manual: F841 (dropped unused `schedule =`); ruff clean
- tests/test_alternating_config.py | fixed | 1 | cycle 18 manual: RUF043 (match pattern -> raw string); ruff clean
- tests/test_alternating_evaluation.py | fixed | 2 | cycle 21 manual: B905 (zip strict=True); ruff clean
- tests/test_alternating_scheduler.py | clean | 0 | -
- tests/test_answer_objective.py | clean | 0 | -
- tests/test_asdiv_stage0_contract.py | clean | 0 | -
- tests/test_benchmark_eggroll.py | fixed | 1 | cycle 18 manual: RUF043 (match -> re.escape); ruff clean
- tests/test_codecs_freezing.py | clean | 0 | -
- tests/test_eggroll_factorized.py | clean | 0 | -
- tests/test_eggroll_perturbations.py | clean | 0 | -
- tests/test_eggroll_reference.py | clean | 0 | -
- tests/test_eggroll_resume_equivalence.py | fixed | 2 | autofix cycle 9, ruff clean
- tests/test_eggroll_stability.py | fixed | 2 | ruff clean cycle 2; regression test added
- tests/test_eggroll_step_equivalence.py | fixed | 1 | autofix cycle 9, ruff clean
- tests/test_eggroll_training.py | fixed | 10 | cycle 22 manual: removed sentence_transformers stub (E402 x6; import-speed opt only, tests don't instantiate it) + hoisted trainer_module import + dropped redundant AsdivRecord import (PLC0415 x2); ruff clean
- tests/test_eggroll_updates.py | clean | 0 | -
- tests/test_eggroll_workflow_guards.py | fixed | 1 | autofix cycle 9, ruff clean
- tests/test_frozen_qwen_backbone.py | clean | 0 | -
- tests/test_gradient_training.py | fixed | 3 | cycle 22 manual: removed sentence_transformers stub (E402 x2); ruff clean
- tests/test_latent_eval.py | clean | 0 | -
- tests/test_latent_loop.py | clean | 0 | -
- tests/test_narration.py | fixed | 1 | autofix (prior cycle-10 commit), ruff clean
- tests/test_plot_training.py | clean | 0 | -
- tests/test_probe_isolation.py | clean | 0 | -
- tests/test_qwen_latent_loop.py | fixed | 1 | autofix (prior cycle-10 commit), ruff clean
- tests/test_qwen_tap_equivalence.py | fixed | 1 | autofix (prior cycle-10 commit), ruff clean
- tests/test_run_alternating.py | fixed | 2 | cycle 21 manual: B007 (loop var _expected_message); ruff clean
- tests/test_run_eggroll.py | fixed | 2 | autofix (prior cycle-10 commit), ruff clean
- tests/test_run_eggroll_stability.py | clean | 0 | -
- tests/test_run_stage0_trainability.py | fixed | 10 | cycle 22 manual: PLC0415 x7 (hoisted train.stage0_trainability imports to top); ruff clean
- tests/test_run_training.py | fixed | 1 | PLC0415 run_training import hoisted, ruff clean
- tests/test_search_alignment_weight.py | fixed | 1 | autofix (prior cycle-10 commit), ruff clean
- tests/test_stage0_checkpoint_container.py | fixed | 9 | cycle 24 manual: PLW0108 x2 (lambda -> calls.append) + SIM117 x3 (nested with -> single with); ruff clean
- tests/test_stage0_checkpoint_resume.py | fixed | 2 | cycle 21 manual: PLW0108 (lambda -> exposed.append); ruff clean
- tests/test_stage0_checkpoint_schema.py | fixed | 4 | autofix cycle 10, ruff clean
- tests/test_stage0_cli_runtime_order.py | clean | 0 | -
- tests/test_stage0_dataset_contract.py | fixed | 1 | autofix cycle 10, ruff clean
- tests/test_stage0_identity.py | fixed | 1 | autofix cycle 10, ruff clean
- tests/test_stage0_shapes.py | fixed | 1 | autofix cycle 10, ruff clean
- tests/test_stage0_trainability_identity.py | fixed | 13 | cycle 22 manual: PLC0415 x13 (hoisted train.stage0_trainability imports to top); ruff clean
- tests/test_stage0_trainability_reports.py | fixed | 210 | cycle 26 manual: hoisted 142 in-function imports to top (PLC0415 x138 incl. 4 plain `import inspect`/`import math`), C408 x25 (tuple() -> ()), F841 (removed unused held_out_record_ids, renamed unpack to _held_out_64_ids), RUF059 (status, _), SIM102 (combined nested if), SIM222 x2 (vacuous assert -> True), BLE001 x3 (dropped try/except Exception -> pytest.fail, exceptions now propagate); ruff clean
- tests/test_standalone_checkpoint.py | fixed | 7 | cycle 21 manual: B905 (zip strict=True in _same_state); ruff clean
- tests/test_trainability_types_verification.py | clean | 0 | -
- tests/test_training_results.py | clean | 0 | -
- tests/test_training_state.py | fixed | 4 | cycle 22 manual: removed sentence_transformers stub (E402 x3); ruff clean
- tests/test_vicreg.py | clean | 0 | -
- tests/test_workspace.py | clean | 0 | -
- train/__init__.py | clean | 0 | -
- train/alternating_checkpoint.py | fixed | 4 | cycle 23 manual: B905 (zip strict=True) + PLC0415 (hoisted ALLOWED_MODEL_PARAMETER_PATHS import); ruff clean
- train/alternating_config.py | fixed | 1 | autofix cycle 6, ruff clean
- train/alternating_evaluation.py | fixed | 2 | B905 zip strict=True + PLC0415 SlotDecoder import hoisted, ruff clean
- train/alternating_scheduler.py | fixed | 3 | cycle 23 manual: UP046 x2 (PEP 695 type params on EvaluationRecord, TrainingEngine, PhaseEvaluator, VarianceHysteresisScheduler; dropped Generic/TypeVar imports); ruff clean
- train/answer_objective.py | fixed | 3 | cycle 23 manual: TRY004 x2 (callable type guards raise TypeError, not ValueError); ruff clean
- train/benchmark_eggroll.py | clean | 0 | -
- train/eggroll_factorized.py | clean | 0 | -
- train/eggroll_perturbations.py | clean | 0 | -
- train/eggroll_stability.py | fixed | 6 | ruff clean cycle 2; empty-tuple thaw bug fixed
- train/eggroll_stability_evaluation.py | clean | 0 | -
- train/eggroll_stability_guard.py | fixed | 1 | cycle 18 manual: ISC004 (parenthesized implicit concat inside tuple arg); ruff clean
- train/eggroll_trainer.py | fixed | 7 | cycle 23 manual: E731 (lambda -> def) + PLC0415 (removed redundant deferred AsdivRecord import, already top-level); ruff clean
- train/eggroll_updates.py | fixed | 1 | cycle 19 manual: SIM108 ternary conversion; ruff clean
- train/plot_training.py | fixed | 8 | cycle 24 manual: TRY004 x2 (type checks -> TypeError), SIM108 (ternary), E731 x2 (lambda -> def), RUF046 (drop redundant int()); ruff clean
- train/run_alternating.py | fixed | 4 | cycle 23 manual: B905 (zip strict=True) + PLC0415 (hoisted numpy import to top); ruff clean
- train/run_eggroll.py | fixed | 9 | cycle 24 manual: B023 x8 (bound _on_step loop captures as default args, same as run_training); ruff clean
- train/run_eggroll_stability.py | fixed | 1 | autofix cycle 7, ruff clean
- train/run_stage0_trainability.py | blocked | 27 | cycle 24 manual: PLC0415 x2 (hoisted time import), F841 (removed dead start_time), BLE001 x3 (removed unreachable try/except; narrowed write/investigation excepts to OSError); 1 left: TRY004 at _validate_args root-object check - conflicts with test_command_rejects_non_object_stability_report which asserts ValueError (a data-validation error, not a type error)
- train/run_training.py | fixed | 8 | cycle 22 manual: B023 x8 (bound _on_step loop captures as default args); ruff clean
- train/search_alignment_weight.py | fixed | 5 | SIM102 combined nested if, ruff clean
- train/stage0_checkpoint.py | fixed | 6 | cycle 24 manual: SIM102 x2 (nested if -> and), SIM105 (contextlib.suppress), PLC0415 (hoisted safetensors.torch.load); ruff clean
- train/stage0_data.py | fixed | 1 | autofix cycle 7, ruff clean
- train/stage0_trainability.py | blocked | 27 | cycle 25 manual: 20 of 24 fixed (SIM102, C416, B904 x2, PLC0415 x5, RUF059 x2, B007, C408, RUF015, TRY300, SIM105, BLE001 x3 removed/narrowed); 4 left: BLE001 x4 - deliberate catch-all graceful degradation (evaluate_single_record_loss -> NaN, run_overfit_attempt train_step + outer fatal, _compute_state_hash -> "")
- train/standalone_checkpoint.py | fixed | 4 | B905 zip strict=True x2, ruff clean
- train/trainer.py | fixed | 5 | autofix cycle 7, ruff clean (4 stale imports removed)
- train/training_results.py | clean | 0 | -
- train/training_state.py | clean | 0 | -
- train/vicreg.py | fixed | 1 | B905 zip strict=True, ruff clean
- workspace/__init__.py | fixed | 1 | autofix cycle 7, ruff clean
- workspace/concept_slots.py | clean | 0 | -

## Additional notes:
- tests taking >10 minutes is not acceptable, optimize performance to get it down to preferably <1 minute, but at least <5

## Cycle log

Format: cycle N | item | outcome
- cycle 0 | setup: ruff config, prompt, ledger | baselines recorded
- cycle 1 | install torchvision 0.26.0+cpu, rerun full suite | mnist test fixed; found 2 real failures in test_search_alignment_weight.py (device_topology)
- cycle 2 | fix empty-tuple thaw bug in train/eggroll_stability.py + clear its ruff findings | 2 failing tests now pass; regression test added; 94 related tests green
- cycle 3 | ruff autofix batch: codecs_module/__init__, core/latent_loop, eval/baselines/naive, eval/gate/{answer_scoring,gate_report,result_cache} | 6 findings fixed, 177 covering tests pass
- cycle 4 | add conftest.py with HF_DEACTIVATE_ASYNC_LOAD=1 | full suite green: 1365 passed, 2 skipped, 511s
- cycle 5 | ruff autofix batch: eval/gate/{run_gate,slot_ablation,token_cot_baseline}, eval/run/runner, eval/stage0_identity, eval/stream/{generator,registry}, eval/stream/generators/{asdiv_a,split_classify} | 12 findings fixed, 129 covering tests pass
- cycle 6 | ruff autofix batch: eval/subjects/latent_core, scripts/{run_gate,show_gate}, train/{alternating_checkpoint,alternating_config,alternating_scheduler,answer_objective,eggroll_trainer,plot_training,run_alternating} | 26 findings fixed, 107 covering tests pass
- cycle 7 | ruff autofix batch: train/{run_eggroll,run_eggroll_stability,run_stage0_trainability,search_alignment_weight,stage0_checkpoint,stage0_data,stage0_trainability,standalone_checkpoint,trainer}, workspace/__init__ | 39 findings fixed, 429 covering tests pass
- cycle 8 | ruff autofix batch: tests/baselines/test_baselines_smoke, tests/gate/{gate_report,latent_evaluation,result_cache,run_gate,token_baseline}, tests/run/{config,resume,runner}, tests/stream/test_asdiv_a | 12 findings fixed, all 10 files clean, 185 tests pass
- cycle 9 | ruff autofix batch: tests/stream/{events,hashing,split_classify,vocab}, tests/test_{alternating_checkpoint,alternating_evaluation,eggroll_resume_equivalence,eggroll_step_equivalence,eggroll_training,eggroll_workflow_guards} | 15 findings fixed, 176 tests pass
- cycle 10 | ruff autofix batch: tests/test_stage0_{checkpoint_schema,dataset_contract,identity,shapes,trainability_reports}, tests/test_{standalone_checkpoint,training_state} | 49 findings fixed, 4 files clean, 302 covering tests pass; full suite re-run: 1365 passed, 2 skipped
- cycle 10 | manual TRY004 fixes in core/qwen_tap.py | ruff clean, 22 covering tests pass
- cycle 10 | manual PLC0206+PLW0127 fixes in eval/baselines/ewc.py | ruff clean, 29 covering tests pass
- cycle 10 | manual PLW0127 fix in eval/baselines/joint.py | ruff clean, 20 covering tests pass
- cycle 10 | manual PLW0127 fix in eval/baselines/naive.py | ruff clean, 20 covering tests pass
- cycle 10 | healed ledger drift: prior commit 4f13aff autofixed 10 test files without a ledger entry; verified current ruff state and corrected those 10 rows
- cycle 11 | full suite verification (manual fixes landed after cycle-10 run) | 1365 passed, 2 skipped, 0 failed, 474s
- cycle 11 | manual fixes in eval/gate/latent_eval.py: TRY004 x7 + PLC0415 x2 | ruff clean, 28 covering tests pass
- cycle 11 | manual fixes in eval/gate/slot_ablation.py: TRY004 x3 | ruff clean, 13 covering tests pass
- cycle 11 | manual fixes in eval/gate/token_cot_baseline.py: TRY004 x1 + PLC0415 x1 | ruff clean, 22 covering tests pass
- cycle 11 | manual fixes in eval/gate/gate_report.py: PLC0415 x3 | ruff clean, 22 covering tests pass
- cycle 11 | manual fix in eval/stream/events.py: UP007 | ruff clean, 38 covering tests pass
- cycle 11 | end-of-cycle full suite (after 5 files incl. import hoisting) | 1365 passed, 2 skipped, 0 failed, 480s
- cycle 12 | manual E402 x2 in eval/metrics/__init__.py | ruff clean, 78 covering tests pass
- cycle 12 | manual E402 x3 in eval/subject/__init__.py | ruff clean, 78 covering tests pass
- cycle 12 | manual PLW2901 in eval/stream/corpus.py | ruff clean, 78 covering tests pass
- cycle 12 | manual PLW2901 in eval/metrics/first_use.py | ruff clean, 78 covering tests pass
- cycle 12 | manual PLC0415 in eval/subject/oracles/forgetful.py | ruff clean, 17 covering tests pass
- cycle 12 | end-of-cycle full suite (5 files incl. 2 __init__ import reorders) | 1365 passed, 2 skipped, 0 failed, 493s
- cycle 13 | manual PLC0415 x7 + TRY301 x2 in eval/stage0_identity.py | ruff clean, 273 covering tests pass (1 skipped)
- cycle 13 | manual TRY301 + TRY004 in eval/stream/generators/asdiv_a.py | ruff clean, 18 covering tests pass
- cycle 13 | manual B905 in eval/stream/generators/difficulty_mix.py | ruff clean, 30 covering tests pass
- cycle 13 | manual PLC0415 in eval/stream/generators/gsm8k.py | ruff clean, 17 covering tests pass
- cycle 13 | split_classify.py PLC0415 marked blocked | torchvision import is an intentional optional-dependency boundary (pyproject `baselines` extra); hoisting would break the documented synthetic-only import path
- cycle 13 | end-of-cycle full suite (4 source files incl. stage0_identity import hoisting) | 1365 passed, 2 skipped, 0 failed, 499s
- cycle 14 | eval/stream/render.py TRY004 marked blocked | spec render/spec.md:102 requires ValueError for Idle; TypeError would violate spec + break tests
- cycle 14 | manual PLC0415 in eval/subject/oracles/perfect_memory.py | ruff clean, 17 covering tests pass
- cycle 14 | manual PLC0415 in eval/subject/oracles/task_wiper.py | ruff clean, 17 covering tests pass
- cycle 14 | scripts/fetch_corpus.py PLW2901 fixed; PLC0415 marked blocked | zstandard import is an optional [corpus] extra; hoisting would drop the friendly SystemExit message
- cycle 14 | manual PLW2901 + UP031 in scripts/run_gate.py | ruff clean (no direct script test; py_compile ok)
- cycle 14 | end-of-cycle full suite (4 source files: 2 oracle import hoists + 2 script fixes) | 1365 passed, 2 skipped, 0 failed, 494s
- cycle 15 | manual UP047 (PEP 695 type param) + SIM108 (ternary) in tests/eggroll_reference.py | ruff clean, 27 covering tests pass (1 skipped)
- cycle 16 | manual B905 (zip strict=True) in tests/instrumentation/test_difficulty_correlation.py | ruff clean, 1 test passes
- cycle 16 | manual B905 (zip strict=True) in tests/metrics/test_oracle_integration.py | ruff clean, 5 tests pass
- cycle 16 | manual SIM117 (combine nested with) in tests/run/test_checkpoint.py | ruff clean, 7 tests pass
- cycle 16 | manual B905 + SIM102 + ISC004 x2 in tests/stream/test_assoc.py | ruff clean, 49 tests pass
- cycle 16 | manual RUF043 (re.escape) in tests/stream/test_corpus.py | ruff clean, 24 tests pass
- cycle 17 | manual PLC0415 x2 (hoisted events/truth imports to top) in tests/stream/test_hashing.py | ruff clean, 33 tests pass
- cycle 17 | manual PLW1510 (explicit check=False) in tests/stream/test_registry.py | ruff clean, 18 tests pass
- cycle 17 | manual B905 x2 + PLC0415 x2 in tests/stream/test_split_classify.py | ruff clean, 36 tests pass
- cycle 17 | manual RUF007 + PLC0415 x5 in tests/stream/test_truth.py | ruff clean, 27 tests pass
- cycle 17 | manual PLC0415 in tests/stream/test_vocab.py | ruff clean, 23 tests pass
- cycle 18 | full suite verification (code fixes landed since cycle 14) | 1365 passed, 2 skipped, 0 failed, 485s
- cycle 18 | manual B007 + C416 in tests/subject/test_oracles.py | ruff clean, 17 tests pass
- cycle 18 | manual PLC0415 (import abc hoisted) in tests/subject/test_protocol.py | ruff clean, 12 tests pass
- cycle 18 | manual RUF043 (raw match pattern) in tests/test_alternating_config.py | ruff clean, 12 tests pass
- cycle 18 | manual RUF043 (match -> re.escape) in tests/test_benchmark_eggroll.py | ruff clean, 7 tests pass
- cycle 18 | manual ISC004 (parenthesized implicit concat) in train/eggroll_stability_guard.py | ruff clean, 16 covering tests pass
- cycle 19 | manual SIM108 (ternary) in train/eggroll_updates.py | ruff clean, 5 covering tests pass
- cycle 19 | spec validation: eval/corpus SHOULD -> MUST (wordless-snapshot requirement) | openspec validate --strict now passes
- cycle 19 | spec validation batch: SHOULD -> MUST in eval/{events,generators/assoc,generators/split-classify,serialize,subject-protocol,vocab} | all strict-valid
- cycle 19 | spec validation batch: skip_specs: true for change/stage-1..5 (planning-pass changes) | openspec validate --all --strict: 39 passed, 0 failed
- cycle 19 | spec coverage scan: openspec/specs/core (latent-loop) | 15 scenarios, 10 covered, 5 uncovered (recorded above)
- cycle 19 | spec coverage scan: openspec/specs/workspace (concept-slots) | 8 scenarios, 8 covered
- cycle 20 | full suite + repo-wide ruff re-scan (rule 11) | 1365 passed, 2 skipped; 291 findings across 31 files, no clean/fixed drift
- cycle 20 | manual B905 (zip strict=True) in train/vicreg.py | ruff clean, 3 tests pass
- cycle 20 | manual B905 + PLC0415 (hoisted SlotDecoder import) in train/alternating_evaluation.py | ruff clean, 7 tests pass
- cycle 20 | manual PLC0415 (hoisted run_training import) in tests/test_run_training.py | ruff clean, 2 tests pass
- cycle 20 | manual SIM102 (combined nested if) in train/search_alignment_weight.py | ruff clean, 16 tests pass
- cycle 20 | manual B905 (zip strict=True) x2 in train/standalone_checkpoint.py | ruff clean, 9 tests pass
- cycle 20 | end-of-cycle full suite (5 files fixed) | 1365 passed, 2 skipped, 0 failed, 493s
- cycle 21 | manual F841 (dropped unused `schedule =`) in tests/test_alternating_checkpoint.py | ruff clean, 21 tests pass
- cycle 21 | manual B905 (zip strict=True) in tests/test_alternating_evaluation.py | ruff clean, 7 tests pass
- cycle 21 | manual B007 (loop var _expected_message) in tests/test_run_alternating.py | ruff clean, 30 tests pass
- cycle 21 | manual B905 (zip strict=True in _same_state) in tests/test_standalone_checkpoint.py | ruff clean, 9 tests pass
- cycle 21 | manual PLW0108 (lambda -> exposed.append) in tests/test_stage0_checkpoint_resume.py | ruff clean, 9 tests pass
- cycle 21 | end-of-cycle full suite (5 test files fixed) | 1365 passed, 2 skipped, 0 failed, 481s
- cycle 22 | manual PLC0415 x13 (hoisted imports to top) in tests/test_stage0_trainability_identity.py | ruff clean, 13 tests pass
- cycle 22 | manual B023 x8 (closure loop-capture -> default args) in train/run_training.py | ruff clean, 2 tests pass
- cycle 22 | removed sentence_transformers stub from tests/test_{eggroll_training,gradient_training,training_state}.py (E402 x11) + PLC0415 x2 in eggroll_training | ruff clean, full suite 1365 passed, 2 skipped, 0 failed, 483s
- cycle 22 | manual PLC0415 x7 (hoisted stage0_trainability imports to top) in tests/test_run_stage0_trainability.py | ruff clean, 23 tests pass
- cycle 23 | manual UP046 x2 (PEP 695 type params) in train/alternating_scheduler.py | ruff clean, 7 tests pass
- cycle 23 | manual TRY004 x2 (callable guards -> TypeError) in train/answer_objective.py | ruff clean, 10 tests pass
- cycle 23 | manual B905 + PLC0415 in train/alternating_checkpoint.py | ruff clean, 21 tests pass
- cycle 23 | manual B905 + PLC0415 (numpy hoist) in train/run_alternating.py | ruff clean, 30 tests pass
- cycle 23 | manual E731 + PLC0415 in train/eggroll_trainer.py | ruff clean, 11 covering tests pass (test_eggroll_training + test_run_eggroll + test_eggroll_step_equivalence)
- cycle 23 | end-of-cycle full suite (5 train/ files fixed) | 1365 passed, 2 skipped, 0 failed, 489s
- cycle 24 | manual PLW0108 x2 + SIM117 x3 in tests/test_stage0_checkpoint_container.py | ruff clean, 34 tests pass
- cycle 24 | manual SIM102 x2 + SIM105 + PLC0415 in train/stage0_checkpoint.py | ruff clean, 140 checkpoint tests pass
- cycle 24 | manual TRY004 x2 + SIM108 + E731 x2 + RUF046 in train/plot_training.py | ruff clean, 3 tests pass
- cycle 24 | manual B023 x8 in train/run_eggroll.py | ruff clean, 10 tests pass
- cycle 24 | manual PLC0415 x2 + F841 + BLE001 x3 in train/run_stage0_trainability.py | 1 left: TRY004 blocked (test asserts ValueError); 23 tests pass
- cycle 24 | end-of-cycle full suite (5 items incl. stage0_checkpoint safetensors hoist) | 1365 passed, 2 skipped, 0 failed, 495s
- cycle 25 | manual fixes in train/stage0_trainability.py: 20 of 24 findings fixed | ruff: 4 BLE001 left (blocked); 204 covering tests pass (identity 12 + types 2 + run_stage0 23 + reports 156; reports 377s)
- cycle 25 | spec coverage scan: openspec/specs/codecs (decoder+encoder+narration) | 17 scenarios, 10 covered, 7 uncovered (recorded above)
- cycle 26 | full suite verification (cycle 25 landed stage0_trainability code fixes) | 1365 passed, 2 skipped, 0 failed, 737s
- cycle 26 | manual fixes in tests/test_stage0_trainability_reports.py: hoisted 138 PLC0415 + C408 x25 + F841 + RUF059 + SIM102 + SIM222 x2 + BLE001 x3 | ruff clean, 156 tests pass (538s)
- cycle 26 | finding: 3 recalibration tests are vacuous placeholders (test_trainability_cannot_authorize_full_gradient/eggroll, test_trainability_recalibration_eligibility_not_full_training) - docstrings claim authorization limits compute_recalibration_eligibility never models; SIM222 exposed the vacuous asserts, fixed to `assert True` to preserve behavior, real assertions left for a later item
- cycle 26 | repo-wide ruff re-scan (rule 11: no unchecked/scanned rows left) | 8 findings across 5 files, all in ledger-blocked rows with documented reasons; no clean/fixed drift
- cycle 26 | spec coverage scan: openspec/specs/eval/subject/snapshot | 2 scenarios, 2 covered (both via tests/subject/test_snapshot_restore.py). Also split the coarse eval/train spec-coverage rows into 27 per-capability rows
- cycle 26 | spec coverage scan: openspec/specs/eval/subject/isolation | 4 scenarios, 4 covered (all via tests/subject/test_isolation.py)
- cycle 27 | spec coverage scan: openspec/specs/eval/baselines | 10 scenarios, 5 covered, 5 uncovered (accuracy/forgetting behaviors of the five baselines have no test)
- cycle 27 | spec coverage scan: openspec/specs/eval/config | 11 scenarios, 11 covered (all via tests/stream/test_hashing.py)
- cycle 27 | spec coverage scan: openspec/specs/eval/corpus | 22 scenarios, 22 covered (all via tests/stream/test_corpus.py)
- cycle 27 | spec coverage scan: openspec/specs/eval/events | 41 scenarios, 41 covered (all via tests/stream/test_events.py + test_truth.py)
- cycle 27 | spec coverage scan: openspec/specs/eval/generator | 17 scenarios, 17 covered (all via tests/stream/test_generator.py + test_registry.py + test_chance.py)
- cycle 28 | spec coverage scan: openspec/specs/eval/generators/asdiv-a | 8 scenarios, 6 covered, 2 uncovered/partial (recorded above)
- cycle 28 | spec coverage scan: openspec/specs/eval/generators/assoc | 40 scenarios, 40 covered (all via tests/stream/test_assoc.py)
- cycle 28 | spec coverage scan: openspec/specs/eval/generators/difficulty-mix | 36 scenarios, 36 covered (all via tests/stream/test_difficulty_mix.py)
- cycle 28 | spec coverage scan: openspec/specs/eval/generators/gsm8k | 8 scenarios, 3 covered, 2 uncovered, 3 partial (recorded above)
- cycle 28 | spec coverage scan: openspec/specs/eval/generators/split-classify | 37 scenarios, 37 tested, CONFLICT: version "2" in spec vs "3" in code+tests (recorded above)
- cycle 29 | spec coverage scan: openspec/specs/eval/instrumentation | 3 scenarios, 2 covered, 1 uncovered; CONFLICT: run record (ProbeLogEntry) lacks the 4 optional instrumentation fields the spec requires (recorded above)
