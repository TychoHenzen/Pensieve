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
- Last full-suite run: cycle 14.

## Ruff baseline (cycle 0)

- 558 findings across 111 files with the [tool.ruff] config in pyproject.toml.
- 174 safely autofixable. Top offender: tests/test_stage0_trainability_reports.py (210).

## Openspec validation baseline (cycle 0)

`openspec validate --all --strict --no-interactive`: 27 passed, 12 failed.
Failing at cycle 0 (any NEW failure beyond these is a regression to fix):
- spec/eval/corpus
- spec/eval/events
- spec/eval/generators/assoc
- spec/eval/generators/split-classify
- spec/eval/serialize
- spec/eval/subject-protocol
- spec/eval/vocab
- change/stage-1-persistent-state
- change/stage-2-fast-weight-hippocampus
- change/stage-3-idle-consolidation
- change/stage-4-bottlenecked-dual
- change/stage-5-always-on-runtime

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
- openspec/specs/codecs | unchecked | - | -
- openspec/specs/core | unchecked | - | -
- openspec/specs/eval | unchecked | - | -
- openspec/specs/train | unchecked | - | -
- openspec/specs/workspace | unchecked | - | -

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
- tests/stream/test_assoc.py | unchecked | 4 | -
- tests/stream/test_chance.py | clean | 0 | -
- tests/stream/test_corpus.py | unchecked | 1 | -
- tests/stream/test_difficulty_mix.py | clean | 0 | -
- tests/stream/test_events.py | fixed | 1 | autofix cycle 9, ruff clean
- tests/stream/test_generator.py | clean | 0 | -
- tests/stream/test_gsm8k.py | clean | 0 | -
- tests/stream/test_hashing.py | scanned | 3 | autofix cycle 9; 2 left: PLC0415
- tests/stream/test_mnist_binding.py | clean | 0 | -
- tests/stream/test_package.py | clean | 0 | -
- tests/stream/test_registry.py | unchecked | 1 | -
- tests/stream/test_render.py | clean | 0 | -
- tests/stream/test_replay.py | clean | 0 | -
- tests/stream/test_split_classify.py | scanned | 5 | autofix cycle 9; 4 left: B905 x2, PLC0415 x2
- tests/stream/test_truth.py | unchecked | 7 | -
- tests/stream/test_vocab.py | scanned | 2 | autofix cycle 9; 1 left: PLC0415
- tests/subject/test_isolation.py | clean | 0 | -
- tests/subject/test_oracles.py | unchecked | 2 | -
- tests/subject/test_protocol.py | unchecked | 1 | -
- tests/subject/test_snapshot_restore.py | clean | 0 | -
- tests/test_alternating_checkpoint.py | scanned | 5 | autofix cycle 9; 1 left: F841
- tests/test_alternating_config.py | unchecked | 1 | -
- tests/test_alternating_evaluation.py | scanned | 2 | autofix cycle 9; 1 left: B905
- tests/test_alternating_scheduler.py | clean | 0 | -
- tests/test_answer_objective.py | clean | 0 | -
- tests/test_asdiv_stage0_contract.py | clean | 0 | -
- tests/test_benchmark_eggroll.py | unchecked | 1 | -
- tests/test_codecs_freezing.py | clean | 0 | -
- tests/test_eggroll_factorized.py | clean | 0 | -
- tests/test_eggroll_perturbations.py | clean | 0 | -
- tests/test_eggroll_reference.py | clean | 0 | -
- tests/test_eggroll_resume_equivalence.py | fixed | 2 | autofix cycle 9, ruff clean
- tests/test_eggroll_stability.py | fixed | 2 | ruff clean cycle 2; regression test added
- tests/test_eggroll_step_equivalence.py | fixed | 1 | autofix cycle 9, ruff clean
- tests/test_eggroll_training.py | scanned | 10 | autofix cycle 9; 8 left: E402 x6 (torch guard at top), PLC0415 x2
- tests/test_eggroll_updates.py | clean | 0 | -
- tests/test_eggroll_workflow_guards.py | fixed | 1 | autofix cycle 9, ruff clean
- tests/test_frozen_qwen_backbone.py | clean | 0 | -
- tests/test_gradient_training.py | scanned | 3 | autofix (prior cycle-10 commit 4f13aff); 2 left: E402 x2
- tests/test_latent_eval.py | clean | 0 | -
- tests/test_latent_loop.py | clean | 0 | -
- tests/test_narration.py | fixed | 1 | autofix (prior cycle-10 commit), ruff clean
- tests/test_plot_training.py | clean | 0 | -
- tests/test_probe_isolation.py | clean | 0 | -
- tests/test_qwen_latent_loop.py | fixed | 1 | autofix (prior cycle-10 commit), ruff clean
- tests/test_qwen_tap_equivalence.py | fixed | 1 | autofix (prior cycle-10 commit), ruff clean
- tests/test_run_alternating.py | scanned | 2 | autofix (prior cycle-10 commit); 1 left: B007
- tests/test_run_eggroll.py | fixed | 2 | autofix (prior cycle-10 commit), ruff clean
- tests/test_run_eggroll_stability.py | clean | 0 | -
- tests/test_run_stage0_trainability.py | scanned | 10 | autofix (prior cycle-10 commit); 7 left: PLC0415 x7
- tests/test_run_training.py | unchecked | 1 | -
- tests/test_search_alignment_weight.py | fixed | 1 | autofix (prior cycle-10 commit), ruff clean
- tests/test_stage0_checkpoint_container.py | scanned | 9 | autofix (prior cycle-10 commit); 5 left: PLW0108 x2, SIM117 x3
- tests/test_stage0_checkpoint_resume.py | scanned | 2 | autofix (prior cycle-10 commit); 1 left: PLW0108
- tests/test_stage0_checkpoint_schema.py | fixed | 4 | autofix cycle 10, ruff clean
- tests/test_stage0_cli_runtime_order.py | clean | 0 | -
- tests/test_stage0_dataset_contract.py | fixed | 1 | autofix cycle 10, ruff clean
- tests/test_stage0_identity.py | fixed | 1 | autofix cycle 10, ruff clean
- tests/test_stage0_shapes.py | fixed | 1 | autofix cycle 10, ruff clean
- tests/test_stage0_trainability_identity.py | unchecked | 13 | -
- tests/test_stage0_trainability_reports.py | scanned | 210 | autofix cycle 10; 171 left: PLC0415 x138, C408 x25, BLE001 x3, SIM222 x2, F841 x1, RUF059 x1, SIM102 x1
- tests/test_standalone_checkpoint.py | scanned | 7 | autofix cycle 10; 1 left: B905
- tests/test_trainability_types_verification.py | clean | 0 | -
- tests/test_training_results.py | clean | 0 | -
- tests/test_training_state.py | scanned | 4 | autofix cycle 10; 3 left: E402 x3
- tests/test_vicreg.py | clean | 0 | -
- tests/test_workspace.py | clean | 0 | -
- train/__init__.py | clean | 0 | -
- train/alternating_checkpoint.py | scanned | 4 | autofix cycle 6; 2 left: B905, PLC0415
- train/alternating_config.py | fixed | 1 | autofix cycle 6, ruff clean
- train/alternating_evaluation.py | unchecked | 2 | -
- train/alternating_scheduler.py | scanned | 3 | autofix cycle 6; 2 left: UP046 x2
- train/answer_objective.py | scanned | 3 | autofix cycle 6; 2 left: TRY004 x2
- train/benchmark_eggroll.py | clean | 0 | -
- train/eggroll_factorized.py | clean | 0 | -
- train/eggroll_perturbations.py | clean | 0 | -
- train/eggroll_stability.py | fixed | 6 | ruff clean cycle 2; empty-tuple thaw bug fixed
- train/eggroll_stability_evaluation.py | clean | 0 | -
- train/eggroll_stability_guard.py | unchecked | 1 | -
- train/eggroll_trainer.py | scanned | 7 | autofix cycle 6 (4 stale imports removed); 2 left: E731, PLC0415
- train/eggroll_updates.py | unchecked | 1 | -
- train/plot_training.py | scanned | 8 | autofix cycle 6; 6 left: TRY004 x2, SIM108, E731 x2, RUF046
- train/run_alternating.py | scanned | 4 | autofix cycle 6; 2 left: B905, PLC0415
- train/run_eggroll.py | scanned | 9 | autofix cycle 7; 8 left: B023 x8 (one closure)
- train/run_eggroll_stability.py | fixed | 1 | autofix cycle 7, ruff clean
- train/run_stage0_trainability.py | scanned | 27 | autofix cycle 7 (11 stale imports removed); 7 left: PLC0415 x2, BLE001 x3, F841, TRY004
- train/run_training.py | unchecked | 8 | -
- train/search_alignment_weight.py | scanned | 5 | autofix cycle 7; 1 left: SIM102
- train/stage0_checkpoint.py | scanned | 6 | autofix cycle 7; 4 left: SIM102 x2, SIM105, PLC0415
- train/stage0_data.py | fixed | 1 | autofix cycle 7, ruff clean
- train/stage0_trainability.py | scanned | 27 | autofix cycle 7; 25 left: BLE001 x8, PLC0415 x5, SIM x3, B904 x2, RUF059 x2, misc
- train/standalone_checkpoint.py | scanned | 4 | autofix cycle 7; 2 left: B905 x2
- train/trainer.py | fixed | 5 | autofix cycle 7, ruff clean (4 stale imports removed)
- train/training_results.py | clean | 0 | -
- train/training_state.py | clean | 0 | -
- train/vicreg.py | unchecked | 1 | -
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
