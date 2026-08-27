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
- Flaky: one full-suite run died with a Windows access violation in
  transformers weight loading (torch/storage.py __getitem__) during
  tests/test_narration.py. Did not reproduce in isolation or on rerun.
- Last full-suite run: cycle 1.

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
- codecs_module/__init__.py | unchecked | 1 | -
- codecs_module/decoder.py | clean | 0 | -
- codecs_module/encoder.py | clean | 0 | -
- codecs_module/narration.py | clean | 0 | -
- core/__init__.py | clean | 0 | -
- core/latent_loop.py | unchecked | 1 | -
- core/qwen_tap.py | unchecked | 2 | -
- eval/__init__.py | clean | 0 | -
- eval/baselines/__init__.py | clean | 0 | -
- eval/baselines/ewc.py | unchecked | 2 | -
- eval/baselines/frozen.py | clean | 0 | -
- eval/baselines/joint.py | unchecked | 1 | -
- eval/baselines/model.py | clean | 0 | -
- eval/baselines/naive.py | unchecked | 2 | -
- eval/baselines/param_match.py | clean | 0 | -
- eval/baselines/replay.py | clean | 0 | -
- eval/gate/__init__.py | clean | 0 | -
- eval/gate/answer_scoring.py | unchecked | 1 | -
- eval/gate/cycling_sweep.py | clean | 0 | -
- eval/gate/gate_report.py | unchecked | 4 | -
- eval/gate/latent_eval.py | unchecked | 9 | -
- eval/gate/result_cache.py | unchecked | 1 | -
- eval/gate/run_gate.py | unchecked | 3 | -
- eval/gate/slot_ablation.py | unchecked | 4 | -
- eval/gate/token_cot_baseline.py | unchecked | 3 | -
- eval/instrumentation.py | clean | 0 | -
- eval/metrics/__init__.py | unchecked | 2 | -
- eval/metrics/accuracy.py | clean | 0 | -
- eval/metrics/compute.py | clean | 0 | -
- eval/metrics/first_use.py | unchecked | 1 | -
- eval/metrics/retention.py | clean | 0 | -
- eval/metrics/transfer.py | clean | 0 | -
- eval/run/__init__.py | clean | 0 | -
- eval/run/checkpoint.py | clean | 0 | -
- eval/run/config.py | clean | 0 | -
- eval/run/retention_policy.py | clean | 0 | -
- eval/run/runner.py | unchecked | 1 | -
- eval/stage0_identity.py | unchecked | 10 | -
- eval/stream/__init__.py | clean | 0 | -
- eval/stream/config.py | clean | 0 | -
- eval/stream/corpus.py | unchecked | 1 | -
- eval/stream/corpus_walk.py | clean | 0 | -
- eval/stream/events.py | unchecked | 1 | -
- eval/stream/generator.py | unchecked | 1 | -
- eval/stream/generators/__init__.py | clean | 0 | -
- eval/stream/generators/asdiv_a.py | unchecked | 4 | -
- eval/stream/generators/assoc.py | clean | 0 | -
- eval/stream/generators/difficulty_mix.py | unchecked | 1 | -
- eval/stream/generators/gsm8k.py | unchecked | 1 | -
- eval/stream/generators/split_classify.py | unchecked | 2 | -
- eval/stream/hashing.py | clean | 0 | -
- eval/stream/registry.py | unchecked | 1 | -
- eval/stream/render.py | unchecked | 1 | -
- eval/stream/serialize.py | clean | 0 | -
- eval/stream/truth.py | clean | 0 | -
- eval/stream/vocab.py | clean | 0 | -
- eval/subject/__init__.py | unchecked | 3 | -
- eval/subject/isolation.py | clean | 0 | -
- eval/subject/oracles/__init__.py | clean | 0 | -
- eval/subject/oracles/chance.py | clean | 0 | -
- eval/subject/oracles/cheater.py | clean | 0 | -
- eval/subject/oracles/forgetful.py | unchecked | 1 | -
- eval/subject/oracles/perfect_memory.py | unchecked | 1 | -
- eval/subject/oracles/task_wiper.py | unchecked | 1 | -
- eval/subject/oracles/variable_compute.py | clean | 0 | -
- eval/subjects/__init__.py | clean | 0 | -
- eval/subjects/latent_core.py | unchecked | 1 | -
- main.py | clean | 0 | -
- scripts/fetch_corpus.py | unchecked | 2 | -
- scripts/run_gate.py | unchecked | 3 | -
- scripts/show_gate.py | unchecked | 8 | -
- scripts/show_stream.py | clean | 0 | -
- tests/baselines/__init__.py | clean | 0 | -
- tests/baselines/test_baselines_smoke.py | unchecked | 1 | -
- tests/baselines/test_param_match.py | clean | 0 | -
- tests/eggroll_reference.py | unchecked | 2 | -
- tests/eggroll_stability_fixtures.py | clean | 0 | -
- tests/gate/__init__.py | clean | 0 | -
- tests/gate/test_gate_smoke.py | clean | 0 | -
- tests/gate/test_numerical_answer_scoring.py | clean | 0 | -
- tests/gate/test_stage0_gate_report.py | unchecked | 1 | -
- tests/gate/test_stage0_latent_evaluation.py | unchecked | 1 | -
- tests/gate/test_stage0_result_cache.py | unchecked | 1 | -
- tests/gate/test_stage0_run_gate.py | unchecked | 2 | -
- tests/gate/test_stage0_slot_ablation.py | clean | 0 | -
- tests/gate/test_stage0_token_baseline.py | unchecked | 1 | -
- tests/instrumentation/__init__.py | clean | 0 | -
- tests/instrumentation/test_difficulty_correlation.py | unchecked | 1 | -
- tests/instrumentation/test_instrumentation_fields.py | clean | 0 | -
- tests/metrics/__init__.py | clean | 0 | -
- tests/metrics/test_accuracy.py | clean | 0 | -
- tests/metrics/test_first_use.py | clean | 0 | -
- tests/metrics/test_oracle_integration.py | unchecked | 1 | -
- tests/metrics/test_retention.py | clean | 0 | -
- tests/metrics/test_transfer.py | clean | 0 | -
- tests/run/test_checkpoint.py | unchecked | 1 | -
- tests/run/test_config.py | unchecked | 1 | -
- tests/run/test_resume.py | unchecked | 1 | -
- tests/run/test_retention_policy.py | clean | 0 | -
- tests/run/test_runner.py | unchecked | 1 | -
- tests/stream/test_asdiv_a.py | unchecked | 1 | -
- tests/stream/test_assoc.py | unchecked | 4 | -
- tests/stream/test_chance.py | clean | 0 | -
- tests/stream/test_corpus.py | unchecked | 1 | -
- tests/stream/test_difficulty_mix.py | clean | 0 | -
- tests/stream/test_events.py | unchecked | 1 | -
- tests/stream/test_generator.py | clean | 0 | -
- tests/stream/test_gsm8k.py | clean | 0 | -
- tests/stream/test_hashing.py | unchecked | 3 | -
- tests/stream/test_mnist_binding.py | clean | 0 | -
- tests/stream/test_package.py | clean | 0 | -
- tests/stream/test_registry.py | unchecked | 1 | -
- tests/stream/test_render.py | clean | 0 | -
- tests/stream/test_replay.py | clean | 0 | -
- tests/stream/test_split_classify.py | unchecked | 5 | -
- tests/stream/test_truth.py | unchecked | 7 | -
- tests/stream/test_vocab.py | unchecked | 2 | -
- tests/subject/test_isolation.py | clean | 0 | -
- tests/subject/test_oracles.py | unchecked | 2 | -
- tests/subject/test_protocol.py | unchecked | 1 | -
- tests/subject/test_snapshot_restore.py | clean | 0 | -
- tests/test_alternating_checkpoint.py | unchecked | 5 | -
- tests/test_alternating_config.py | unchecked | 1 | -
- tests/test_alternating_evaluation.py | unchecked | 2 | -
- tests/test_alternating_scheduler.py | clean | 0 | -
- tests/test_answer_objective.py | clean | 0 | -
- tests/test_asdiv_stage0_contract.py | clean | 0 | -
- tests/test_benchmark_eggroll.py | unchecked | 1 | -
- tests/test_codecs_freezing.py | clean | 0 | -
- tests/test_eggroll_factorized.py | clean | 0 | -
- tests/test_eggroll_perturbations.py | clean | 0 | -
- tests/test_eggroll_reference.py | clean | 0 | -
- tests/test_eggroll_resume_equivalence.py | unchecked | 2 | -
- tests/test_eggroll_stability.py | fixed | 2 | ruff clean cycle 2; regression test added
- tests/test_eggroll_step_equivalence.py | unchecked | 1 | -
- tests/test_eggroll_training.py | unchecked | 10 | -
- tests/test_eggroll_updates.py | clean | 0 | -
- tests/test_eggroll_workflow_guards.py | unchecked | 1 | -
- tests/test_frozen_qwen_backbone.py | clean | 0 | -
- tests/test_gradient_training.py | unchecked | 3 | -
- tests/test_latent_eval.py | clean | 0 | -
- tests/test_latent_loop.py | clean | 0 | -
- tests/test_narration.py | unchecked | 1 | -
- tests/test_plot_training.py | clean | 0 | -
- tests/test_probe_isolation.py | clean | 0 | -
- tests/test_qwen_latent_loop.py | unchecked | 1 | -
- tests/test_qwen_tap_equivalence.py | unchecked | 1 | -
- tests/test_run_alternating.py | unchecked | 2 | -
- tests/test_run_eggroll.py | unchecked | 2 | -
- tests/test_run_eggroll_stability.py | clean | 0 | -
- tests/test_run_stage0_trainability.py | unchecked | 10 | -
- tests/test_run_training.py | unchecked | 1 | -
- tests/test_search_alignment_weight.py | unchecked | 1 | -
- tests/test_stage0_checkpoint_container.py | unchecked | 9 | -
- tests/test_stage0_checkpoint_resume.py | unchecked | 2 | -
- tests/test_stage0_checkpoint_schema.py | unchecked | 4 | -
- tests/test_stage0_cli_runtime_order.py | clean | 0 | -
- tests/test_stage0_dataset_contract.py | unchecked | 1 | -
- tests/test_stage0_identity.py | unchecked | 1 | -
- tests/test_stage0_shapes.py | unchecked | 1 | -
- tests/test_stage0_trainability_identity.py | unchecked | 13 | -
- tests/test_stage0_trainability_reports.py | unchecked | 210 | -
- tests/test_standalone_checkpoint.py | unchecked | 7 | -
- tests/test_trainability_types_verification.py | clean | 0 | -
- tests/test_training_results.py | clean | 0 | -
- tests/test_training_state.py | unchecked | 4 | -
- tests/test_vicreg.py | clean | 0 | -
- tests/test_workspace.py | clean | 0 | -
- train/__init__.py | clean | 0 | -
- train/alternating_checkpoint.py | unchecked | 4 | -
- train/alternating_config.py | unchecked | 1 | -
- train/alternating_evaluation.py | unchecked | 2 | -
- train/alternating_scheduler.py | unchecked | 3 | -
- train/answer_objective.py | unchecked | 3 | -
- train/benchmark_eggroll.py | clean | 0 | -
- train/eggroll_factorized.py | clean | 0 | -
- train/eggroll_perturbations.py | clean | 0 | -
- train/eggroll_stability.py | fixed | 6 | ruff clean cycle 2; empty-tuple thaw bug fixed
- train/eggroll_stability_evaluation.py | clean | 0 | -
- train/eggroll_stability_guard.py | unchecked | 1 | -
- train/eggroll_trainer.py | unchecked | 7 | -
- train/eggroll_updates.py | unchecked | 1 | -
- train/plot_training.py | unchecked | 8 | -
- train/run_alternating.py | unchecked | 4 | -
- train/run_eggroll.py | unchecked | 9 | -
- train/run_eggroll_stability.py | unchecked | 1 | -
- train/run_stage0_trainability.py | unchecked | 27 | -
- train/run_training.py | unchecked | 8 | -
- train/search_alignment_weight.py | unchecked | 5 | -
- train/stage0_checkpoint.py | unchecked | 6 | -
- train/stage0_data.py | unchecked | 1 | -
- train/stage0_trainability.py | unchecked | 27 | -
- train/standalone_checkpoint.py | unchecked | 4 | -
- train/trainer.py | unchecked | 5 | -
- train/training_results.py | clean | 0 | -
- train/training_state.py | clean | 0 | -
- train/vicreg.py | unchecked | 1 | -
- workspace/__init__.py | unchecked | 1 | -
- workspace/concept_slots.py | clean | 0 | -

## Cycle log

Format: cycle N | item | outcome
- cycle 0 | setup: ruff config, prompt, ledger | baselines recorded
- cycle 1 | install torchvision 0.26.0+cpu, rerun full suite | mnist test fixed; found 2 real failures in test_search_alignment_weight.py (device_topology)
- cycle 2 | fix empty-tuple thaw bug in train/eggroll_stability.py + clear its ruff findings | 2 failing tests now pass; regression test added; 94 related tests green
