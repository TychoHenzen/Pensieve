# Pensive

- Staged research implementation of brain-inspired continual learning. Each stage needs measured gate evidence before advancing.
- Core invariant: internal reasoning uses continuous concept-slot vectors. Tokens exist only at encoder and decoder edges.
- Source map: `workspace/` persistent concept slots; `core/` latent loop and backbone tap; `codecs_module/` text boundary; `train/` runners, trainers, checkpoints; `eval/` deterministic streaming harness, baselines, metrics, gates; `runtime/`, `memory/`, and `consolidate/` are later-stage areas; `tests/` holds pytest coverage; `openspec/` holds executable requirements.
- Protect the known-good basic learning baseline before crediting Stage 0, latent, or EGGROLL progress. Passing tests or decreasing loss alone is not learning evidence.
- Training runs must preserve complete checkpoint state and deterministic resume behavior.
- Probe events are evaluation-only and must not leak into learning.
- Read `mem:tech_stack` for versions and dependencies, `mem:suggested_commands` for Windows commands, `mem:conventions` for code rules, and `mem:task_completion` for completion gates.