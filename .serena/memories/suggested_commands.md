# Suggested commands

- Full tests: `rtk .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`.
- One file: `rtk .\.venv\Scripts\python.exe -m pytest tests\test_latent_loop.py -q -p no:cacheprovider`.
- One test: `rtk .\.venv\Scripts\python.exe -m pytest tests\test_latent_loop.py::test_latent_loop_step_shape -q -p no:cacheprovider`.
- Ruff check: `rtk .\.venv\Scripts\ruff.exe check .`.
- Ruff safe fixes and format: `rtk .\.venv\Scripts\ruff.exe check --fix .` then `rtk .\.venv\Scripts\ruff.exe format .`.
- Stage 0 training: `rtk .\.venv\Scripts\python.exe -m train.run_training --epochs 10`.
- Stage 0 gate: `rtk .\.venv\Scripts\python.exe scripts\run_gate.py`.
- Strict spec validation: `rtk openspec validate --all --strict`.
- Serena health: `$env:PYTHONUTF8=\"1\"; rtk serena project health-check`.
- Inspect changes: `rtk git status --short`; search with `rtk rg <pattern> <localized-path>`.
- For PowerShell cmdlets through the proxy: `rtk powershell -NoProfile -Command '<cmdlet>'`.