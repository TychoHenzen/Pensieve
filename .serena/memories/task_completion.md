# Task completion

- Run focused tests during iteration. Run the full pytest suite for broad or shared-path changes when practical.
- Before completion, run `rtk .\.venv\Scripts\ruff.exe check .` and `rtk .\.venv\Scripts\ruff.exe format --check .`.
- Canonical full test command: `rtk .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`.
- If OpenSpec artifacts changed, run `rtk openspec validate --all --strict`. Validation proves artifact structure, not runtime behavior.
- If learning behavior changed, run the relevant real training/evaluation gate. A single seed does not satisfy the Stage 0 five-seed gate.
- If checkpoint logic changed, run paired resume-equivalence tests and load a trusted local checkpoint through the production path.
- Use `rtk git status --porcelain` and account for every changed or untracked file. Preserve unrelated user edits.
- Do not claim completion from checkboxes, commit text, fixture-only tests, passing syntax checks, or loss alone.