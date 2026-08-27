# Incremental Cleanup Autopilot

You are one cycle of a 100-cycle cleanup loop. Every cycle starts with a fresh
context and this same prompt. The ONLY memory between cycles is the file
`CLEANUP_LEDGER.md` in the repo root. Read it first, always.

This prompt is the approved plan. Do not enter plan mode. Do not ask the user
questions. If you are blocked, record the block in the ledger and stop. Commit
after each verified item; that instruction overrides any global rule that says
to commit only when asked.

## Hard rules

1. Work on branch `chore/incremental-cleanup`. If it does not exist, create it
   from the current HEAD. If it exists, check it out.
2. Do exactly ONE work item per cycle, then update the ledger, commit, and stop.
   Exception: a "scan" item may update the status of many files at once.
3. Never delete or weaken a test to make it pass. Never change an assertion to
   match buggy output. If a test exposes a real bug, fix the code.
4. Never modify anything under `openspec/changes/archive/`. It is read-only
   history you compare implementations against.
5. Never delete a file unless it matches the pre-approved delete list in the
   ledger's "Delete candidates" section. Anything else that looks stale gets a
   new row in that section instead, and a later human decides.
6. If the ledger shows any item with status `in-progress` when you start, your
   whole cycle is to finish or revert THAT item. Never start a new item while
   one is in progress.
7. `CLEANUP_LEDGER.md` is committed state, never gitignored.
8. Before any `git add`, run `git status --porcelain` and account for every
   `??` line: it is already ignored, gets a new `.gitignore` entry, or is a
   file you created on purpose. Nothing else goes into the commit.
9. Commit message format: `cleanup(cycle-N): <one line describing the item>`
   where N is the cycle number you append to the Cycle log.

## Commands (use these verbatim)

- Lint one file:            `python -m ruff check <path>`
- Lint whole repo:          `python -m ruff check .`
- Safe autofix one file:    `python -m ruff check <path> --fix`
- Run one test file:        `python -m pytest <path> -q`
- Run full suite:           `python -m pytest -q` (takes about 2-3 minutes)
- Validate specs:           `openspec validate --all --strict --no-interactive`

The ruff rule set lives in `pyproject.toml` under `[tool.ruff]`. Do not change
it. Do not add `# noqa` comments; fix the code. If a rule genuinely cannot be
satisfied (rare), mark the file `blocked` in the ledger with the rule code and
one line of reason.

## A cycle, start to finish

1. `git status` - if the working tree is dirty, a previous cycle crashed.
   Look at the diff. If it matches the `in-progress` ledger item and passes its
   verification, finish it (ledger, commit). Otherwise `git checkout -- .` and
   mark that item `blocked` with a note.
2. Open `CLEANUP_LEDGER.md`. Read the "How to pick work" order below. Pick the
   single highest-priority item.
3. Set that item's status to `in-progress` in the ledger. (Do not commit yet.)
4. Do the work.
5. Verify (see per-task verification below). If verification fails and you
   cannot fix it within this cycle, revert your code changes with
   `git checkout -- <files>`, set the item to `blocked` with a one-line reason,
   commit only the ledger, and stop.
6. On success: set the item's status to `fixed` (or `scanned` for scan items),
   append one line to the Cycle log, commit everything, stop.

## How to pick work (strict priority order)

1. **Finish in-progress.** See hard rule 6.
2. **Failing tests.** Run `python -m pytest -q` if the ledger says the last
   full-suite run is older than 10 cycles or a code fix landed since. Any
   failure not already listed as environment-blocked in "Pytest baseline" is
   the top item. Known: `tests/stream/test_mnist_binding.py` fails because
   torchvision is not installed; fixing it means `pip install torchvision`
   once, then rerunning - that is a valid single item.
3. **Ruff, safe autofixes.** Pick the first file in the ledger with status
   `unchecked` or `scanned` that has autofixable issues. Run
   `python -m ruff check <file> --fix`, then `python -m ruff check <file>`.
   You may batch autofix-only changes across up to 10 files in one item, since
   the fixes are mechanical. Verify: ruff reports fewer issues, and run the
   test files that cover those modules (matching `tests/` path or name).
4. **Ruff, manual fixes.** One file per cycle. Fix every remaining ruff finding
   in that file by hand. Common ones here: BLE001 blind excepts (narrow the
   exception type to what the code actually raises), B023 loop-variable
   capture (bind with a default argument), TRY004 (raise TypeError, not
   ValueError, for type checks), C408 (use literals, not `dict()`/`list()`
   calls). Verify: `python -m ruff check <file>` is clean AND the related test
   file passes. If the file has no test coverage, note `no-tests` in its
   ledger row - do not invent behavior changes in untested code; keep manual
   fixes strictly behavior-preserving there.
5. **Spec-to-test linkage.** Pick one capability directory under
   `openspec/specs/` whose ledger row in the "Spec coverage" section is
   `unchecked`. For each `#### Scenario:` in its `spec.md`, find a test that
   exercises it (grep `tests/` for the behavior, not the scenario title).
   Record in the ledger row: scenarios total, scenarios with a matching test,
   and list the uncovered ones by name. Writing a missing test is a SEPARATE
   later item, one test file per cycle, using the spec's GIVEN/WHEN/THEN as the
   only source of expected values. If the implementation contradicts the spec,
   do not adjust the test to fit - mark the row `conflict` with one line
   describing the mismatch.
6. **Placeholder or incorrect implementations.** Pick one archived change under
   `openspec/changes/archive/` whose ledger row is `unchecked`. Read its
   `tasks.md` and `specs/` deltas, then check the current code actually does
   what the archived change claims was shipped. Look especially for stub
   returns, `pass` bodies, hardcoded values, and TODO/FIXME markers in the
   touched files. Record findings in the row. Fixing a found defect is a
   separate later item with a test that pins the corrected behavior.
7. **Stale files.** Only paths already listed under "Delete candidates" with
   decision `approved`. Delete, update `.gitignore` if a process regenerates
   them, verify the full suite still passes, commit.

## Verification, non-negotiable

A code change is `fixed` only when:
- `python -m ruff check <changed files>` is clean, and
- the test files covering the changed modules pass, and
- for anything touching more than 3 files, the full suite passes.

Report only what the commands actually printed. Never mark `fixed` on the
expectation that it works.
