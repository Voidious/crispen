---
name: test-patch-rewrite
description: >
  Automates crispen's manual FileLimiter/patch-rewriter test loop: force a
  target file to be re-split by feeding a full-file diff through `uv run
  crispen`, then check flake8/pytest and whether the LLM path was actually
  exercised (configurable success signal), retrying (bounded) only when it
  wasn't. Use when asked to test, verify, or exercise patch rewriting /
  FileLimiter splitting in crispen with live LLM calls after a code change.
version: 2
---

# Test Patch Rewrite

Automates the loop a crispen developer runs by hand to verify a change to
FileLimiter/patch_rewriter actually gets exercised by live LLM calls, without
babysitting each iteration.

## How this gets invoked

This is a plain bash script (`scripts/run.sh`) — it needs no AI to run it. A
developer can call it directly from a terminal at any time. The `SKILL.md`
packaging exists so that when a Claude Code agent is working in this repo
and is asked to "test patch rewriting" (or similar), it already knows the
command, the params, and — critically — the cost/retry rules below, instead
of being told them fresh each time or improvising its own retry loop.

When an agent invokes this skill, it should still confirm the target file,
branch suffix, and configured provider/model with the user first (see
"What this costs money to run"), then run `scripts/run.sh` with those flags
and report the summary back — it should not loop the script itself or
increase `--max-retries` beyond what the user asked for.

## What this costs money to run

Every attempt invokes `uv run crispen`, which makes **real, billed LLM API
calls** (provider/model from `pyproject.toml [tool.crispen]` or
`.crispen.toml`). The number of calls per attempt is not something this
script controls — it depends on how many functions/classes the FileLimiter
decides to split out and how many of those need patch-rewrite fixups
elsewhere. The only cost lever this skill gives you is bounding **how many
attempts** run, via `--max-retries` (default 3). Do not raise it past what
you're prepared to pay for, and do not wrap this script in your own outer
retry loop.

Before running: confirm with the user which provider/model is configured
(check `.crispen.toml` / `pyproject.toml [tool.crispen]`) and that they're
OK spending on it. If they want a free rehearsal first, point them at
`provider = "lmstudio"` or `"ollama"` in `.crispen.toml` (no API key, no
cost) to sanity-check the mechanics before switching to a real provider.

## Usage

```bash
.claude/skills/test-patch-rewrite/scripts/run.sh \
  --target <target-file> \
  --branch-suffix <branch-suffix> \
  [--max-retries N] \
  [--success-labels "label1,label2,..."]
```

- `--target`: repo-relative path to the file to force-resplit, e.g.
  `crispen/file_limiter/advisor.py`. This is the file the developer is
  currently iterating on (matches whatever they set up in `.crispen.toml`).
- `--branch-suffix`: becomes `crispen-<suffix>`, e.g.
  `070-anthropic-sonnet46-01` → branch `crispen-070-anthropic-sonnet46-01`.
  Follow the developer's `VERSION-PROVIDER_OR_MODEL-COUNTER` convention.
- `--max-retries`: hard cap on attempts (default `3`). Each attempt is one
  live `uv run crispen` invocation.
- `--success-labels`: comma-separated crispen summary line labels (include
  the trailing colon) whose counts get summed to decide whether the LLM
  path was actually exercised this attempt. **Default is
  `"LLM no-change:,LLM rename:,LLM rewrite:"`** — the three patch-rewrite
  verdict lines from crispen's summary, matching "at least one of LLM
  no-change/rename/rewrite" as the bar for having verified the LLM side.
  Change this when testing something other than patch-rewrite verdicts —
  e.g. `"file limiter:"` under `LLM calls:` to check FileLimiter itself got
  invoked, or `"veto:"` for the veto path. Any label from crispen's
  `--- crispen summary ---` output works; unmatched labels just count as 0
  (so a typo silently reads as "never succeeds" — double check the label
  text against `crispen/stats.py`'s `format_summary()` if a run seems stuck
  retrying).

The script works from wherever the repo is checked out (resolves the repo
root with `git rev-parse --show-toplevel`), so it's not tied to WSL or any
specific clone path.

## What it does, per attempt

1. Reset the working tree and delete the FileLimiter-generated package dirs
   (`crispen/engine`, `crispen/patch_rewriter`, `crispen/file_limiter/advisor`,
   `crispen/file_limiter/code_gen`, `crispen/refactors/duplicate_extractor`)
   so the target file starts from a clean, unsplit state.
2. `git diff --no-index -- /dev/null <target-file> | uv run crispen` —
   treats the whole file as new/changed so FileLimiter reconsiders splitting
   it. (Uses `--no-index` and writes the diff to a temp file rather than
   piping straight through, because plain `git diff /dev/null <file>` — as
   in the original manual recipe — breaks under Git Bash on Windows, and
   `--no-index` legitimately exits 1 whenever the files differ, which would
   otherwise look like a crispen failure under `pipefail`.)
3. `git add -A`, then `flake8 --extend-ignore=E501` on exactly the changed
   `.py` files (line-too-long is excepted — often unavoidable on
   LLM-rewritten code, not a real bug signal), then
   `pytest --cov-fail-under=0` (overrides the repo's normal
   `--cov-fail-under=100` from `pyproject.toml` — last CLI occurrence wins —
   so a red run here means an actual test failure, not the coverage gate
   catching a newly-split file with incomplete tests).
4. Sum the `--success-labels` counts and check `edit failures:` from
   crispen's summary.

## Retry / stop logic (the part that controls spend)

Only one thing triggers a retry: **zero matching `--success-labels`
count** — that's expected nondeterminism (depends on how FileLimiter
happens to split the file), not a failure.

Everything else is a hard stop, no retry, because retrying a real bug
signal just spends more money reproducing it:

- **flake8 failure** (scoped to the exact files this run touched).
- **pytest failure** (with coverage overridden to 0, this is now a genuine
  test failure, not the coverage gate).
- **Non-zero `edit failures`** in crispen's summary.
- **Reaching `max-retries`** without a clean attempt.

## After the loop

- Success: commits on the test branch (`git commit -m "crispen changes"
  --no-verify`), then checks out back to the branch you started from.
  Leaves the commit on the test branch for the developer to inspect/discard.
- Hard stop (flake8/pytest/edit-failures) or retries exhausted: leaves the
  working tree as-is (uncommitted) on the test branch for inspection —
  does not reset or commit — then reports and exits non-zero. Checking out
  back to the base branch is left as a manual step in the hard-stop case so
  nothing gets discarded before a human looks at it.
- Retries exhausted with zero LLM-path signal (no real bug found, just never
  triggered): resets the test branch and checks out back to the starting
  branch. The empty test branch is left behind (matches the developer's
  manual habit); offer to delete it if the user wants.

## Known issue on `patch/misc-fixes` (as of this writing)

`uv run pytest --cov-fail-under=0` on this branch currently surfaces 6
pre-existing failures in `tests/test_patch_rewriter.py` (OSError-handling
tests, e.g. `test_rewrite_oserror_skipped`,
`test_cross_file_disk_oserror`) that are unrelated to FileLimiter
splitting. Since this skill now hard-stops on any pytest failure, running it
as-is on this branch will stop on those, not on anything caused by the
attempt. Worth fixing or noting before relying on the hard-stop behavior.

## Always report back

Regardless of outcome, tell the user: how many attempts it took, the final
success-label sum and which labels were used, the `edit failures:` count,
the LLM token usage line (`LLM tokens: N in / M out`) from crispen's summary
if present, and flake8/pytest status. Never report success without those
numbers — a silent "done" hides whether real LLM calls actually happened.
