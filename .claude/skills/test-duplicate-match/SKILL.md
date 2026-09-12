---
name: test-duplicate-match
description: >
  Automates crispen's manual DuplicateExtractor test loop — both the
  "match existing function" sub-pass (file-local or repo-wide) and plain
  new-duplicate extraction: set up (or find) a real duplicate, feed a
  full-file diff through `uv run crispen`, then check flake8/pytest and
  whether the LLM path was actually exercised, retrying (bounded) only when
  it wasn't. Use when asked to test, verify, or exercise duplicate
  extraction / match-function in crispen with live LLM calls after a code
  change.
version: 1
---

# Test Duplicate Match

Automates the loop a crispen developer runs by hand to verify a change to
`DuplicateExtractor` (new-duplicate extraction or match-function, file-local
or repo-wide) actually gets exercised by live LLM calls, without babysitting
each iteration.

## How this gets invoked

`scripts/run.sh` is a plain bash script — it needs no AI to run it once a
test case exists on disk. The AI part of this skill is upstream of the
script: **finding or contriving the duplicate to test with** (see "Setting
up a test case" below), which a bash script can't do on its own. An agent
invoking this skill should do that setup, confirm the target file, branch
suffix, and configured provider/model with the user first (see "What this
costs money to run"), then run `scripts/run.sh` and report the summary back
— it should not loop the script itself or increase `--max-retries` beyond
what the user asked for.

## Setting up a test case first

Unlike FileLimiter (any oversized file is a valid test case on its own),
DuplicateExtractor's match-function needs an actual duplicate to exist.
Before running the script:

1. **Look for a real one first.** Search the repo for a small free function
   or `@staticmethod` (module-level `def`, 3+ statements, generic/reusable
   shape) that has a structurally-identical twin elsewhere — same sequence
   of statement *shapes* (assignments, calls, returns), variable/call names
   don't matter, they get normalized away. If you find one, just diff
   whichever file holds the duplicate block; no scratch files needed and the
   found example is inherently realistic. Report what you found either way,
   even if it's "codebase looks clean, contriving one" — a well-factored
   repo genuinely may not have one.

2. **If none exists, contrive a minimal pair.** Two small files:
   - One with a small free function (or `@staticmethod`), e.g.
     `crispen/_livetest_helpers.py`.
   - One whose target function's body structurally duplicates it with
     different names, e.g. `crispen/_livetest_target.py`. This is the
     `--target` for the diff.

   Two gotchas discovered the hard way, both worth checking before you spend
   a live call on a run that silently no-ops:
   - **Docstrings are part of the fingerprint, verbatim.** The body-source
     capture used for fingerprinting does not strip docstrings, so a
     documented function will never fingerprint-match an undocumented twin
     (or one with a different docstring). Keep both functions docstring-free,
     or give them the *identical* docstring, if you want a guaranteed match.
   - **For repo-wide matching**, the repo-wide index scans the whole repo
     including the file you're about to diff — so the target file's own
     function will always appear as a same-fingerprint "candidate" for its
     own body. This is filtered out automatically (same-module candidates
     don't count), so it's not something you need to work around — just
     don't be surprised to see it in debug output.

   Use `crispen/_livetest_*.py` naming (or similar, clearly temporary) so
   there's no ambiguity about what to delete afterward — these are never
   meant to be committed for real. `git add` + commit them on your current
   branch (the script requires a clean working tree before it creates its
   own test branch), run the script, then `git reset --hard <the commit
   before your scratch setup>` on your original branch afterward to drop the
   scratch commit — the script's own test branch is separate and gets
   handled by its own success/failure logic (see "After the loop" below).

3. **Configure `.crispen.toml`** with the provider/model to test and scope
   it to just what you're testing, e.g.:
   ```toml
   provider = "moonshot"
   model = "kimi-k2.5"
   enabled_refactors = ["duplicate_extractor", "match_function"]
   match_functions_scope = "repo"  # or "file" to test the local-only path
   timing = "detailed"
   ```
   Back up any existing `.crispen.toml` first (it's gitignored, so it won't
   show up in the clean-tree check, but you'll want it back afterward).
   `enabled_refactors = ["duplicate_extractor"]` alone (omitting
   `"match_function"`) is how you'd isolate *new*-duplicate extraction
   instead — set up two duplicate blocks in the target file rather than a
   duplicate-of-an-existing-function.

## What this costs money to run

Every attempt invokes `uv run crispen`, which makes **real, billed LLM API
calls**. The number of calls per attempt is small and bounded for this
feature (a veto call, plus a call-generation or extraction call, per
matched/extracted block) — much cheaper per-attempt than FileLimiter runs.
The only cost lever this skill gives you is bounding **how many attempts**
run, via `--max-retries` (default 3). Do not raise it past what you're
prepared to pay for, and do not wrap this script in your own outer retry
loop.

Before running: confirm with the user which provider/model is configured
and that they're OK spending on it.

## Usage

```bash
.claude/skills/test-duplicate-match/scripts/run.sh \
  --target <target-file> \
  --branch-suffix <branch-suffix> \
  [--max-retries N] \
  [--success-labels "label1,label2,..."]
```

- `--target`: repo-relative path to the file holding the duplicate block to
  diff, e.g. `crispen/_livetest_target.py`.
- `--branch-suffix`: becomes `crispen-<suffix>`, e.g.
  `080a-moonshot-kimik25-01` → branch `crispen-080a-moonshot-kimik25-01`.
- `--max-retries`: hard cap on attempts (default `3`).
- `--success-labels`: comma-separated crispen summary line labels (include
  the trailing colon). **Default is `"match existing:,duplicate extracted:"`**
  — both DuplicateExtractor outcome lines from crispen's summary. Narrow it
  to just one if you're isolating match-function vs. new-duplicate
  extraction specifically. Any label from crispen's `--- crispen summary ---`
  output works; unmatched labels just count as 0 (so a typo silently reads
  as "never succeeds" — double check against `crispen/stats.py`'s
  `format_summary()` if a run seems stuck retrying).

The script works from wherever the repo is checked out (resolves the repo
root with `git rev-parse --show-toplevel`).

## What it does, per attempt

1. Reset the working tree to the branch's starting commit (your committed
   test-case setup).
2. `git diff --no-index -- /dev/null <target-file> | uv run crispen` —
   treats the whole file as new/changed. (Uses `--no-index` and writes the
   diff to a temp file rather than piping straight through, because plain
   `git diff /dev/null <file>` breaks under Git Bash on Windows, and
   `--no-index` legitimately exits 1 whenever the files differ, which would
   otherwise look like a crispen failure under `pipefail`.)
3. `git add -A`, then `flake8 --extend-ignore=E501` on exactly the changed
   `.py` files, then `pytest --cov-fail-under=0`.
4. Sum the `--success-labels` counts.

## Retry / stop logic (the part that controls spend)

Only one thing triggers a retry: **zero matching `--success-labels`
count** — the LLM veto can legitimately reject a match; that's expected
nondeterminism, not a failure.

Everything else is a hard stop, no retry:

- **flake8 failure** (scoped to the exact files this run touched) — this is
  how a real formatting bug in the generated code/import gets caught; see
  the blank-lines-before-def bug found and fixed during 0.8.0-a's live
  verification as a concrete example of exactly this.
- **pytest failure** (coverage overridden to 0).
- **Reaching `max-retries`** without a clean attempt.

## After the loop

Same outcome handling as `test-patch-rewrite`: success commits on the test
branch and returns to your starting branch (commit left for inspection);
a hard stop leaves the tree as-is on the test branch, uncommitted, for
review; retries-exhausted-with-zero-signal resets the test branch and
returns you to your starting branch. Either way, remember your scratch
test-case commit is still sitting on your *original* branch — drop it once
you're done (see "Setting up a test case" above).

## Always report back

Regardless of outcome, tell the user: how many attempts it took, the final
success-label sum and which labels were used, the `LLM tokens: N in / M
out` line from crispen's summary if present, flake8/pytest status, and
whether you found a real duplicate or had to contrive one. Never report
success without those numbers.
