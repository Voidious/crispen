#!/usr/bin/env bash
# Automates crispen's manual FileLimiter/patch-rewriter test loop.
# See ../SKILL.md for what this does and why, especially the cost/retry rules.
#
# Usage:
#   run.sh --target <file> --branch-suffix <suffix> \
#          [--max-retries N] [--success-labels "label1,label2,..."]
#
#   --target          repo-relative path, e.g. crispen/file_limiter/advisor.py
#   --branch-suffix   becomes branch "crispen-<suffix>"
#   --max-retries     default 3; each retry is one live `uv run crispen` call
#   --success-labels  comma-separated crispen summary line labels (with
#                      trailing colon) whose counts are summed to decide
#                      whether the LLM path was actually exercised. Default
#                      is the three patch-rewrite verdict lines. Change this
#                      when testing something other than patch rewriting,
#                      e.g. "file limiter:" for FileLimiter LLM calls, or
#                      "veto:" for the veto path.
#                      Default: "LLM no-change:,LLM rename:,LLM rewrite:"

set -euo pipefail

MAX_RETRIES=3
SUCCESS_LABELS="LLM no-change:,LLM rename:,LLM rewrite:"
TARGET_FILE=""
BRANCH_SUFFIX=""

usage() {
  echo "usage: run.sh --target <file> --branch-suffix <suffix> [--max-retries N] [--success-labels \"label1,label2,...\"]" >&2
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET_FILE="$2"; shift 2 ;;
    --branch-suffix) BRANCH_SUFFIX="$2"; shift 2 ;;
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    --success-labels) SUCCESS_LABELS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

[ -n "$TARGET_FILE" ] || usage
[ -n "$BRANCH_SUFFIX" ] || usage

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ ! -f "$TARGET_FILE" ]; then
  echo "error: target file '$TARGET_FILE' not found relative to repo root $REPO_ROOT" >&2
  exit 1
fi

BRANCH="crispen-${BRANCH_SUFFIX}"
BASE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Directories FileLimiter may generate when it splits an oversized file into
# a package. Cleared before each attempt so the target starts unsplit.
GENERATED_DIRS=(
  crispen/engine
  crispen/patch_rewriter
  crispen/file_limiter/advisor
  crispen/file_limiter/code_gen
  crispen/refactors/duplicate_extractor
)

if [ -n "$(git status --porcelain)" ]; then
  echo "error: working tree is not clean. Commit or stash changes before running this script." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "==> no .venv found, running: uv sync"
  uv sync
fi

echo "==> git checkout -b $BRANCH (from $BASE_BRANCH)"
git checkout -b "$BRANCH"

return_to_base() {
  echo "==> git checkout $BASE_BRANCH"
  git checkout "$BASE_BRANCH"
}
trap return_to_base EXIT

# Sums the counts on every summary line matching one of the comma-separated
# SUCCESS_LABELS, e.g. "LLM no-change:,LLM rename:,LLM rewrite:" ->
# sum of those three lines' trailing numbers. Missing lines count as 0.
sum_success_labels() {
  local log="$1" total=0 val
  local IFS=','
  for label in $SUCCESS_LABELS; do
    val=$(grep -F "$label" "$log" | grep -o '[0-9]\+' || true)
    val="${val:-0}"
    total=$((total + val))
  done
  echo "$total"
}

attempt=1
outcome="failed"
success_count=0
edit_failures=0

while [ "$attempt" -le "$MAX_RETRIES" ]; do
  echo ""
  echo "===== Attempt $attempt / $MAX_RETRIES ====="

  git reset --hard HEAD >/dev/null
  git clean -fdx -- "${GENERATED_DIRS[@]}" >/dev/null 2>&1 || true
  rm -rf "${GENERATED_DIRS[@]}"

  # `git diff --no-index` exits 1 whenever the files differ (which they
  # always will here) — that's expected, not an error, so don't let it
  # trip `set -e`/pipefail. Write to a temp file rather than piping
  # directly into crispen, so its exit code never gets conflated with
  # git diff's.
  diff_file="$(mktemp)"
  diff_status=0
  git diff --no-index -- /dev/null "$TARGET_FILE" >"$diff_file" || diff_status=$?
  if [ "$diff_status" -gt 1 ]; then
    echo "error: git diff --no-index failed unexpectedly (exit $diff_status)" >&2
    exit 1
  fi
  diff_lines=$(wc -l < "$diff_file")
  echo "diff size ($TARGET_FILE vs /dev/null): $diff_lines lines"

  stdout_log="$(mktemp)"
  stderr_log="$(mktemp)"

  if ! uv run crispen <"$diff_file" >"$stdout_log" 2>"$stderr_log"; then
    echo "!! crispen exited non-zero — stderr tail:"
    tail -n 40 "$stderr_log"
    attempt=$((attempt + 1))
    continue
  fi

  success_count=$(sum_success_labels "$stdout_log")
  edit_failures=$(grep -F 'edit failures:' "$stdout_log" | grep -o '[0-9]\+' || true)
  edit_failures="${edit_failures:-0}"
  tokens_line=$(grep -F 'LLM tokens:' "$stdout_log" || true)

  echo "success-label count ($SUCCESS_LABELS): $success_count"
  echo "edit failures:                          $edit_failures"
  [ -n "$tokens_line" ] && echo "$tokens_line"

  git add -A

  # Scope flake8 to exactly the files this attempt touched (staged above),
  # same as the pre-commit hook would, but with an extra E501 exception:
  # long lines are a known, often-unavoidable false positive on LLM-rewritten
  # code, not a real bug signal. flake8's --extend-ignore on the command line
  # REPLACES (not merges with) .flake8's own `extend-ignore` setting, so a
  # naive `--extend-ignore=E501` here would silently re-enable whatever the
  # repo's config already exempts (e.g. E203, black's known slice-whitespace
  # disagreement with flake8) and produce false failures unrelated to this
  # run. Read the repo's existing extend-ignore and append E501 to it instead.
  repo_extend_ignore=$(sed -n 's/^extend-ignore[[:space:]]*=[[:space:]]*//p' .flake8 2>/dev/null | tr -d ' \r')
  if [ -n "$repo_extend_ignore" ]; then
    extend_ignore="${repo_extend_ignore},E501"
  else
    extend_ignore="E501"
  fi
  py_files=$(git diff --cached --name-only --diff-filter=ACM -- '*.py' | grep -v '^examples/' || true)
  if [ -n "$py_files" ]; then
    echo "==> flake8 (--extend-ignore=$extend_ignore) on changed files"
    if ! echo "$py_files" | xargs uv run flake8 --extend-ignore="$extend_ignore"; then
      echo "!! flake8 failed — stopping (real bug signal, not retrying)"
      outcome="flake8_failed"
      break
    fi
  else
    echo "==> no changed .py files to lint"
  fi

  # --cov-fail-under=0 overrides pyproject.toml's --cov-fail-under=100 (last
  # occurrence wins), so a red run here means an actual test failure, not
  # just missing coverage on newly-split code (which is expected/OK).
  echo "==> pytest (full suite, coverage gate disabled for this check)"
  if ! uv run pytest --cov-fail-under=0; then
    echo "!! pytest failed — stopping (real bug signal, not retrying)"
    outcome="pytest_failed"
    break
  fi

  if [ "$edit_failures" -ne 0 ]; then
    echo "!! non-zero edit failures — stopping (real bug signal, not retrying)"
    outcome="edit_failures"
    break
  fi

  if [ "$success_count" -eq 0 ]; then
    echo "!! zero matching LLM outcomes this attempt — retrying (split/outcome is nondeterministic)"
    attempt=$((attempt + 1))
    continue
  fi

  outcome="success"
  break
done

echo ""
echo "===== SUMMARY ====="
echo "attempts:            $attempt / $MAX_RETRIES"
echo "success-label count: $success_count ($SUCCESS_LABELS)"
echo "edit failures:       $edit_failures"
[ -n "${tokens_line:-}" ] && echo "tokens:              $tokens_line"

case "$outcome" in
  success)
    echo "result:        SUCCESS — committing on $BRANCH"
    git commit -m "crispen changes" --no-verify
    echo "Committed on branch $BRANCH. Inspect with: git show $BRANCH"
    ;;
  flake8_failed|pytest_failed|edit_failures)
    echo "result:        STOPPED ($outcome) — needs human review, not auto-retried"
    echo "Leaving working tree as-is on $BRANCH for inspection (not resetting, not committing)."
    trap - EXIT
    echo "Run 'git checkout $BASE_BRANCH' manually once you're done inspecting $BRANCH."
    exit 2
    ;;
  *)
    echo "result:        FAILED — exhausted $MAX_RETRIES attempts with zero LLM-path signal, resetting, no commit"
    git reset --hard HEAD
    exit 1
    ;;
esac
