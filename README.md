# Crispen

Crispen is a Python code refactoring CLI tool. It reads a unified diff from stdin, identifies which lines changed, and applies a set of automated refactors only to the changed regions of affected files, writing the modified files back in place.

It uses [libcst](https://github.com/Instagram/LibCST) for format-preserving AST transformations and an LLM for intelligent, context-aware changes. Supported providers: Anthropic Claude, OpenAI, DeepSeek, Moonshot/Kimi, Gemini, Mistral, z.ai (GLM), and LM Studio or Ollama (local).

## Overview

```
git diff | crispen
```

Crispen operates only on the lines you actually changed — not the whole file. Each refactor receives the diff's line ranges and skips code outside those ranges. This makes it safe to run on any in-progress change without disturbing surrounding code.

## Installation

Crispen requires Python 3.12+.

```bash
git clone https://github.com/Voidious/crispen
cd crispen
uv sync
```

## Usage

Pipe any unified diff to crispen:

```bash
# Refactor uncommitted changes
git diff | crispen

# Refactor staged changes
git diff --cached | crispen

# Refactor changes since a specific commit
git diff HEAD~1 | crispen

# Refactor a specific file as if it were entirely new (triggers whole-file subdir split)
git diff /dev/null crispen/file_limiter/advisor.py | crispen
```

Crispen prints a summary of every change it applies, then writes the modified files back in place.

## Configuration

Crispen reads configuration from `[tool.crispen]` in `pyproject.toml`, with optional overrides in `.crispen.toml` in the project root.

```toml
[tool.crispen]
# LLM provider: "anthropic" (default), "openai", "deepseek", "moonshot", "gemini",
# "mistral", "zai", "lmstudio", or "ollama"
provider = "anthropic"

# LLM model to use (default: "claude-sonnet-4-6")
model = "claude-sonnet-4-6"

# Optional base URL override for OpenAI-compatible providers.
# Useful for LM Studio on a non-default port, or other self-hosted endpoints.
# base_url = "http://localhost:1234/v1"

# Optional tool_choice override for OpenAI-compatible providers (default: unset).
# Use "required" for local models (e.g. LM Studio with qwen3) that do not
# support the default named-function form.
# tool_choice = "required"

# HTTP timeout in seconds for each LLM API call (default: 60.0).
# Raise this when using slow local models.
# api_timeout = 60.0

# FunctionSplitter: max function body lines before splitting (default: 75)
max_function_length = 75

# FileLimiter: max file lines before splitting into sibling files (default: 1000).
# Set to 0 to disable FileLimiter entirely.
max_file_lines = 1000

# FileLimiter: when the diff covers every line of the file (whole-file add or
# replacement), place new files in a subdirectory named after the module
# (e.g. service/*.py for service.py). Default: true.
file_limiter_subdir_split = true

# FileLimiter: route pytest fixtures split out of test files to conftest.py
# instead of a regular sibling module, so pytest auto-discovers them without
# any import in the original file (avoids F401/F811 flake8 warnings).
# Set to false if not using pytest or using custom fixture decorators. Default: true.
file_limiter_pytest_conftest = true

# FileLimiter: when to keep re-export stubs in the original file for public
# names that were moved to new files. Re-exports preserve the module's public
# API so existing callers need no changes. Default: "imported".
#
# "always"      — Always add re-exports for every public name. Best for
#                 library packages whose public API may not be imported within
#                 this codebase.
# "application" — Add re-exports in non-test files (same as "always"), but
#                 omit them in test files. Pragmatic middle ground.
# "imported"    — Only add a re-export when the name is actually imported from
#                 the original module somewhere else in the project (the same
#                 rule already used for private names). Best for most
#                 application codebases. Note: only detects
#                 "from module import name" style imports, not qualified access
#                 via "module.name".
file_limiter_reexports = "imported"

# Tuple Return to Dataclass: min tuple element count to trigger replacement (default: 4)
min_tuple_size = 4

# Tuple Return to Dataclass: update callers in diff files even outside changed ranges (default: true)
# When false and unreachable callers exist, the transformation is skipped instead
update_diff_file_callers = true

# DuplicateExtractor: min statement weight for a duplicate group (default: 3)
min_duplicate_weight = 3

# DuplicateExtractor: max sequence length for duplicate search (default: 8)
max_duplicate_seq_len = 8

# Whether to generate docstrings in extracted helper functions (default: false)
helper_docstrings = false

# Retry counts for extraction and LLM verification failures
extraction_retries = 2
llm_verify_retries = 2

# FileLimiter: additional retry attempts after an LLM-related failure (default: 2)
file_limiter_retries = 2

# Rate-limit (HTTP 429) retry settings.
# rate_limit_retries: additional attempts after a 429 response (default: 6 = 7 total).
#   Set to 0 to disable retries and raise an error immediately on the first 429.
# rate_limit_backoff: initial wait in seconds before the first retry (default: 20.0).
#   Each retry doubles the delay: 20, 40, 80, 160, 320, 640 seconds.
rate_limit_retries = 6
rate_limit_backoff = 20.0

# FileLimiter: recursively split newly-created files that are still over the
# limit (default: true). The recursion terminates when each new file is either
# under the limit, aborted (cannot be split), or produces no further oversized files.
file_limiter_recursive = true

# FileLimiter: how to update @patch string literals in test files after
# FileLimiter moves entities to new sub-modules (default: "basic").
#
# "ignore"  — Leave all @patch strings as-is.
# "basic"   — Scan every *.py file and update patch strings that are
#             unambiguous: entity imported by exactly one new sub-module,
#             or resolved via call-graph traversal to a single sub-module.
#             Import aliases from the original file that land in exactly one
#             new file are also updated.
# "rewrite" — Performs "basic" first, then uses an LLM to resolve forking
#             cases (entities imported by multiple new files). A verify step
#             checks each proposal; failed functions are retried up to
#             patch_update_retries additional times before being skipped.
#             Also handles @patch(module.CONSTANT), with patch(...) forms,
#             and patch.multiple(...) (including with patch.multiple(...)).
# file_limiter_patch_update = "basic"

# FileLimiter: additional LLM verify+retry attempts per function in
# "rewrite" mode (default: 2 = three total attempts; 0 = no retry).
# patch_update_retries = 2

# BFS call-graph traversal limits for patch_update "basic" and "rewrite" modes.
# Increase callgraph_max_depth for codebases with deep call chains.
# Increase callgraph_max_modules for large codebases with wide import graphs.
# callgraph_max_depth = 12
# callgraph_max_modules = 50

# Timing output printed at the end of each run (default: "detailed").
# "off"      — no timing output.
# "basic"    — total run time, total LLM time, and total token counts.
# "detailed" — adds per-call timing lines during the run (in verbose mode)
#              plus per-call-type, per-refactor, and per-file breakdowns.
# timing = "detailed"

# Run only specific refactors (default: run all).
# Valid names: "if_not_else", "duplicate_extractor", "function_splitter",
# "tuple_dataclass", "file_limiter", "match_function"
# ("match_function" controls the sub-pass inside duplicate_extractor that
# replaces inline code with calls to existing functions; only takes effect
# when duplicate_extractor is also enabled.)
# enabled_refactors = ["function_splitter", "file_limiter"]

# Always skip specific refactors (ignored when enabled_refactors is set).
# disabled_refactors = ["file_limiter"]
```

### API Keys

Set the appropriate environment variable for your chosen provider:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # for provider = "anthropic"
export OPENAI_API_KEY=sk-...          # for provider = "openai"
export DEEPSEEK_API_KEY=sk-...        # for provider = "deepseek"
export MOONSHOT_API_KEY=sk-...        # for provider = "moonshot"
export GEMINI_API_KEY=...             # for provider = "gemini"
export MISTRAL_API_KEY=...            # for provider = "mistral"
export ZAI_API_KEY=...                # for provider = "zai"
# LM Studio (provider = "lmstudio") and Ollama (provider = "ollama") run
# locally and require no API key.
```

### LM Studio

Point crispen at a running [LM Studio](https://lmstudio.ai) local server. LM Studio exposes an OpenAI-compatible API, which crispen uses directly:

```toml
[tool.crispen]
provider = "lmstudio"
model = "your-loaded-model-name"
# base_url defaults to "http://localhost:1234/v1"; override if needed:
# base_url = "http://localhost:8080/v1"
```

No API key is required — LM Studio does not authenticate requests.

### Ollama

Point crispen at a running [Ollama](https://ollama.com) local server. Ollama exposes an OpenAI-compatible API, which crispen uses directly:

```toml
[tool.crispen]
provider = "ollama"
model = "your-pulled-model-name"
# base_url defaults to "http://localhost:11434/v1"; override if needed:
# base_url = "http://localhost:11435/v1"
```

No API key is required — Ollama does not authenticate requests.

### Escaping refactors

Add a `# crispen: skip` comment to protect specific code from every refactor, or `# crispen: skip=<name>[,<name>...]` to protect it from only the named refactor(s) (same names as `enabled_refactors`/`disabled_refactors`: `if_not_else`, `duplicate_extractor`, `function_splitter`, `tuple_dataclass`, `file_limiter`). The marker attaches to whatever starts on its own line — either as a trailing comment on that exact line, or as a standalone comment on the line(s) directly above it (blank lines and decorator lines in between are transparent, so a marker above a stack of decorators still attaches to the `def`/`class` below them):

```python
if not user.is_active:  # crispen: skip
    deactivate()
else:
    activate()

# crispen: skip=file_limiter
def load_legacy_config():
    ...

return (id, name, email, created_at)  # crispen: skip=tuple_dataclass
```

For `# crispen: skip` on a function or class, FileLimiter treats it the same as code that predates the diff — it never proposes moving it to a new file. `duplicate_extractor` and `tuple_dataclass` match sub-statement sequences and individual tuple literals respectively, so their marker must sit on the specific statement to protect, not just anywhere in the enclosing function.

`# crispen: skip-file`, placed anywhere in the file, opts the whole file out of every refactor (including FileLimiter) before it's even read for changes. There is no region/block form (`skip-start` / `skip-end`) — each protected spot needs its own marker.

## Refactors

### 1. File limiter

**Splits files that exceed the line-count limit into smaller sibling modules.**

When a file in the diff exceeds `max_file_lines` (default: 1000) after all other refactors have run, crispen splits it. It classifies every top-level entity by whether it was added, modified, or left unchanged by the diff, builds a dependency graph to find groups that can safely move together, and asks the LLM to assign each group to a new file. The original file is updated with `from .module import name` re-exports so existing callers are unaffected.

The algorithm:
1. Parse all top-level entities (functions, classes, and statement blocks) from the post-refactor source.
2. Classify each entity as **NEW** (added by the diff), **MODIFIED** (existed before and changed), or **UNMODIFIED** (existed before and unchanged).
3. Build a name-reference dependency graph and compute strongly connected components (SCCs). Entities in the same SCC must be moved together.
4. SCCs containing any **UNMODIFIED** entity must stay (Set 1). SCCs of **NEW** entities can be freely moved (Set 2). SCCs of purely **MODIFIED** entities may be migrated (Set 3) — the LLM decides which.
5. Ask the LLM to assign each movable SCC group to a target filename. Groups with related purposes may share a file.
6. Generate the new files, add `from .module import name` re-exports to the original, and verify every entity's source is preserved before writing.

**Whole-file mode (subdir split):** When the diff covers every line of the file — e.g. a brand-new file — crispen places the new sibling files in a subdirectory named after the module (e.g. `service/*.py` for `service.py`). For non-test files, a `service/__init__.py` is generated that re-exports the public API so callers need no updates. For test files, the original `test_service.py` keeps re-export stubs so pytest can still find the tests. Test classes that contain `test_` methods are not re-exported (re-importing them would cause pytest to discover and run every test twice).

**Script entry points:** When a non-test file contains `if __name__ == '__main__':`, the original file is kept on disk as the runnable entry point (with re-export stubs) rather than being replaced by `subdir/__init__.py`. The subdirectory is named with a `_lib`, `_helpers`, `_impl`, `_internals`, or `_support` suffix to avoid shadowing the original module name.

**Directories with dashes:** Crispen skips files located under directories whose names contain dashes (e.g. `my-project/service.py`), because dashes are illegal in Python package names and would produce a `SyntaxError` in generated imports. Rename the directory to use underscores first.

**Before** (`service.py`, 1200 lines):
```python
import json

class Config:
    # ... 80 lines ...

class DataStore:
    # ... 200 lines ...

def fetch_records(store, query):
    # ... 60 lines ...

def export_csv(records, path):
    # ... 40 lines ...
```

**After:**
```python
# service.py (updated, with re-exports)
import json
from .models import Config, DataStore  # fmt: skip # noqa: F401, E501
from .utils import export_csv, fetch_records  # fmt: skip # noqa: F401, E501
```

```python
# service/models.py (new file)
import json

class Config:
    # ... 80 lines ...

class DataStore:
    # ... 200 lines ...
```

```python
# service/utils.py (new file)
from .models import DataStore

def fetch_records(store, query):
    # ... 60 lines ...

def export_csv(records, path):
    # ... 40 lines ...
```

Safety checks applied before writing:
- Every entity's source (minus import lines) must appear verbatim in the combined output.
- The proposed split must not introduce circular file imports.

Configuration:
- `max_file_lines` — file line count threshold (default: 1000). Set to `0` to disable.
- `file_limiter_subdir_split` — use a subdirectory for whole-file diffs (default: `true`).
- `file_limiter_retries` — additional LLM retry attempts on failure (default: 2).
- `file_limiter_recursive` — recursively split newly-created files that are still over the limit (default: `true`).
- `file_limiter_pytest_conftest` — route fixtures split out of test files to `conftest.py` so pytest auto-discovers them without imports (avoids F401/F811 warnings). Set to `false` if not using pytest or using custom fixture decorators (default: `true`).
- `file_limiter_reexports` — controls when re-export stubs are added to the original file for public names moved to new files: `"always"` (every public name), `"application"` (all non-test files, no test files), or `"imported"` (only when the name is imported from this module elsewhere in the project — the same rule used for private names). Default: `"imported"`.
- `file_limiter_patch_update` — controls how `@patch` string literals are updated after FileLimiter moves entities to new sub-modules:
  - `"ignore"` — leave all patch strings as-is.
  - `"basic"` — scan every `*.py` file in the repo and update patch strings that are unambiguous (non-forking): when the entity is imported by exactly one new sub-module, or resolved via call-graph traversal to a single sub-module. Import aliases from the original file that appear in exactly one new file are also updated. Default.
  - `"rewrite"` — performs `"basic"` updates first, then uses an LLM to resolve the remaining forking cases (entities imported by multiple new files). The LLM receives the original file, the diff, all new sub-files, and one test function at a time and proposes updated `@patch` strings. A second LLM verify step checks each proposal; failed functions are retried up to `patch_update_retries` additional times before being skipped. Also handles `@patch(module.CONSTANT)` attribute-access and `with patch(...)` context-manager forms.
- `patch_update_retries` — additional LLM verify+retry attempts per function in `"rewrite"` mode. `0` means no retry: a function rejected by the verify LLM is skipped. Default: `2` (three total attempts).
- `callgraph_max_depth` — maximum BFS hops from a test function to a target sub-module during call-graph traversal (`"basic"` and `"rewrite"` modes). When the limit is reached the path is left unresolved and a warning is printed to stderr. Increase for codebases with very deep call chains. Default: `12`.
- `callgraph_max_modules` — maximum distinct modules visited per resolution attempt during call-graph traversal. When the limit is reached the path is left unresolved and a warning is printed to stderr. Increase for large codebases with wide import graphs. Default: `50`.

---

### 2. Duplicate code extraction

**Extracts duplicate code blocks into shared helper functions using an LLM.**

Crispen scans the changed functions for repeated sequences of statements. When a duplicate group is found, it calls the LLM to produce a single extracted helper function with an appropriate name, and replaces each occurrence with a call to that helper.

The algorithm:
1. Hashes each statement in a function by its AST structure (ignoring whitespace and comments).
2. Finds repeated subsequences above a minimum weight threshold.
3. Asks the LLM if the matching sections of code are a semantic match ("veto check"), and requests any pitfalls to note for the extraction step. These notes are passed to every subsequent extraction attempt.
4. If accepted, asks the LLM to write the helper, determine its parameters, and update the call sites.
5. Runs a series of algorithmic checks to verify the code change. A failure triggers a retry (step 4, up to `extraction_retries`), including a note to the LLM about the verification failure.
6. After passing the algorithmic checks, asks the LLM to verify the output. A failure triggers a retry (step 4, up to `llm_verify_retries`), including both the previous code change and detailed feedback from the LLM verification step.
7. If accepted, validates the output syntactically and with pyflakes before applying it.

**Before:**
```python
def process_users(users, archived_users):
    for user in users:
        name = user["name"].strip().lower()
        email = user["email"].strip().lower()
        db.save(name, email)

    for person in archived_users:
        full_name = person["name"].strip().lower()
        contact = person["email"].strip().lower()
        archive.save(full_name, contact)
```

Note that the two blocks use different variable names (`name`/`email` vs `full_name`/`contact`, `user` vs `person`). The algorithm detects the structural match regardless, and the LLM generalises the variable names into appropriate parameter names.

**After:**
```python
def _normalize_contact(record):
    name = record["name"].strip().lower()
    email = record["email"].strip().lower()
    return name, email

def process_users(users, archived_users):
    for user in users:
        name, email = _normalize_contact(user)
        db.save(name, email)

    for person in archived_users:
        full_name, contact = _normalize_contact(person)
        archive.save(full_name, contact)
```

Configuration:
- `min_duplicate_weight` — minimum "weight" (sum of statement sizes) a repeated group must have to be extracted (default: 3).
- `max_duplicate_seq_len` — maximum number of statements in a duplicate sequence (default: 8).
- `extraction_retries` — how many times to retry after an algorithmic check fails (default: 2).
- `llm_verify_retries` — how many times to retry after the LLM verification step rejects the output (default: 2).

---

### 3. Match existing function

**Replaces a code block with a call to an existing function that performs the same operation.**

When a block of code in the diff is semantically equivalent to the body of an existing function in the same file, crispen replaces the inline block with a call to that function. This is the complement of DuplicateExtractor: instead of creating a new helper, it recognises that one already exists.

The algorithm:
1. Fingerprints every function body in the file by its normalised AST structure (ignoring variable names, whitespace, and comments).
2. For each statement sequence in the diff, checks whether its fingerprint matches any function body.
3. Asks the LLM to verify the match is semantically valid and not a coincidental structural similarity.
4. If confirmed, asks the LLM to generate the correct call expression (mapping arguments as needed) and replaces the block.

**Before:**
```python
def _fetch_json(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def sync_orders():
    # Inline copy of _fetch_json's body with different variable names:
    resp = requests.get(orders_url, headers=api_headers)
    resp.raise_for_status()
    orders = resp.json()
    process(orders)
```

**After:**
```python
def _fetch_json(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def sync_orders():
    orders = _fetch_json(orders_url, api_headers)
    process(orders)
```

The LLM veto step ensures the replacement is only applied when the semantics genuinely match — structural similarity alone is not enough.

---

### 4. Function splitter

**Splits functions that exceed the line-count limit into smaller helpers.**

When a function in the changed region is too long (more than `max_function_length` body lines, default 75), crispen splits it. It identifies the best split point — the one that minimises the number of free variables passed to the helper — and asks the LLM to name the extracted helper function.

The extracted tail becomes a private helper (`_helper_name`) placed immediately after the original function. If the tail references `self`, the helper is extracted as a regular instance method; otherwise it is extracted as a `@staticmethod` (for class methods) or a module-level function.

**Before:**
```python
def build_report(config, data):
    # ... 40 lines of setup ...
    headers = compute_headers(config)
    rows = []
    for item in data:
        row = format_row(item, headers)
        validate_row(row)
        rows.append(row)
    totals = compute_totals(rows)
    footer = format_footer(totals)
    return assemble(headers, rows, footer)
```

**After:**
```python
def build_report(config, data):
    # ... 40 lines of setup ...
    headers = compute_headers(config)
    return _format_rows_and_assemble(headers, data)


def _format_rows_and_assemble(headers, data):
    rows = []
    for item in data:
        row = format_row(item, headers)
        validate_row(row)
        rows.append(row)
    totals = compute_totals(rows)
    footer = format_footer(totals)
    return assemble(headers, rows, footer)
```

Functions are skipped if they are:
- `async` functions
- Generator functions (contain `yield`)
- Functions with nested `def` statements (closures)

Safety checks applied before writing:
- The rewritten source must compile without `SyntaxError`.
- pyflakes must not report any new `UndefinedName` warnings after the split.

Configuration:
- `max_function_length` — maximum allowed body lines (default: 75).
- `helper_docstrings` — whether to include a docstring in extracted helper functions (default: `false`).

---

### 5. Tuple Return to Dataclass

**Replaces large tuple return values with `@dataclass` instances and updates call sites.**

Functions that return large tuples (4+ elements by default) are difficult to read at call sites — callers must remember which index means what. Crispen replaces the tuple literal with a named `@dataclass` constructor call, automatically generates the dataclass definition, and rewrites every tuple-unpacking call site to use the dataclass's named attributes.

Only fires when:
- The tuple is inside a `return` statement (not a function argument).
- Every in-file caller of the function uses tuple-unpacking assignment (`a, b = func()`).
- The tuple has at least `min_tuple_size` elements (default 4).

**Before:**
```python
def get_metrics(data):
    count = len(data)
    total = sum(data)
    average = total / count
    peak = max(data)
    return count, total, average, peak

# elsewhere:
count, total, average, peak = get_metrics(data)
```

**After:**
```python
from dataclasses import dataclass
from typing import Any

@dataclass
class GetMetricsResult:
    count: Any
    total: Any
    average: Any
    peak: Any


def get_metrics(data):
    count = len(data)
    total = sum(data)
    average = total / count
    peak = max(data)
    return GetMetricsResult(count=count, total=total, average=average, peak=peak)

# elsewhere:
_ = get_metrics(data)
count = _.count
total = _.total
average = _.average
peak = _.peak
```

Field names are inferred from unpacking assignments at call sites (e.g., `count, total, average, peak = get_metrics(data)`), from the variable names in the tuple itself, or defaulted to `field_0`, `field_1`, etc. The intermediate variable name (`_`, `_result`, or the snake_case dataclass name) is chosen to avoid collisions with existing names in the file.

Configuration:
- `min_tuple_size` — minimum tuple element count to trigger replacement (default: 4).

---

### 6. Flip negated if/else

**Flips negated `if/else` conditions to eliminate the `not`.**

When an `if not condition:` has an `else` clause, crispen rewrites it to `if condition:` and swaps the two branches. This eliminates a layer of logical indirection and makes intent clearer.

**Before:**
```python
if not is_valid(data):
    handle_error(data)
else:
    process(data)
```

**After:**
```python
if is_valid(data):
    process(data)
else:
    handle_error(data)
```

Skipped when there is no `else` clause, or when the `else` is an `elif` chain.

## Architecture

```
stdin (unified diff)
        │
        ▼
crispen/cli.py         # Entry point: reads stdin, calls parse_diff then run_engine
        │
        ├── crispen/diff_parser.py   # Parses diff → Dict[filepath, List[(start, end)]]
        │
        └── crispen/engine.py        # Loads files, runs all refactors, writes back
                │
                ├── crispen/refactors/
                │       ├── base.py                # Refactor base class (libcst.CSTTransformer)
                │       ├── if_not_else.py         # if not x: A else B  →  if x: B else A
                │       ├── tuple_dataclass.py     # Large tuple returns → @dataclass
                │       ├── caller_updater.py      # Update tuple-unpacking call sites
                │       ├── duplicate_extractor.py # Extract duplicate blocks
                │       └── function_splitter.py   # Split oversized functions
                │
                ├── crispen/file_limiter/
                │       ├── runner.py              # Orchestrates FileLimiter phases
                │       ├── classifier.py          # Classify entities (Set 1/2/3)
                │       ├── advisor.py             # LLM placement planner
                │       ├── code_gen.py            # Generate split file contents
                │       ├── entity_parser.py       # Parse top-level entities
                │       └── dep_graph.py           # Dependency graph + SCC finder
                │
                ├── crispen/patch_updater.py       # Basic (non-forking) @patch string updater
                └── crispen/patch_rewriter.py      # LLM-powered @patch rewriter (rewrite mode)
```

### Adding a new refactor

1. Create `crispen/refactors/my_refactor.py` subclassing `Refactor` from `base.py`.
2. Override `leave_*` methods; guard with `self._in_changed_range(original_node)`.
3. Append change descriptions to `self.changes_made`.
4. Register the class in `engine.py`'s `_REFACTORS` list.
5. Add `tests/test_my_refactor.py` — 100% branch coverage is enforced.
