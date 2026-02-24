# Crispen

Crispen is a Python code refactoring CLI tool. It reads a unified diff from stdin, identifies which lines changed, and applies a set of automated refactors only to the changed regions of affected files, writing the modified files back in place.

It uses [libcst](https://github.com/Instagram/LibCST) for format-preserving AST transformations and an LLM for intelligent, context-aware changes. Supported providers: Anthropic Claude, OpenAI, DeepSeek, Moonshot/Kimi, and LM Studio (local).

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
```

Crispen prints a summary of every change it applies, then writes the modified files back in place.

## Configuration

Crispen reads configuration from `[tool.crispen]` in `pyproject.toml`, with optional overrides in `.crispen.toml` in the project root.

```toml
[tool.crispen]
# LLM provider: "anthropic" (default), "openai", "deepseek", "moonshot", or "lmstudio"
provider = "anthropic"

# LLM model to use (default: "claude-sonnet-4-6")
model = "claude-sonnet-4-6"

# Optional base URL override for OpenAI-compatible providers.
# Useful for LM Studio on a non-default port, or other self-hosted endpoints.
# base_url = "http://localhost:1234/v1"

# FunctionSplitter: max function body lines before splitting (default: 75)
max_function_length = 75

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
extraction_retries = 1
llm_verify_retries = 1
```

### API Keys

Set the appropriate environment variable for your chosen provider:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # for provider = "anthropic"
export OPENAI_API_KEY=sk-...          # for provider = "openai"
export DEEPSEEK_API_KEY=sk-...        # for provider = "deepseek"
export MOONSHOT_API_KEY=sk-...        # for provider = "moonshot"
# LM Studio (provider = "lmstudio") runs locally and requires no API key.
```

### LM Studio

Point crispen at a running [LM Studio](https://lmstudio.ai) local server:

```toml
[tool.crispen]
provider = "lmstudio"
model = "your-loaded-model-name"
# base_url defaults to "http://localhost:1234/v1"; override if needed:
# base_url = "http://localhost:8080/v1"
```

No API key is required — LM Studio does not authenticate requests.

## Refactors

### 1. IfNotElse

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

---

### 2. Tuple Return to Dataclass

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

### 3. DuplicateExtractor

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
def process_users(users):
    for user in users:
        name = user["name"].strip().lower()
        email = user["email"].strip().lower()
        db.save(name, email)

    for user in archived_users:
        name = user["name"].strip().lower()
        email = user["email"].strip().lower()
        archive.save(name, email)
```

**After:**
```python
def _normalize_user(user):
    name = user["name"].strip().lower()
    email = user["email"].strip().lower()
    return name, email

def process_users(users):
    for user in users:
        name, email = _normalize_user(user)
        db.save(name, email)

    for user in archived_users:
        name, email = _normalize_user(user)
        archive.save(name, email)
```

Configuration:
- `min_duplicate_weight` — minimum "weight" (sum of statement sizes) a repeated group must have to be extracted (default: 3).
- `max_duplicate_seq_len` — maximum number of statements in a duplicate sequence (default: 8).
- `extraction_retries` — how many times to retry after an algorithmic check fails (default: 1).
- `llm_verify_retries` — how many times to retry after the LLM verification step rejects the output (default: 1).

---

### 4. MatchExistingFunction

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

### 5. FunctionSplitter

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
- `helper_docstrings` — whether to add a `"""..."""` stub docstring to helpers (default: `false`).

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
                └── crispen/refactors/
                        ├── base.py                # Refactor base class (libcst.CSTTransformer)
                        ├── if_not_else.py         # if not x: A else B  →  if x: B else A
                        ├── tuple_dataclass.py     # Large tuple returns → @dataclass
                        ├── caller_updater.py      # Update tuple-unpacking call sites
                        ├── duplicate_extractor.py # Extract duplicate blocks
                        └── function_splitter.py   # Split oversized functions
```

### Adding a new refactor

1. Create `crispen/refactors/my_refactor.py` subclassing `Refactor` from `base.py`.
2. Override `leave_*` methods; guard with `self._in_changed_range(original_node)`.
3. Append change descriptions to `self.changes_made`.
4. Register the class in `engine.py`'s `_REFACTORS` list.
5. Add `tests/test_my_refactor.py` — 100% branch coverage is enforced.
