# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Crispen Does

Crispen is a Python code refactoring CLI tool. It reads a unified diff from stdin, identifies which lines changed, and applies AST-based refactors only to the changed regions of affected files, writing the modified files back in place.

**Usage:** `git diff | crispen` or `git diff HEAD | crispen`

## Commands

```bash
# Install in development mode
uv sync

# Run all tests (requires 100% branch coverage)
uv run pytest

# Run a single test file
uv run pytest tests/test_engine.py

# Run a single test by name
uv run pytest tests/test_engine.py::test_applies_refactor_and_writes

# Format code
uv run black .

# Lint
uv run flake8

# Run pre-commit hooks (runs black, flake8, and pytest with coverage)
uv run pre-commit run --all-files
```

The pre-commit hook runs `uv run pytest` on every commit, which enforces 100% branch coverage via `pyproject.toml`.

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
                        ├── base.py             # Refactor base class (libcst.CSTTransformer)
                        ├── if_not_else.py      # `if not x: A else B` → `if x: B else A`
                        └── tuple_dataclass.py  # Large tuple literals → @dataclass instances
```

### Key design decisions

- **Line-range scoping**: The diff parser converts added lines into `(start, end)` line ranges per file. Each refactor receives these ranges and calls `self._in_changed_range(node)` to skip nodes outside the diff.
- **libcst**: All AST work uses `libcst` (not the stdlib `ast` module) because it preserves formatting and supports round-trip code generation. Every refactor is a `cst.CSTTransformer` using `MetadataWrapper` + `PositionProvider` for line number access.
- **Sequential refactors**: `engine.py` applies refactors one at a time in the `_REFACTORS` list. Each refactor receives the output of the previous one as its input source.
- **Verification step**: After each refactor, the engine compiles the output with `compile()` to confirm it is valid Python before writing it back.

### Adding a new refactor

1. Create `crispen/refactors/my_refactor.py` subclassing `Refactor` from `base.py`.
2. Override `leave_*` methods; guard with `self._in_changed_range(original_node)`.
3. Append change descriptions to `self.changes_made`.
4. Register the class in `engine.py`'s `_REFACTORS` list.
5. Add a test file `tests/test_my_refactor.py` — 100% branch coverage is enforced.
