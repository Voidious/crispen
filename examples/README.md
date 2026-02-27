# Crispen Refactor Examples

Each subdirectory contains a set of hand-crafted examples for one refactor type.
Every example has three files:

| File | Purpose |
|------|---------|
| `input.py` | The Python source on disk when crispen runs |
| `diff.patch` | The unified diff that marks which lines are "changed" (determines scope) |
| `expected.py` | The expected output after crispen applies the refactor |

The test suite `tests/test_examples.py` drives each example end-to-end.
LLM calls are mocked so the suite runs offline without API keys.

## Refactor types covered

| Directory | Refactor | LLM? |
|-----------|----------|-------|
| `if_not_else/` | Flip `if not x: A else B` → `if x: B else A` | No |
| `duplicate_extraction/` | Extract repeated code blocks into helpers | Yes |
| `match_existing_function/` | Replace block with call to existing function | Yes |
| `function_splitter/` | Split oversized functions | Yes |
| `tuple_dataclass/` | Convert large tuple returns to `@dataclass` | No |

## Running the examples

```bash
uv run pytest tests/test_examples.py -v
```

The examples suite runs as part of the normal test suite (including pre-commit).
