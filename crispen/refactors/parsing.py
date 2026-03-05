from __future__ import annotations
import ast
import textwrap


def _parse_validation_result(result):
    if result is not None:
        return (
            result["is_valid_duplicate"],
            result.get("reason", ""),
            result.get("extraction_notes", ""),
        )
    return False, "no tool response", ""


def _parse_block_source(source: str) -> ast.AST | None:
    try:
        return ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
