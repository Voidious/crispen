from __future__ import annotations
import ast
from typing import Optional


def _safe_parse_ast(source_code: str) -> Optional[ast.AST]:
    try:
        return ast.parse(source_code)
    except SyntaxError:
        return None
