from __future__ import annotations
import ast
from typing import Optional, Tuple


def _parse_source_and_init_names(source: str) -> Optional[Tuple[ast.AST, set]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    return tree, set()
