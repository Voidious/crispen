from __future__ import annotations
import ast
import textwrap
from .ast_helpers import _ASTNormalizer


def _normalize_source(source: str) -> str:
    """Return a normalized fingerprint of source code."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return source
    normalizer = _ASTNormalizer()
    normalized = normalizer.visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.unparse(normalized)
