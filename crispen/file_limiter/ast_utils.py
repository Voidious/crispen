from __future__ import annotations
import ast
import re
from typing import Dict, List, Optional, Set


def _try_parse_ast(source: str) -> Optional[ast.AST]:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _collect_referenced_name_loads(
    entity_names: List[str],
    entity_source_map: Dict[str, str],
) -> Set[str]:
    referenced: Set[str] = set()
    for name in entity_names:
        src = entity_source_map.get(name, "")
        referenced |= _collect_name_loads(src)
    return referenced


# Matches any line that is an import statement (plain or from-import).
_IMPORT_LINE_RE = re.compile(r"^(import\s+|from\s+\S.*\s+import\s+)")

# Matches a `from __future__ import …` line (with optional trailing newline).
_FUTURE_IMPORT_LINE_RE = re.compile(r"^from __future__ import .*\n?", re.MULTILINE)


def _collect_name_loads(source: str) -> Set[str]:
    """Return all Name loads referenced in *source*."""
    tree = _try_parse_ast(source)
    if tree is None:
        return set()
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
    return names
