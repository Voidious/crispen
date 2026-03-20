from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import ast
import textwrap
import libcst as cst


_MODEL = "claude-sonnet-4-6"
_MIN_WEIGHT = 3
_MAX_SEQ_LEN = 8


def _node_weight(node: cst.CSTNode) -> int:
    """Recursive statement weight: count all semantic statement units."""
    if isinstance(node, cst.SimpleStatementLine):
        return len(node.body)
    if isinstance(node, cst.IndentedBlock):
        return sum(_node_weight(s) for s in node.body)
    if isinstance(node, cst.Else):
        return _node_weight(node.body)
    if isinstance(node, cst.Finally):
        return _node_weight(node.body)
    if isinstance(node, (cst.FunctionDef, cst.ClassDef)):
        return 1
    if not isinstance(node, (cst.If, cst.For, cst.While, cst.Try, cst.With)):
        return 0
    weight = 1 + _node_weight(node.body)
    orelse = getattr(node, "orelse", None)
    if orelse is not None:
        weight += _node_weight(orelse)
    finalbody = getattr(node, "finalbody", None)
    if finalbody is not None:
        weight += _node_weight(finalbody)
    if isinstance(node, cst.Try):
        for handler in node.handlers:
            weight += _node_weight(handler.body)
    return weight


def _sequence_weight(stmts: List[cst.BaseStatement]) -> int:
    return sum(_node_weight(s) for s in stmts)


def _has_def(stmts: List[cst.BaseStatement]) -> bool:
    """Return True if any top-level statement is a function or class definition."""
    return any(isinstance(s, (cst.FunctionDef, cst.ClassDef)) for s in stmts)


class _ASTNormalizer(ast.NodeTransformer):
    """Replace assignment-target Names with positional placeholders."""

    def __init__(self) -> None:
        self._map: Dict[str, str] = {}
        self._counter = 0

    def _placeholder(self, name: str) -> str:
        if name not in self._map:
            self._map[name] = f"_v{self._counter}"
            self._counter += 1
        return self._map[name]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if isinstance(node.ctx, (ast.Store, ast.Load)):
            return ast.Name(id=self._placeholder(node.id), ctx=node.ctx)
        return node


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


@dataclass
class _SeqInfo:
    stmts: List[cst.BaseStatement]
    start_line: int
    end_line: int
    scope: str
    source: str
    fingerprint: str
    class_scope: Optional[str] = None  # enclosing class name, or None if module-level


@dataclass
class _FunctionInfo:
    name: str
    source: str  # raw source of complete function definition
    scope: str  # "<module>" or enclosing class name
    body_source: str  # raw source of the function body (indented)
    body_stmt_count: int  # number of top-level statements in the body
    params: List[str]  # positional parameter names (empty → no-arg function)


def _seq_source_contains_yield(source: str) -> bool:
    """Return True if *source* contains ``yield`` or ``yield from`` outside
    any nested function definition.

    Sequences with a yield cannot be safely extracted into a plain helper
    function: extraction would make the helper a generator, forcing call sites
    to iterate via ``for``/``async for`` instead of calling it directly.  This
    is a semantic transformation (e.g. ``async with X as c: yield c`` →
    ``async for c in helper(): yield c``) that the extractor must not attempt.
    """
    wrapped = "def _f():\n" + textwrap.indent(textwrap.dedent(source), "    ")
    try:
        tree = ast.parse(wrapped)
    except SyntaxError:
        return False
    if not tree.body or not isinstance(
        tree.body[0], ast.FunctionDef
    ):  # pragma: no cover
        return False

    def _walk(nodes):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # don't cross into nested scope
            if isinstance(node, (ast.Yield, ast.YieldFrom)):
                return True
            if _walk(ast.iter_child_nodes(node)):
                return True
        return False

    return _walk(tree.body[0].body)
