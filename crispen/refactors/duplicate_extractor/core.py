from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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


def _overlaps_diff(seq: _SeqInfo, changed_ranges: List[Tuple[int, int]]) -> bool:
    return any(
        seq.start_line <= r_end and seq.end_line >= r_start
        for r_start, r_end in changed_ranges
    )


def _filter_maximal_groups(groups: List[List[_SeqInfo]]) -> List[List[_SeqInfo]]:
    """Return only maximal groups, discarding those overlapping a larger group.

    Groups are sorted by their longest sequence (descending) and greedily selected:
    a group is kept only if none of its sequences overlap an already-claimed line range.
    This prevents multiple helpers being extracted for overlapping spans, where the
    smaller extractions would end up unused after the larger one is applied.
    """
    sorted_groups = sorted(
        groups,
        key=lambda g: max(s.end_line - s.start_line for s in g),
        reverse=True,
    )
    claimed: List[Tuple[int, int]] = []
    result = []
    for group in sorted_groups:
        overlaps = any(
            seq.start_line <= c_end and seq.end_line >= c_start
            for seq in group
            for c_start, c_end in claimed
        )
        if not overlaps:
            result.append(group)
            for seq in group:
                claimed.append((seq.start_line, seq.end_line))
    return result


def _has_internal_overlap(seqs: List[_SeqInfo]) -> bool:
    """Return True if any two sequences in the group overlap each other.

    Overlapping sequences within a group indicate sequential repetition
    (e.g. [A,B] and [B,C] both matching) rather than true duplication at
    distinct call sites.  Extracting a helper from such a group would leave
    part of the original pattern unreplaced.
    """
    sorted_seqs = sorted(seqs, key=lambda s: s.start_line)
    for i in range(len(sorted_seqs) - 1):
        if sorted_seqs[i].end_line >= sorted_seqs[i + 1].start_line:
            return True
    return False


def _find_duplicate_groups(
    sequences: List[_SeqInfo],
    changed_ranges: List[Tuple[int, int]],
    max_groups: int = 5,
) -> List[List[_SeqInfo]]:
    by_fp: Dict[str, List[_SeqInfo]] = {}
    for seq in sequences:
        by_fp.setdefault(seq.fingerprint, []).append(seq)
    groups = []
    for seqs in by_fp.values():
        if len(seqs) < 2:
            continue
        if not any(_overlaps_diff(s, changed_ranges) for s in seqs):
            continue
        if _has_internal_overlap(seqs):
            continue
        groups.append(seqs)
    groups = _filter_maximal_groups(groups)
    return groups[:max_groups]
