from __future__ import annotations
import ast
from typing import Dict, List, Tuple
import libcst as cst
from .models import _FunctionInfo, _SeqInfo
from .normalization import _normalize_source


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


def _collect_called_names(source: str) -> set:
    """Return a set of all names called (as functions) in *source*.

    Uses ast.parse + ast.walk to find all ast.Call nodes.  Returns the
    called name: func.id for ast.Name callees, func.attr for ast.Attribute
    callees.  On SyntaxError, returns an empty set.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def _build_function_body_fps(
    all_functions: List[_FunctionInfo],
    called_names: set,
) -> Dict[str, _FunctionInfo]:
    """Map normalized body fingerprint → _FunctionInfo for called functions.

    Only functions whose name appears in *called_names* are indexed, since
    only those could be the target of a "replace with existing function" edit.
    """
    fps: Dict[str, _FunctionInfo] = {}
    for func in all_functions:
        if func.name in called_names:
            fp = _normalize_source(func.body_source)
            fps[fp] = func
    return fps


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
        groups.append(seqs)
    groups = _filter_maximal_groups(groups)
    return groups[:max_groups]
