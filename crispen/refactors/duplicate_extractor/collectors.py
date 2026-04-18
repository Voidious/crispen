from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import ast
import textwrap
from libcst.metadata import PositionProvider
import libcst as cst
from .common import (
    _FunctionInfo,
    _MAX_SEQ_LEN,
    _MIN_WEIGHT,
    _SeqInfo,
    _has_def,
    _normalize_source,
    _sequence_weight,
)


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


class _SequenceCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        source_lines: List[str],
        max_seq_len: int = _MAX_SEQ_LEN,
        min_weight: int = _MIN_WEIGHT,
    ) -> None:
        self.sequences: List[_SeqInfo] = []
        self._scope_stack: List[str] = ["<module>"]
        self._class_stack: List[str] = []
        self._source_lines = source_lines
        self._max_seq_len = max_seq_len
        self._min_weight = min_weight

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        self._scope_stack.append(node.name.value)
        return None

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._scope_stack.pop()

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self._scope_stack.append(node.name.value)
        self._class_stack.append(node.name.value)
        return None

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self._scope_stack.pop()
        self._class_stack.pop()

    def _process_body(self, body: Sequence) -> None:
        stmt_info: List[Tuple[cst.BaseStatement, int, int]] = []
        for stmt in body:
            try:
                pos = self.get_metadata(PositionProvider, stmt)
                stmt_info.append((stmt, pos.start.line, pos.end.line))
            except KeyError:  # pragma: no cover
                continue

        n = len(stmt_info)
        scope = self._scope_stack[-1]
        class_scope = self._class_stack[-1] if self._class_stack else None
        for start_i in range(n):
            for end_i in range(
                start_i + 1, min(start_i + self._max_seq_len + 1, n + 1)
            ):
                window: List[cst.BaseStatement] = [
                    s[0] for s in stmt_info[start_i:end_i]
                ]
                if _has_def(window):
                    continue
                if _sequence_weight(window) < self._min_weight:
                    continue
                start_line = stmt_info[start_i][1]
                end_line = stmt_info[end_i - 1][2]
                seq_source = "".join(self._source_lines[start_line - 1 : end_line])
                if _seq_source_contains_yield(seq_source):
                    continue
                self.sequences.append(
                    _SeqInfo(
                        stmts=window,
                        start_line=start_line,
                        end_line=end_line,
                        scope=scope,
                        source=seq_source,
                        fingerprint=_normalize_source(seq_source),
                        class_scope=class_scope,
                    )
                )

    def visit_Module(self, node: cst.Module) -> Optional[bool]:
        self._process_body(node.body)
        return None

    def visit_IndentedBlock(self, node: cst.IndentedBlock) -> Optional[bool]:
        self._process_body(node.body)
        return None


class _FunctionCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, source_lines: List[str]) -> None:
        self.functions: List[_FunctionInfo] = []
        self._scope_stack: List[str] = ["<module>"]
        self._scope_kind_stack: List[str] = ["module"]
        self._source_lines = source_lines

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        parent_kind = self._scope_kind_stack[-1]
        if parent_kind in ("module", "class"):
            try:
                pos = self.get_metadata(PositionProvider, node)
                func_source = "".join(
                    self._source_lines[pos.start.line - 1 : pos.end.line]
                )
                body_pos = self.get_metadata(PositionProvider, node.body)
                body_source = "".join(
                    self._source_lines[body_pos.start.line - 1 : body_pos.end.line]
                )
            except KeyError:  # pragma: no cover
                func_source = ""
                body_source = ""
            body_stmt_count = len(node.body.body)
            params = [p.name.value for p in node.params.params]
            self.functions.append(
                _FunctionInfo(
                    name=node.name.value,
                    source=func_source,
                    scope=self._scope_stack[-1],
                    body_source=body_source,
                    body_stmt_count=body_stmt_count,
                    params=params,
                )
            )
        self._scope_stack.append(node.name.value)
        self._scope_kind_stack.append("function")
        return None

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._scope_stack.pop()
        self._scope_kind_stack.pop()

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self._scope_stack.append(node.name.value)
        self._scope_kind_stack.append("class")
        return None

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self._scope_stack.pop()
        self._scope_kind_stack.pop()


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
