from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from libcst.metadata import PositionProvider
import ast
import re
import textwrap
import libcst as cst


_MODEL = "claude-sonnet-4-6"
_MIN_WEIGHT = 3
_MAX_SEQ_LEN = 8


def _strip_helper_docstring(helper_source: str) -> str:
    """Remove the docstring from helper_source if the first function has one."""
    try:
        tree = cst.parse_module(textwrap.dedent(helper_source))
    except cst.ParserSyntaxError:
        return helper_source

    if not tree.body or not isinstance(tree.body[0], cst.FunctionDef):
        return helper_source

    func = tree.body[0]
    body = func.body
    if not isinstance(body, cst.IndentedBlock) or not body.body:  # pragma: no cover
        return helper_source

    first = body.body[0]
    if not (
        isinstance(first, cst.SimpleStatementLine)
        and len(first.body) == 1
        and isinstance(first.body[0], cst.Expr)
        and isinstance(first.body[0].value, (cst.SimpleString, cst.ConcatenatedString))
    ):
        return helper_source

    rest = list(body.body[1:])
    if not rest:
        return helper_source

    new_func = func.with_changes(body=body.with_changes(body=rest))
    return tree.with_changes(body=[new_func] + list(tree.body[1:])).code


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


def _build_helper_insertion(
    source_lines: List[str],
    insert_pos: int,
    helper_source: str,
    placement: str,
) -> Tuple[int, int, str]:
    """Build an edit tuple that inserts helper_source with correct surrounding blanks.

    Absorbs existing blank lines around the insertion point so the result has
    exactly 2 blank lines before and after module-level helpers, or 1 blank
    line for staticmethod insertions inside a class body.
    """
    blank_lines = 1 if placement.startswith("staticmethod:") else 2

    # Count consecutive blank lines immediately before insert_pos.
    before_blanks = 0
    i = insert_pos - 1
    while i >= 0 and not source_lines[i].strip():
        before_blanks += 1
        i -= 1

    # Count consecutive blank lines at and immediately after insert_pos.
    after_blanks = 0
    i = insert_pos
    while i < len(source_lines) and not source_lines[i].strip():
        after_blanks += 1
        i += 1

    # Replace surrounding blank lines so we don't double-count them.
    start = insert_pos - before_blanks
    end = insert_pos + after_blanks
    clean = helper_source.strip("\n") + "\n"
    text = "\n" * blank_lines + clean + "\n" * blank_lines
    return (start, end, text)


def _apply_edits(source: str, edits: List[Tuple[int, int, str]]) -> str:
    """Apply (start_0, end_0, text) edits bottom-to-top.

    Indices are 0-based; lines[start_0:end_0] is replaced with text.
    An insertion before line N uses start_0 == end_0 == N.
    Overlapping replacement ranges are skipped.
    """
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    applied: List[Tuple[int, int]] = []
    for start, end, text in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        is_insertion = start == end
        if not is_insertion:
            if any(a_start < end and a_end > start for a_start, a_end in applied):
                continue
            applied.append((start, end))
        new_lines = text.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        lines[start:end] = new_lines

    return "".join(lines)


def _find_insertion_point(source: str, scope: str) -> int:
    """Return 0-based line index to insert before.

    For module scope, inserts after the last import.
    For a named scope, inserts before the def/class line.

    If the named scope resolves to an indented ``def`` (i.e. a class method),
    inserting a module-level helper immediately before it would end the class
    definition prematurely — the remaining class methods would be silently
    re-parsed as nested functions of the helper, producing valid-syntax but
    broken code that ``compile()`` does not catch.  In that case we walk
    backwards to the enclosing class definition and insert before it instead.
    """
    source_lines = source.splitlines()
    if scope == "<module>":
        last_import = -1
        for i, line in enumerate(source_lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                last_import = i
        return last_import + 1

    pattern = re.compile(rf"^\s*(?:async\s+def|def|class)\s+{re.escape(scope)}\s*[\(:]")
    for i, line in enumerate(source_lines):
        if pattern.match(line):
            method_indent = len(line) - len(line.lstrip())
            if method_indent > 0:
                # The def is inside a class body.  Walk backwards to find the
                # enclosing class definition and insert before that instead.
                # If the first lower-indent non-blank line is NOT a class
                # definition (i.e. the def is a nested function inside a
                # regular function), stop immediately so we don't mis-identify
                # an unrelated class above the outer function as the enclosing
                # class.
                for j in range(i - 1, -1, -1):
                    prev = source_lines[j]
                    if not prev.strip():
                        continue
                    prev_indent = len(prev) - len(prev.lstrip())
                    if prev_indent < method_indent:
                        if re.match(r"\s*class\s+\w+", prev):
                            return j
                        break  # nested function — fall through to decorator walk
            # Walk backwards over any preceding decorator lines (including
            # multi-line decorator arguments) so the helper is inserted
            # before the decorator block, not between decorators and the def.
            j = i - 1
            paren_depth = 0
            while j >= 0:
                stripped = source_lines[j].strip()
                if not stripped:
                    break
                for ch in stripped:
                    if ch == ")":
                        paren_depth += 1
                    elif ch == "(":
                        paren_depth -= 1
                if paren_depth == 0 and not stripped.startswith("@"):
                    break
                j -= 1
            return j + 1
    return 0
