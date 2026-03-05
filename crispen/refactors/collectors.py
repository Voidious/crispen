from __future__ import annotations
import ast
from typing import List, Optional, Sequence, Tuple
import libcst as cst
from libcst.metadata import PositionProvider
from .ast_utils import _has_def
from .blocks import _MAX_SEQ_LEN, _MIN_WEIGHT
from .models import _FunctionInfo, _SeqInfo
from .normalizer import _normalize_source
from .weights import _sequence_weight


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


def _collect_attribute_names(source: str) -> set:
    """Return all attribute names (dot-access names) anywhere in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def _collect_called_attr_names(source: str) -> set:
    """Return attribute names used as method calls in *source*.

    Unlike :func:`_collect_attribute_names`, this only returns names that
    appear as the attribute of a call expression (i.e. ``obj.method(...)``).
    Plain attribute reads and type annotations like ``ast.AST`` are ignored,
    so the new-method-call check does not produce false positives for
    standard-library type references.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
