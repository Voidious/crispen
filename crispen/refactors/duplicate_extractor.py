"""Refactor: extract duplicate code blocks into helper functions using an LLM."""

from __future__ import annotations

import ast
import os
import re
import sys
import textwrap
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import anthropic
import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from ..errors import CrispenAPIError
from .base import Refactor

_MODEL = "claude-sonnet-4-6"
_MIN_WEIGHT = 3
_MAX_SEQ_LEN = 8
_API_HARD_TIMEOUT = 90  # seconds — hard wall-clock limit per LLM call


# ---------------------------------------------------------------------------
# Hard-timeout helper
# ---------------------------------------------------------------------------


class _ApiTimeout(Exception):
    """Raised when an LLM API call exceeds the hard per-call timeout."""


def _run_with_timeout(func, timeout, *args, **kwargs):
    """Run *func* in a daemon thread; raise _ApiTimeout if it doesn't finish.

    This enforces a hard wall-clock limit that is not affected by OS-level
    blocking (e.g. DNS resolution) which application-layer timeouts cannot
    interrupt.
    """
    result: list = [None]
    exc: list = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except BaseException as e:
            exc[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise _ApiTimeout(f"API call exceeded {timeout}s hard limit")
    if exc[0] is not None:
        raise exc[0]
    return result[0]


# ---------------------------------------------------------------------------
# Recursive statement weight
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sequence info
# ---------------------------------------------------------------------------


@dataclass
class _SeqInfo:
    stmts: List[cst.BaseStatement]
    start_line: int
    end_line: int
    scope: str
    source: str
    fingerprint: str


@dataclass
class _FunctionInfo:
    name: str
    source: str  # raw source of complete function definition
    scope: str  # "<module>" or enclosing class name
    body_source: str  # raw source of the function body (indented)
    body_stmt_count: int  # number of top-level statements in the body
    params: List[str]  # positional parameter names (empty → no-arg function)


# ---------------------------------------------------------------------------
# Sequence collector
# ---------------------------------------------------------------------------


class _SequenceCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self, source_lines: List[str], max_seq_len: int = _MAX_SEQ_LEN
    ) -> None:
        self.sequences: List[_SeqInfo] = []
        self._scope_stack: List[str] = ["<module>"]
        self._source_lines = source_lines
        self._max_seq_len = max_seq_len

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        self._scope_stack.append(node.name.value)
        return None

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._scope_stack.pop()

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self._scope_stack.append(node.name.value)
        return None

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self._scope_stack.pop()

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
        for start_i in range(n):
            for end_i in range(
                start_i + 1, min(start_i + self._max_seq_len + 1, n + 1)
            ):
                window: List[cst.BaseStatement] = [
                    s[0] for s in stmt_info[start_i:end_i]
                ]
                if _has_def(window):
                    continue
                if _sequence_weight(window) < _MIN_WEIGHT:
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
                    )
                )

    def visit_Module(self, node: cst.Module) -> Optional[bool]:
        self._process_body(node.body)
        return None

    def visit_IndentedBlock(self, node: cst.IndentedBlock) -> Optional[bool]:
        self._process_body(node.body)
        return None


# ---------------------------------------------------------------------------
# Function collector
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Function body fingerprint helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Duplicate group finding
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LLM integration
# ---------------------------------------------------------------------------

_VETO_TOOL: dict = {
    "name": "evaluate_duplicate",
    "description": (
        "Evaluate whether code blocks are semantic duplicates worth extracting"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "is_valid_duplicate": {
                "type": "boolean",
                "description": (
                    "True if extracting a shared helper would improve clarity"
                ),
            },
            "reason": {"type": "string"},
        },
        "required": ["is_valid_duplicate", "reason"],
    },
}

_EXTRACT_TOOL: dict = {
    "name": "extract_helper",
    "description": "Extract duplicate code blocks into a helper function",
    "input_schema": {
        "type": "object",
        "properties": {
            "function_name": {"type": "string"},
            "placement": {
                "type": "string",
                "description": (
                    "Where to place the helper: 'module_level' or "
                    "'staticmethod:ClassName'"
                ),
            },
            "helper_source": {
                "type": "string",
                "description": "Complete source of the helper function",
            },
            "call_site_replacements": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Replacement source for each duplicate block, "
                    "in the same order as the input blocks"
                ),
            },
        },
        "required": [
            "function_name",
            "placement",
            "helper_source",
            "call_site_replacements",
        ],
    },
}

_CALL_GEN_TOOL: dict = {
    "name": "generate_call",
    "description": "Generate a call to an existing function that replaces a code block",
    "input_schema": {
        "type": "object",
        "properties": {
            "replacement": {
                "type": "string",
                "description": (
                    "Complete replacement source "
                    "(including indentation and trailing newline)"
                ),
            }
        },
        "required": ["replacement"],
    },
}


def _llm_veto(client: anthropic.Anthropic, group: List[_SeqInfo]) -> Tuple[bool, str]:
    blocks_text = "\n\n".join(
        f"Block {i + 1} (scope: {s.scope}, lines {s.start_line}-{s.end_line}):\n"
        f"```python\n{s.source.rstrip()}\n```"
        for i, s in enumerate(group)
    )
    prompt = (
        f"Here are {len(group)} structurally similar code blocks from the same "
        f"Python file:\n\n{blocks_text}\n\n"
        "Do these blocks represent the same semantic operation such that extracting "
        "a shared helper function would improve clarity? Or are they coincidentally "
        "similar but conceptually distinct?"
    )
    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=256,
            tools=[_VETO_TOOL],
            tool_choice={"type": "tool", "name": "evaluate_duplicate"},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as exc:
        raise CrispenAPIError(
            f"DuplicateExtractor: Anthropic API error: {exc}\n"
            "Commit blocked. To skip all hooks: git commit --no-verify"
        ) from exc
    for block in response.content:
        if block.type == "tool_use" and block.name == "evaluate_duplicate":
            inp = block.input
            return inp["is_valid_duplicate"], inp.get("reason", "")
    return False, "no tool response"  # pragma: no cover


def _llm_extract(
    client: anthropic.Anthropic,
    group: List[_SeqInfo],
    full_source: str,
    escaping_vars: frozenset = frozenset(),
) -> Optional[dict]:
    blocks_text = "\n\n".join(
        f"Block {i + 1} (scope: {s.scope}, lines {s.start_line}-{s.end_line}):\n"
        f"```python\n{s.source.rstrip()}\n```"
        for i, s in enumerate(group)
    )
    snippet = full_source[:4000] if len(full_source) > 4000 else full_source
    escaping_note = ""
    if escaping_vars:
        vars_str = ", ".join(sorted(escaping_vars))
        escaping_note = (
            f"\n\nThe following variables are assigned within the duplicate block "
            f"and referenced by code that immediately follows the block at one or "
            f"more call sites: {vars_str}. The helper function must return these "
            f"variables. At call sites where the return value is needed, capture it; "
            f"at call sites where it is not needed, discard the return value."
        )
    prompt = (
        "Extract the following duplicate code blocks from this Python file into a "
        f"helper function.\n\nFile source:\n```python\n{snippet}\n```\n\n"
        f"Duplicate blocks:\n{blocks_text}\n\n"
        "Place the helper immediately before the enclosing function of its first use. "
        "If both call sites are inside the same class, use a @staticmethod. "
        "Return complete, valid Python for the helper and each call site replacement."
        f"{escaping_note}"
    )
    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            tools=[_EXTRACT_TOOL],
            tool_choice={"type": "tool", "name": "extract_helper"},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as exc:
        raise CrispenAPIError(
            f"DuplicateExtractor: Anthropic API error: {exc}\n"
            "Commit blocked. To skip all hooks: git commit --no-verify"
        ) from exc
    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_helper":
            return block.input
    return None  # pragma: no cover


def _llm_veto_func_match(
    client: anthropic.Anthropic,
    seq: _SeqInfo,
    func: _FunctionInfo,
    full_source: str,
) -> Tuple[bool, str]:
    """Ask the LLM whether *seq* performs the same operation as *func*'s body."""
    snippet = full_source[:4000] if len(full_source) > 4000 else full_source
    prompt = (
        "A code block in a Python file may be replaceable by a call to an existing "
        "function.\n\n"
        f"Code block (scope: {seq.scope}, lines {seq.start_line}-{seq.end_line}):\n"
        f"```python\n{seq.source.rstrip()}\n```\n\n"
        f"Existing function '{func.name}':\n"
        f"```python\n{func.source.rstrip()}\n```\n\n"
        f"File source:\n```python\n{snippet}\n```\n\n"
        "Does this code block perform the same semantic operation as the function "
        "body, such that it could be replaced by a call to the function? "
        "Use the evaluate_duplicate tool to answer."
    )
    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=256,
            tools=[_VETO_TOOL],
            tool_choice={"type": "tool", "name": "evaluate_duplicate"},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as exc:
        raise CrispenAPIError(
            f"DuplicateExtractor: Anthropic API error: {exc}\n"
            "Commit blocked. To skip all hooks: git commit --no-verify"
        ) from exc
    for block in response.content:
        if block.type == "tool_use" and block.name == "evaluate_duplicate":
            inp = block.input
            return inp["is_valid_duplicate"], inp.get("reason", "")
    return False, "no tool response"  # pragma: no cover


def _generate_no_arg_call(seq: _SeqInfo, func: _FunctionInfo) -> str:
    """Algorithmically generate a no-argument call to *func*, preserving indentation."""
    first_line = seq.source.splitlines()[0]
    indent = first_line[: len(first_line) - len(first_line.lstrip())]
    return indent + func.name + "()\n"


def _llm_generate_call(
    client: anthropic.Anthropic,
    seq: _SeqInfo,
    func: _FunctionInfo,
    full_source: str,
) -> Optional[str]:
    """Ask the LLM to generate a call expression replacing *seq* with *func*."""
    snippet = full_source[:4000] if len(full_source) > 4000 else full_source
    prompt = (
        f"Replace this code block with a call to the existing function"
        f" '{func.name}'.\n\n"
        f"Code block (scope: {seq.scope}, lines {seq.start_line}-{seq.end_line}):\n"
        f"```python\n{seq.source.rstrip()}\n```\n\n"
        f"Function '{func.name}':\n"
        f"```python\n{func.source.rstrip()}\n```\n\n"
        f"File source:\n```python\n{snippet}\n```\n\n"
        "Generate a replacement that preserves the original indentation and ends "
        "with a newline. Pass the replacement to the generate_call tool."
    )
    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=256,
            tools=[_CALL_GEN_TOOL],
            tool_choice={"type": "tool", "name": "generate_call"},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as exc:
        raise CrispenAPIError(
            f"DuplicateExtractor: Anthropic API error: {exc}\n"
            "Commit blocked. To skip all hooks: git commit --no-verify"
        ) from exc
    for block in response.content:
        if block.type == "tool_use" and block.name == "generate_call":
            return block.input["replacement"]
    return None  # pragma: no cover


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _verify_extraction(
    helper_source: Optional[str], call_replacements: List[str]
) -> bool:
    """Verify the extraction produces syntactically valid Python.

    Replacements are dedented before checking since they may be
    indented for their original context inside a function body.
    Pass helper_source=None to skip the helper compilation check (used when
    replacing with an existing function rather than a newly extracted one).
    """
    if helper_source is not None:
        try:
            compile(textwrap.dedent(helper_source), "<helper>", "exec")
        except SyntaxError:
            return False
    for replacement in call_replacements:
        try:
            compile(textwrap.dedent(replacement), "<replacement>", "exec")
        except SyntaxError:
            return False
    return True


def _names_assigned_in(block_source: str) -> set:
    """Return names assigned at the top level of block_source.

    Covers bare ``x = ...`` (ast.Assign) and augmented ``x += ...``
    (ast.AugAssign) statements only; other assignment forms are ignored.
    """
    try:
        tree = ast.parse(textwrap.dedent(block_source))
    except SyntaxError:
        return set()
    names: set = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for n in ast.walk(target):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(node, ast.AugAssign):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    names.add(n.id)
    return names


def _find_escaping_vars(group: List[_SeqInfo], source_lines: List[str]) -> set:
    """Return names assigned in any group sequence that are referenced after it.

    A variable "escapes" when the block assigns it and subsequent code in the
    same scope (at the same or deeper indentation level) references it.
    The helper must return these variables so callers that need them can
    capture the return value.
    """
    escaping: set = set()
    for seq in group:
        block_src = "".join(source_lines[seq.start_line - 1 : seq.end_line])
        assigned = _names_assigned_in(block_src)
        if not assigned:
            continue

        # Infer the block's indentation level from its first non-empty line.
        first_line = next(
            (
                ln
                for ln in source_lines[seq.start_line - 1 : seq.end_line]
                if ln.strip()
            ),
            "",
        )
        block_indent = len(first_line) - len(first_line.lstrip())

        # Collect lines that follow the block within the same scope.
        # For indented blocks: stop when indentation falls below block_indent.
        # For module-level (indent 0): stop at the next def/class statement.
        after_lines: List[str] = []
        for line in source_lines[seq.end_line :]:
            if not line.strip():
                after_lines.append(line)
                continue
            line_indent = len(line) - len(line.lstrip())
            if block_indent == 0:
                if re.match(r"def |class ", line):
                    break
            elif line_indent < block_indent:
                break
            after_lines.append(line)

        if not after_lines:
            continue

        after_src = "".join(after_lines)
        try:
            after_tree = ast.parse(textwrap.dedent(after_src))
        except SyntaxError:
            continue

        used_after = {n.id for n in ast.walk(after_tree) if isinstance(n, ast.Name)}
        escaping |= assigned & used_after

    return escaping


# ---------------------------------------------------------------------------
# Text editing
# ---------------------------------------------------------------------------


def _ensure_final_newline(text: str) -> list[str]:
    """Split *text* into lines and guarantee the last line ends with a newline."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def _apply_edits(source: str, edits: List[Tuple[int, int, str]]) -> str:
    """Apply (start_0, end_0, text) edits bottom-to-top.

    Indices are 0-based; lines[start_0:end_0] is replaced with text.
    An insertion before line N uses start_0 == end_0 == N.
    Overlapping replacement ranges are skipped.
    """
    lines = _ensure_final_newline(source)

    applied: List[Tuple[int, int]] = []
    for start, end, text in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        is_insertion = start == end
        if not is_insertion:
            if any(a_start < end and a_end > start for a_start, a_end in applied):
                continue
            applied.append((start, end))
        new_lines = _ensure_final_newline(text)
        lines[start:end] = new_lines

    return "".join(lines)


def _find_insertion_point(source: str, scope: str) -> int:
    """Return 0-based line index to insert before.

    For module scope, inserts after the last import.
    For a named scope, inserts before the def/class line.
    """
    source_lines = source.splitlines()
    if scope == "<module>":
        last_import = -1
        for i, line in enumerate(source_lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                last_import = i
        return last_import + 1

    pattern = re.compile(rf"^\s*(?:def|class)\s+{re.escape(scope)}\s*[\(:]")
    for i, line in enumerate(source_lines):
        if pattern.match(line):
            return i
    return 0


# ---------------------------------------------------------------------------
# Main refactor
# ---------------------------------------------------------------------------


class DuplicateExtractor(Refactor):
    """Detect and extract duplicate code blocks into helper functions via LLM."""

    def __init__(
        self,
        changed_ranges: List[Tuple[int, int]],
        source: str = "",
        verbose: bool = True,
    ) -> None:
        super().__init__(changed_ranges, source=source, verbose=verbose)
        self._new_source: Optional[str] = None
        if source:
            self._analyze(source)

    def _analyze(self, source: str) -> None:
        # 1. Parse tree; early-return on syntax error.
        try:
            tree = cst.parse_module(source)
        except cst.ParserSyntaxError:
            return

        # 2. Source lines.
        source_lines = source.splitlines(keepends=True)

        # 3. Collect functions.
        func_collector = _FunctionCollector(source_lines)
        MetadataWrapper(tree).visit(func_collector)
        all_functions = func_collector.functions

        # 4-5. Build function body fingerprint map (only for called functions).
        called_names = _collect_called_names(source)
        func_body_fps = _build_function_body_fps(all_functions, called_names)

        # 6. Compute max sequence length to capture full function bodies.
        max_seq_len = max(
            max(f.body_stmt_count for f in all_functions) if all_functions else 0,
            _MAX_SEQ_LEN,
        )

        # 7. Collect sequences.
        collector = _SequenceCollector(source_lines, max_seq_len=max_seq_len)
        MetadataWrapper(tree).visit(collector)

        # 8. Preliminary duplicate groups.
        groups = _find_duplicate_groups(collector.sequences, self.changed_ranges)

        # 9. Check whether any sequence can be replaced with an existing function.
        has_func_matches = func_body_fps and any(
            _overlaps_diff(seq, self.changed_ranges)
            and seq.fingerprint in func_body_fps
            and func_body_fps[seq.fingerprint].name != seq.scope
            for seq in collector.sequences
        )

        # 10. Early exit — nothing to do.
        if not has_func_matches and not groups:
            return

        # 12. Create API client.
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise CrispenAPIError(
                "DuplicateExtractor: ANTHROPIC_API_KEY is not set.\n"
                "Commit blocked. To skip all hooks: git commit --no-verify"
            )

        client = anthropic.Anthropic(api_key=api_key, timeout=60.0)
        edits: List[Tuple[int, int, str]] = []
        pending_changes: List[str] = []
        matched_line_ranges: set = set()

        # 14. Function body match pass.
        if func_body_fps:
            for seq in collector.sequences:
                if not _overlaps_diff(seq, self.changed_ranges):
                    continue
                if seq.fingerprint not in func_body_fps:
                    continue
                func = func_body_fps[seq.fingerprint]
                if func.name == seq.scope:
                    continue
                if self.verbose:
                    print(
                        f"crispen: DuplicateExtractor: func-match check — "
                        f"scope '{seq.scope}': lines {seq.start_line}-{seq.end_line}"
                        f" → '{func.name}'",
                        file=sys.stderr,
                        flush=True,
                    )
                try:
                    is_valid, reason = _run_with_timeout(
                        _llm_veto_func_match,
                        _API_HARD_TIMEOUT,
                        client,
                        seq,
                        func,
                        source,
                    )
                except _ApiTimeout:
                    print(
                        "crispen: DuplicateExtractor:   → func-match veto timed out",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                if self.verbose:
                    status = "ACCEPTED" if is_valid else "VETOED"
                    print(
                        f"crispen: DuplicateExtractor:   → {status}: {reason}",
                        file=sys.stderr,
                        flush=True,
                    )
                if not is_valid:
                    continue
                if func.scope == "<module>" and not func.params:
                    replacement = _generate_no_arg_call(seq, func)
                else:
                    try:
                        replacement = _run_with_timeout(
                            _llm_generate_call,
                            _API_HARD_TIMEOUT,
                            client,
                            seq,
                            func,
                            source,
                        )
                    except _ApiTimeout:
                        print(
                            "crispen: DuplicateExtractor:"
                            "   → call generation timed out",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    if replacement is None:
                        continue  # pragma: no cover
                if not _verify_extraction(None, [replacement]):
                    continue
                if self.verbose:
                    print(
                        f"crispen: DuplicateExtractor:   → replacing '{seq.scope}'"
                        f" with '{func.name}()'",
                        file=sys.stderr,
                        flush=True,
                    )
                edits.append((seq.start_line - 1, seq.end_line, replacement))
                matched_line_ranges.add((seq.start_line, seq.end_line))
                pending_changes.append(
                    f"DuplicateExtractor: replaced '{seq.scope}' body"
                    f" with call to '{func.name}'"
                )

        # 15. Recompute duplicate groups excluding matched sequences.
        if matched_line_ranges:
            remaining = [
                s
                for s in collector.sequences
                if not any(
                    s.start_line <= r_end and s.end_line >= r_start
                    for r_start, r_end in matched_line_ranges
                )
            ]
            groups = _find_duplicate_groups(remaining, self.changed_ranges)

        # 16. Log group count.
        if groups and self.verbose:
            print(
                f"crispen: DuplicateExtractor: found {len(groups)} duplicate group(s)",
                file=sys.stderr,
                flush=True,
            )

        # 17. Duplicate group extraction pass.
        for group in groups:
            # Compute escaping vars algorithmically before any LLM call so the
            # extraction prompt can instruct the LLM to return them.
            escaping_vars = frozenset(_find_escaping_vars(group, source_lines))
            if self.verbose:
                ranges_str = ", ".join(
                    f"lines {s.start_line}-{s.end_line}" for s in group
                )
                print(
                    f"crispen: DuplicateExtractor: veto check — "
                    f"scope '{group[0].scope}': {ranges_str}",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                is_valid, reason = _run_with_timeout(
                    _llm_veto, _API_HARD_TIMEOUT, client, group
                )
            except _ApiTimeout:
                print(
                    "crispen: DuplicateExtractor: API call timed out, skipping group",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if self.verbose:
                status = "ACCEPTED" if is_valid else "VETOED"
                print(
                    f"crispen: DuplicateExtractor:   → {status}: {reason}",
                    file=sys.stderr,
                    flush=True,
                )
            if not is_valid:
                continue

            try:
                extraction = _run_with_timeout(
                    _llm_extract,
                    _API_HARD_TIMEOUT,
                    client,
                    group,
                    source,
                    escaping_vars,
                )
            except _ApiTimeout:
                print(
                    "crispen: DuplicateExtractor: API call timed out, skipping group",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if extraction is None:
                continue  # pragma: no cover

            helper_source = extraction["helper_source"]
            call_replacements = extraction["call_site_replacements"]
            placement = extraction.get("placement", "module_level")

            if len(call_replacements) != len(group):
                continue

            if not _verify_extraction(helper_source, call_replacements):
                continue

            for seq, replacement in zip(group, call_replacements):
                edits.append((seq.start_line - 1, seq.end_line, replacement))

            first_seq = min(group, key=lambda s: s.start_line)
            if placement.startswith("staticmethod:"):
                # Insert inside the class body, one line after "class Foo:".
                scope = placement.split(":", 1)[1]
                insert_pos = _find_insertion_point(source, scope) + 1
            else:
                scope = first_seq.scope
                insert_pos = _find_insertion_point(source, scope)
            edits.append((insert_pos, insert_pos, helper_source + "\n\n"))

            func_name = extraction["function_name"]
            if self.verbose:
                print(
                    f"crispen: DuplicateExtractor: extracting '{func_name}'",
                    file=sys.stderr,
                    flush=True,
                )
            pending_changes.append(
                f"DuplicateExtractor: extracted '{func_name}' "
                f"from {len(group)} duplicate blocks"
            )

        # 18. Apply edits, compile, write.
        if edits:
            candidate = _apply_edits(source, edits)
            try:
                compile(candidate, "<rewritten>", "exec")
            except SyntaxError as exc:
                if self.verbose:
                    print(
                        f"crispen: DuplicateExtractor: skipping — "
                        f"assembled output not valid Python: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            else:
                self._new_source = candidate
                self.changes_made.extend(pending_changes)

    def get_rewritten_source(self) -> Optional[str]:
        return self._new_source
