"""Refactor: extract duplicate code blocks into helper functions using an LLM."""

from __future__ import annotations

import ast
import re
import sys
import textwrap
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from .. import llm_client as _llm_client
from .base import Refactor

_MODEL = "claude-sonnet-4-6"
_MIN_WEIGHT = 3
_MAX_SEQ_LEN = 8
_API_HARD_TIMEOUT = 90  # seconds — hard wall-clock limit per LLM call


# ---------------------------------------------------------------------------
# Docstring stripping
# ---------------------------------------------------------------------------


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
    class_scope: Optional[str] = None  # enclosing class name, or None if module-level


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


def _parse_source_or_empty_set(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, set()
    return tree, set()


def _collect_called_names(source: str) -> set:
    """Return a set of all names called (as functions) in *source*.

    Uses ast.parse + ast.walk to find all ast.Call nodes.  Returns the
    called name: func.id for ast.Name callees, func.attr for ast.Attribute
    callees.  On SyntaxError, returns an empty set.
    """
    tree, names = _parse_source_or_empty_set(source)
    if tree == None:
        return set()
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
            "extraction_notes": {
                "type": "string",
                "description": (
                    "If accepting, note any potential pitfalls the extraction "
                    "step should watch out for — e.g., tricky variable scoping, "
                    "mutable arguments, subtle differences in variable names, or "
                    "return-value handling. Leave empty if none."
                ),
            },
        },
        "required": ["is_valid_duplicate", "reason"],
    },
}

_VERIFY_TOOL: dict = {
    "name": "verify_extraction",
    "description": "Verify that an extracted helper function is semantically correct",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_correct": {
                "type": "boolean",
                "description": (
                    "True if the extraction is semantically equivalent to the originals"
                ),
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Specific issues found. Empty if correct. Each issue should "
                    "describe what is wrong and how the extraction should be fixed."
                ),
            },
        },
        "required": ["is_correct", "issues"],
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
                    "in the same order as the input blocks. "
                    "Each replacement must preserve the original block's "
                    "leading indentation and end with a trailing newline. "
                    "Cover only the exact lines of the specified block — "
                    "do not include any code from before or after the block."
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


def _handle_llm_veto_result(result):
    if result is not None:
        return (
            result["is_valid_duplicate"],
            result.get("reason", ""),
            result.get("extraction_notes", ""),
        )
    return False, "no tool response", ""  # pragma: no cover


def _llm_veto(
    client,
    group: List[_SeqInfo],
    model: str = _MODEL,
    provider: str = "anthropic",
) -> Tuple[bool, str, str]:
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
        "similar but conceptually distinct?\n\n"
        "If you accept (is_valid_duplicate=True), also fill in extraction_notes "
        "with any potential pitfalls the extraction step should watch out for — "
        "e.g., tricky variable scoping, mutable arguments, subtle differences in "
        "variable names between blocks, or return-value handling edge cases."
    )
    result = _llm_client.call_with_tool(
        client,
        provider,
        model,
        384,
        _VETO_TOOL,
        "evaluate_duplicate",
        [{"role": "user", "content": prompt}],
        caller="DuplicateExtractor",
    )
    return _handle_llm_veto_result(result)


def _llm_extract(
    client,
    group: List[_SeqInfo],
    full_source: str,
    escaping_vars: frozenset = frozenset(),
    used_names: frozenset = frozenset(),
    model: str = _MODEL,
    helper_docstrings: bool = True,
    provider: str = "anthropic",
    veto_notes: str = "",
    prev_failures: List[str] = [],
    prev_output: Optional[dict] = None,
) -> Optional[dict]:
    src_lines = full_source.splitlines(keepends=True)
    block_entries = []
    for i, s in enumerate(group):
        entry = (
            f"Block {i + 1} (scope: {s.scope}, lines {s.start_line}-{s.end_line}):\n"
            f"```python\n{s.source.rstrip()}\n```"
        )
        next_idx = s.end_line  # 0-based index of the first line after the block
        if next_idx < len(src_lines):
            next_line = src_lines[next_idx].rstrip()
            if next_line.strip():
                entry += (
                    f"\nLine immediately after this block"
                    f" (must NOT appear in the replacement): `{next_line}`"
                )
        block_entries.append(entry)
    blocks_text = "\n\n".join(block_entries)
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
    used_names_note = ""
    if used_names:
        names_str = ", ".join(sorted(used_names))
        used_names_note = (
            f"\n\nThe following function names are already defined in this file "
            f"or reserved by a previous extraction: {names_str}. "
            f"Do not use any of these names for the helper function."
        )
    docstring_note = (
        ""
        if helper_docstrings
        else "\n\nDo not include a docstring in the helper function."
    )
    veto_notes_note = ""
    if veto_notes:
        veto_notes_note = (
            f"\n\nNotes from code review (watch out for these pitfalls): "
            f"{veto_notes[:500]}"
        )
    failures_note = ""
    if prev_failures:
        failures_str = "\n".join(f"- {f}" for f in prev_failures)
        if prev_output is not None:
            prior_helper = prev_output.get("helper_source", "")
            prior_repls = prev_output.get("call_site_replacements", [])
            repls_text = "\n".join(
                f"  [{i + 1}] {r!r}" for i, r in enumerate(prior_repls)
            )
            failures_note = (
                f"\n\nThe previous extraction attempt produced:\n\n"
                f"helper_source:\n```python\n{prior_helper}```\n\n"
                f"call_site_replacements:\n{repls_text}\n\n"
                f"But failed verification with these issues:\n{failures_str}\n\n"
                f"Please correct these issues in your new attempt."
            )
        else:
            failures_note = (
                f"\n\nThe previous extraction attempt failed. Please correct these "
                f"issues:\n{failures_str}"
            )
    class_scopes = {s.class_scope for s in group}
    all_same_class = len(class_scopes) == 1 and None not in class_scopes
    if all_same_class:
        staticmethod_instruction = (
            "If all call sites are inside the same class, use a @staticmethod. "
        )
    else:
        staticmethod_instruction = (
            "Use module_level placement — call sites span different classes or scopes. "
        )
    prompt = (
        "Extract the following duplicate code blocks from this Python file into a "
        f"helper function.\n\nFile source:\n```python\n{snippet}\n```\n\n"
        f"Duplicate blocks:\n{blocks_text}\n\n"
        "Place the helper immediately before the enclosing function of its first use. "
        f"{staticmethod_instruction}"
        "Return complete, valid Python for the helper and each call site replacement. "
        "Each call site replacement must start with the same leading indentation as "
        "the block it replaces, end with a trailing newline, and cover only the exact "
        "lines of the duplicate block — stopping before the 'Line immediately after "
        "this block' marker shown above. Do not include any code from before or after "
        "the block. "
        "Double-check that only required parameters are passed to the helper — do not "
        "include an unused parameter, or one that is overwritten before being read. "
        "Be mindful of the code being removed from the call site: if variable "
        "assignments are moved into the helper, those variables may no longer be "
        "defined in the calling scope at that point. "
        "If the helper uses a sentinel return value to signal an error path (such as "
        "returning an empty collection), check for it at the call site with `==`, not "
        "`is` — `is` only gives correct results for singletons like `None`, `True`, "
        "and `False`, not for constructed objects like `set()`."
        f"{escaping_note}"
        f"{used_names_note}"
        f"{docstring_note}"
        f"{veto_notes_note}"
        f"{failures_note}"
    )
    return _llm_client.call_with_tool(
        client,
        provider,
        model,
        1024,
        _EXTRACT_TOOL,
        "extract_helper",
        [{"role": "user", "content": prompt}],
        caller="DuplicateExtractor",
    )


def _llm_veto_func_match(
    client,
    seq: _SeqInfo,
    func: _FunctionInfo,
    full_source: str,
    model: str = _MODEL,
    provider: str = "anthropic",
) -> Tuple[bool, str, str]:
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
    result = _llm_client.call_with_tool(
        client,
        provider,
        model,
        256,
        _VETO_TOOL,
        "evaluate_duplicate",
        [{"role": "user", "content": prompt}],
        caller="DuplicateExtractor",
    )
    return _handle_llm_veto_result(result)


def _generate_no_arg_call(seq: _SeqInfo, func: _FunctionInfo) -> str:
    """Algorithmically generate a no-argument call to *func*, preserving indentation."""
    first_line = seq.source.splitlines()[0]
    indent = first_line[: len(first_line) - len(first_line.lstrip())]
    return indent + func.name + "()\n"


def _llm_generate_call(
    client,
    seq: _SeqInfo,
    func: _FunctionInfo,
    full_source: str,
    model: str = _MODEL,
    provider: str = "anthropic",
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
    result = _llm_client.call_with_tool(
        client,
        provider,
        model,
        256,
        _CALL_GEN_TOOL,
        "generate_call",
        [{"role": "user", "content": prompt}],
        caller="DuplicateExtractor",
    )
    if result is not None:
        return result["replacement"]
    return None  # pragma: no cover


def _llm_verify_extraction(
    client,
    group: List[_SeqInfo],
    helper_source: str,
    call_replacements: List[str],
    full_source: str,
    model: str = _MODEL,
    provider: str = "anthropic",
) -> Tuple[bool, List[str]]:
    """Ask the LLM to verify the extraction is semantically correct.

    Returns ``(is_correct, issues)`` where *issues* is a list of specific
    problems found.  Returns ``(True, [])`` if the call times out or the LLM
    cannot respond, so a verification failure never silently blocks commits.
    """
    blocks_text = "\n\n".join(
        f"Original block {i + 1} (scope: {s.scope}, "
        f"lines {s.start_line}-{s.end_line}):\n"
        f"```python\n{s.source.rstrip()}\n```"
        for i, s in enumerate(group)
    )
    replacements_text = "\n\n".join(
        f"Replacement for block {i + 1}:\n```python\n{r.rstrip()}\n```"
        for i, r in enumerate(call_replacements)
    )
    snippet = full_source[:2000] if len(full_source) > 2000 else full_source
    prompt = (
        "Verify that the following helper function extraction is semantically "
        "correct by tracing through the code carefully.\n\n"
        f"Original duplicate blocks:\n{blocks_text}\n\n"
        f"Extracted helper:\n```python\n{helper_source.rstrip()}\n```\n\n"
        f"Call site replacements:\n{replacements_text}\n\n"
        f"File context (truncated):\n```python\n{snippet}\n```\n\n"
        "Check each of the following:\n"
        "1. Every variable read (but not locally assigned) in the original block "
        "is passed as a parameter to the helper\n"
        "2. Every variable assigned in the original block and used afterward is "
        "returned by the helper and captured at the call site\n"
        "3. No parameter is assigned before it is first read in the helper body\n"
        "4. If the original block ends with a non-None return, the call site "
        "replacement also propagates that return value\n"
        "5. The call site replacements match the original indentation and cover "
        "exactly the lines of the original block\n"
        "If correct, set is_correct=True and issues=[]. "
        "Otherwise set is_correct=False and list each specific issue."
    )
    result = _llm_client.call_with_tool(
        client,
        provider,
        model,
        512,
        _VERIFY_TOOL,
        "verify_extraction",
        [{"role": "user", "content": prompt}],
        caller="DuplicateExtractor",
    )
    if result is None:
        return True, []  # pragma: no cover
    return result["is_correct"], result.get("issues", [])


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _normalize_replacement_indentation(seq: _SeqInfo, replacement: str) -> str:
    """Re-indent *replacement* to match the original block's leading whitespace.

    The LLM sometimes returns replacements at column 0.  This function
    re-indents them to match the indentation of the corresponding original
    block, so the assembled edit remains valid Python.
    """
    orig_lines = [ln for ln in seq.source.splitlines() if ln.strip()]
    if not orig_lines:
        return replacement
    first = orig_lines[0]
    expected_indent = first[: len(first) - len(first.lstrip())]
    dedented = textwrap.dedent(replacement)
    if not expected_indent:
        return dedented
    return textwrap.indent(dedented, expected_indent)


_MUTABLE_CONSTRUCTORS = frozenset({"set", "list", "dict", "frozenset", "bytearray"})


def _has_mutable_literal_is_check(source: str) -> bool:
    """Return True if *source* contains identity checks against mutable literals.

    Patterns like ``x is set()``, ``x is []``, or ``x is {}`` are always
    False in Python because each literal creates a new object at runtime.
    Such patterns are a common LLM mistake when using a ``set()`` sentinel.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for op, comp in zip(node.ops, node.comparators):
            if not isinstance(op, (ast.Is, ast.IsNot)):
                continue
            if isinstance(comp, (ast.List, ast.Set, ast.Dict, ast.Tuple)):
                return True
            if (
                isinstance(comp, ast.Call)
                and isinstance(comp.func, ast.Name)
                and comp.func.id in _MUTABLE_CONSTRUCTORS
            ):
                return True
    return False


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


def _has_call_to(func_name: str, source: str) -> bool:
    """Return True if func_name is called anywhere in source.

    Checks both direct calls (``func_name(...)``) and attribute calls
    (``obj.func_name(...)``), covering both module-level helpers and
    staticmethod calls.  Returns False if source cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == func_name:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
            return True
    return False


def _verify_extraction(
    helper_source: Optional[str], call_replacements: List[str]
) -> bool:
    """Verify the extraction produces syntactically valid Python.

    Replacements are dedented and then wrapped in a dummy function before
    compilation so that ``return`` / ``yield`` statements — which are legal
    inside a function body — do not cause false SyntaxError rejections.
    Pass helper_source=None to skip the helper compilation check (used when
    replacing with an existing function rather than a newly extracted one).
    """
    if helper_source is not None:
        dedented_helper = textwrap.dedent(helper_source)
        try:
            compile(dedented_helper, "<helper>", "exec")
        except SyntaxError:
            return False
        if _has_param_overwritten_before_read(helper_source):
            return False
        # Dedent before checking: helper may be indented (e.g. staticmethod).
        # compile() already confirmed it's valid Python, so ast.parse will succeed.
        if _has_mutable_literal_is_check(dedented_helper):
            return False
    for replacement in call_replacements:
        dedented = textwrap.dedent(replacement)
        # Wrap in a dummy function that contains a for loop so that
        # ``return`` / ``yield`` (valid inside a function body) AND
        # ``continue`` / ``break`` (valid inside a loop body) do not cause
        # false SyntaxError rejections.  Replacements are always placed back
        # inside the caller's original context, which may include a loop.
        wrapped = "def _check():\n    for _ in []:\n" + textwrap.indent(
            dedented, "        "
        )
        try:
            compile(wrapped, "<replacement>", "exec")
        except SyntaxError:
            # Retry with async wrapper for replacements that contain `await`
            async_wrapped = "async def _check():\n    for _ in []:\n" + textwrap.indent(
                dedented, "        "
            )
            try:
                compile(async_wrapped, "<replacement>", "exec")
            except SyntaxError:
                return False
            wrapped = async_wrapped
        # Check the wrapped form so that indented/return-containing replacements
        # parse successfully and give a definitive True/False answer.
        if _has_mutable_literal_is_check(wrapped):
            return False
    return True


def _has_param_overwritten_before_read(helper_source: str) -> bool:
    """Return True if any parameter is assigned before it is first read.

    This detects a common LLM mistake where a parameter is included in the
    function signature but then immediately overwritten on the first line,
    making the parameter useless and causing UnboundLocalError at call sites
    that try to pass a value that was not yet assigned.
    """
    try:
        tree = ast.parse(textwrap.dedent(helper_source))
    except SyntaxError:  # pragma: no cover
        return False  # pragma: no cover
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {arg.arg for arg in node.args.args}
        params |= {arg.arg for arg in node.args.posonlyargs}
        params |= {arg.arg for arg in node.args.kwonlyargs}
        if node.args.vararg:
            params.add(node.args.vararg.arg)
        if node.args.kwarg:
            params.add(node.args.kwarg.arg)
        for stmt in node.body:
            for n in ast.walk(stmt):
                if isinstance(n, ast.Name) and n.id in params:
                    if isinstance(n.ctx, ast.Store):
                        return True
                    params.discard(n.id)  # first use is a read — param is legitimate
    return False


def _pyflakes_new_undefined_names(original: str, candidate: str) -> set:
    """Return undefined names (F821) introduced by the edit.

    Compares pyflakes output before and after the edit and returns only names
    that are newly undefined in the candidate — not ones already present in the
    original source. This avoids false positives from pre-existing bare function
    calls or module-level references that are valid in context but not resolvable
    from a standalone snippet.
    """
    import pyflakes.api
    import pyflakes.messages

    class _Collector:
        def __init__(self):
            self.names: set = set()

        def unexpectedError(self, filename, msg):  # pragma: no cover
            pass

        def syntaxError(self, filename, msg, lineno, offset, text):  # pragma: no cover
            pass

        def flake(self, msg):
            if isinstance(msg, pyflakes.messages.UndefinedName):
                self.names.add(msg.message_args[0])

    before = _Collector()
    pyflakes.api.check(original, "<original>", reporter=before)
    after = _Collector()
    pyflakes.api.check(candidate, "<rewritten>", reporter=after)
    return after.names - before.names


def _missing_free_vars(
    block_src: str, call_srcs: List[str], helper_src: str, source: str
) -> set:
    """Return locally-scoped free variable names from block_src absent from the
    replacement.

    Free variables are names that are *read* (appear in a ``Load`` context) but
    not locally *assigned* (``Store``/``Del``) within the original block.  To
    avoid false positives from module-level names (imported symbols, globally-
    defined functions) that the extracted helper can reference directly, the
    check is restricted to names that appear as assignment targets or function
    parameters somewhere in *source* — these are variables that live in a local
    scope and cannot be reached by the helper without being threaded through as
    arguments.

    After this filtering, every remaining name must appear as a bare ``Name``
    node somewhere in the call-site replacements or the helper body.  A name
    that vanishes from both indicates the LLM silently changed the data flow —
    for example by turning a local variable reference into an attribute access
    on one of the parameters (``new_source`` → ``transformer.new_source``).

    Returns the set of names that are absent from both.  An empty set means the
    check passes.  Returns an empty set on any ``SyntaxError`` so a parse
    failure does not block the extraction — the later ``compile()`` guard will
    catch real syntax problems.
    """
    try:
        block_tree = ast.parse(textwrap.dedent(block_src))
    except SyntaxError:
        return set()

    reads: set = set()
    stores: set = set()
    for node in ast.walk(block_tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                reads.add(node.id)
            else:
                stores.add(node.id)

    free_vars = reads - stores
    if not free_vars:
        return set()

    # Restrict to names that are locally assigned or are function/lambda
    # parameters somewhere in the full source.  Module-level names that are
    # only ever read (e.g. imported functions, global constants) are in scope
    # from the helper definition too and do not need to be passed as args.
    source_tree, source_locals = _parse_source_or_empty_set(source)
    if source_tree == None:
        return set()
    for node in ast.walk(source_tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            source_locals.add(node.id)
        elif isinstance(node, ast.arg):
            source_locals.add(node.arg)

    free_vars = free_vars & source_locals
    if not free_vars:
        return set()

    replacement_names: set = set()
    for src in list(call_srcs) + [helper_src]:
        try:
            repl_tree = ast.parse(textwrap.dedent(src))
        except SyntaxError:
            return set()
        for node in ast.walk(repl_tree):
            if isinstance(node, ast.Name):
                replacement_names.add(node.id)

    return free_vars - replacement_names


def _seq_ends_with_return(seq: _SeqInfo) -> bool:
    """Return True if the last top-level statement is a non-None return.

    Detects the case where the LLM includes a ``return`` statement inside the
    duplicate block but the generated replacement omits it, producing a
    function that silently returns ``None`` instead of the original value.

    Bare ``return`` and ``return None`` are excluded: both are semantically
    equivalent to falling off the end of a function, so dropping them in a
    replacement causes no behavioral change.
    """
    try:
        tree = ast.parse(textwrap.dedent(seq.source))
    except SyntaxError:
        return False
    if not tree.body:
        return False
    last = tree.body[-1]
    if not isinstance(last, ast.Return):
        return False
    # Bare `return` and `return None` are equivalent to implicit None return.
    if last.value is None:
        return False
    if isinstance(last.value, ast.Constant) and last.value.value is None:
        return False
    return True


def _replacement_contains_return(replacement: str) -> bool:
    """Return True if *replacement* contains any return statement.

    Wraps the replacement in a dummy function before parsing so that
    ``return`` statements — which are legal inside a function body — do not
    cause false SyntaxError rejections.
    """
    try:
        wrapped = "def _check():\n" + textwrap.indent(
            textwrap.dedent(replacement), "    "
        )
        tree = ast.parse(wrapped)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            return True
    return False


def _replacement_steals_post_block_line(
    group: List[_SeqInfo], call_replacements: List[str], source_lines: List[str]
) -> bool:
    """Return True if any replacement's last line duplicates the line after its block.

    The LLM occasionally appends the first statement *after* the replaced block
    to the end of the replacement text.  When applied, that statement then appears
    twice in the assembled output: once inside the replacement and once as the
    original untouched line.
    """
    for seq, replacement in zip(group, call_replacements):
        next_idx = seq.end_line  # 0-based index of the first line after the block
        if next_idx >= len(source_lines):
            continue
        post_block = source_lines[next_idx].strip()
        if not post_block:
            continue
        repl_lines = [ln.strip() for ln in replacement.splitlines() if ln.strip()]
        if repl_lines and repl_lines[-1] == post_block:
            return True
    return False


def _add_imported_names(node: ast.AST, target_set: set) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split(".")[0]
            target_set.add(name)
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            target_set.add(name)


def _helper_imports_local_name(helper_source: str, original_source: str) -> bool:
    """Return True if helper_source imports a name that is only a local in original.

    Detects the LLM mistake of writing ``import X`` in the helper when ``X``
    was a function parameter or other local name in the original file, not an
    importable module.  Such imports fail at runtime with ModuleNotFoundError.
    """
    try:
        helper_tree = ast.parse(textwrap.dedent(helper_source))
    except SyntaxError:
        return False

    helper_imports: set = set()
    for node in ast.walk(helper_tree):
        _add_imported_names(node, helper_imports)

    if not helper_imports:
        return False

    try:
        orig_tree = ast.parse(original_source)
    except SyntaxError:
        return False

    # Names already imported at the top level of the original file.
    orig_top_imports: set = set()
    for node in orig_tree.body:
        _add_imported_names(node, orig_top_imports)

    new_helper_imports = helper_imports - orig_top_imports
    if not new_helper_imports:
        return False

    # Parameter names in the original file (potential mock-injected locals).
    orig_params: set = set()
    for node in ast.walk(orig_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                orig_params.add(arg.arg)
            if node.args.vararg:
                orig_params.add(node.args.vararg.arg)
            if node.args.kwarg:
                orig_params.add(node.args.kwarg.arg)

    return bool(new_helper_imports & orig_params)


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


def _extract_defined_names(source: str) -> set:
    """Return all function and class names defined anywhere in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


# ---------------------------------------------------------------------------
# Text editing
# ---------------------------------------------------------------------------


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
        min_weight: int = _MIN_WEIGHT,
        max_seq_len: int = _MAX_SEQ_LEN,
        model: str = _MODEL,
        helper_docstrings: bool = False,
        provider: str = "anthropic",
        extraction_retries: int = 1,
        llm_verify_retries: int = 1,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(changed_ranges, source=source, verbose=verbose)
        self._min_weight = min_weight
        self._base_max_seq_len = max_seq_len
        self._model = model
        self._helper_docstrings = helper_docstrings
        self._provider = provider
        self._extraction_retries = extraction_retries
        self._llm_verify_retries = llm_verify_retries
        self._base_url = base_url
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
            self._base_max_seq_len,
        )

        # 7. Collect sequences.
        collector = _SequenceCollector(
            source_lines, max_seq_len=max_seq_len, min_weight=self._min_weight
        )
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
        api_key = _llm_client.get_api_key(self._provider, caller="DuplicateExtractor")
        client = _llm_client.make_client(
            self._provider, api_key, timeout=60.0, base_url=self._base_url
        )
        edits: List[Tuple[int, int, str]] = []
        pending_changes: List[str] = []
        # Extraction groups tracked separately so the final combined check can
        # drop any whose call-site edits were silently overridden by overlapping
        # edits from another group or the func-match pass.
        extraction_groups: List[Tuple[str, List[Tuple[int, int, str]], str]] = []
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
                    is_valid, reason, _veto_notes = _run_with_timeout(
                        _llm_veto_func_match,
                        _API_HARD_TIMEOUT,
                        client,
                        seq,
                        func,
                        source,
                        self._model,
                        self._provider,
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
                            self._model,
                            self._provider,
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
        used_names = _extract_defined_names(source)
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
                is_valid, reason, veto_notes = _run_with_timeout(
                    _llm_veto,
                    _API_HARD_TIMEOUT,
                    client,
                    group,
                    self._model,
                    self._provider,
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

            # Extraction retry loop: attempt extraction up to
            # 1 + _extraction_retries times on algorithmic failure, and up to
            # 1 + _llm_verify_retries additional times on LLM verify failure.
            alg_retries_left = self._extraction_retries
            llm_verify_retries_left = self._llm_verify_retries
            prev_failures: List[str] = []
            prev_output: Optional[dict] = None

            while True:
                try:
                    extraction = _run_with_timeout(
                        _llm_extract,
                        _API_HARD_TIMEOUT,
                        client,
                        group,
                        source,
                        escaping_vars,
                        used_names=frozenset(used_names),
                        model=self._model,
                        helper_docstrings=self._helper_docstrings,
                        provider=self._provider,
                        veto_notes=veto_notes,
                        prev_failures=prev_failures,
                        prev_output=prev_output,
                    )
                except _ApiTimeout:
                    print(
                        "crispen: DuplicateExtractor: API call timed out,"
                        " skipping group",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                if extraction is None:
                    break  # pragma: no cover

                helper_source = extraction["helper_source"]
                if not self._helper_docstrings:
                    helper_source = _strip_helper_docstring(helper_source)
                call_replacements = extraction["call_site_replacements"]
                placement = extraction.get("placement", "module_level")
                func_name = extraction["function_name"]

                _check_failed = False
                _failures: List[str] = []

                # Check 1: name collision
                # Pre-check: staticmethod placement is invalid when sequences span
                # multiple class scopes — flag it so the retry loop can correct it.
                if placement.startswith("staticmethod:"):
                    group_class_scopes = {s.class_scope for s in group}
                    if len(group_class_scopes) != 1 or None in group_class_scopes:
                        _failures.append(
                            "staticmethod placement is invalid when call sites span "
                            "multiple classes or scopes; use module_level instead"
                        )
                        if self.verbose:
                            print(
                                "crispen: DuplicateExtractor: extraction FAILED — "
                                "staticmethod placement invalid for cross-class group",
                                file=sys.stderr,
                                flush=True,
                            )
                        _check_failed = True
                if func_name in used_names:
                    _failures.append(
                        f"name collision: '{func_name}' is already defined,"
                        " choose a different name"
                    )
                    if self.verbose:
                        print(
                            f"crispen: DuplicateExtractor: extraction FAILED — "
                            f"name collision: '{func_name}' is already defined",
                            file=sys.stderr,
                            flush=True,
                        )
                    _check_failed = True

                # Check 2: call site count
                if not _check_failed and len(call_replacements) != len(group):
                    _failures.append(
                        f"wrong call_site_replacements count"
                        f" (expected {len(group)}, got {len(call_replacements)})"
                    )
                    if self.verbose:
                        print(
                            f"crispen: DuplicateExtractor: extraction FAILED — "
                            f"wrong call_site_replacements count "
                            f"(expected {len(group)}, got {len(call_replacements)})",
                            file=sys.stderr,
                            flush=True,
                        )
                        print(
                            f"crispen: DuplicateExtractor:   helper_source: "
                            f"{helper_source!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                        print(
                            f"crispen: DuplicateExtractor:   call_site_replacements: "
                            f"{call_replacements!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                    _check_failed = True

                if not _check_failed:
                    # Normalize each replacement's indentation to match its
                    # original block.  The LLM sometimes returns replacements at
                    # column 0; this re-indents them so the assembled edit is
                    # valid Python.
                    call_replacements = [
                        _normalize_replacement_indentation(seq, r)
                        for seq, r in zip(group, call_replacements)
                    ]

                    # Check 3: post-block line theft
                    if _replacement_steals_post_block_line(
                        group, call_replacements, source_lines
                    ):
                        _failures.append(
                            "replacement duplicates the line after the block"
                        )
                        if self.verbose:
                            print(
                                "crispen: DuplicateExtractor: extraction FAILED — "
                                "replacement duplicates the line after the block",
                                file=sys.stderr,
                                flush=True,
                            )
                        _check_failed = True

                # Check 4: syntax validation
                if not _check_failed and not _verify_extraction(
                    helper_source, call_replacements
                ):
                    _failures.append("invalid helper or replacement syntax")
                    if self.verbose:
                        print(
                            "crispen: DuplicateExtractor: extraction FAILED — "
                            "invalid helper or replacement syntax",
                            file=sys.stderr,
                            flush=True,
                        )
                        print(
                            f"crispen: DuplicateExtractor:   helper_source: "
                            f"{helper_source!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                        print(
                            f"crispen: DuplicateExtractor:   call_site_replacements: "
                            f"{call_replacements!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                    _check_failed = True

                # Check 5: return statement consistency
                if not _check_failed and any(
                    _seq_ends_with_return(seq)
                    and not _replacement_contains_return(repl)
                    for seq, repl in zip(group, call_replacements)
                ):
                    _failures.append("block ends with return but replacement omits it")
                    if self.verbose:
                        print(
                            "crispen: DuplicateExtractor: extraction FAILED — "
                            "block ends with return but replacement omits it",
                            file=sys.stderr,
                            flush=True,
                        )
                    _check_failed = True

                # Check 6: helper must not import local names
                if not _check_failed and _helper_imports_local_name(
                    helper_source, source
                ):
                    _failures.append(
                        "helper imports a name that is a parameter/local"
                        " in the original file"
                    )
                    if self.verbose:
                        print(
                            "crispen: DuplicateExtractor: extraction FAILED — "
                            "helper imports a name that is a parameter/local "
                            "in the original file",
                            file=sys.stderr,
                            flush=True,
                        )
                    _check_failed = True

                # Check 7: new attribute access
                if not _check_failed:
                    new_attrs = _collect_called_attr_names(
                        textwrap.dedent(helper_source)
                    ) - _collect_called_attr_names(source)
                    if new_attrs:
                        _failures.append(
                            f"helper introduces new attribute access(es) not in"
                            f" original: {', '.join(sorted(new_attrs))}"
                        )
                        if self.verbose:
                            print(
                                f"crispen: DuplicateExtractor: extraction FAILED — "
                                f"helper introduces new attribute access(es) not in"
                                f" original: {', '.join(sorted(new_attrs))}",
                                file=sys.stderr,
                                flush=True,
                            )
                        _check_failed = True

                # Check 8: free variable preservation
                if not _check_failed:
                    seq0 = group[0]
                    block_src = "".join(
                        source_lines[seq0.start_line - 1 : seq0.end_line]
                    )
                    missing = _missing_free_vars(
                        block_src, call_replacements, helper_source, source
                    )
                    if missing:
                        _failures.append(
                            f"free variable(s) from original block missing in"
                            f" replacement: {', '.join(sorted(missing))}"
                        )
                        if self.verbose:
                            print(
                                f"crispen: DuplicateExtractor: extraction FAILED — "
                                f"free variable(s) from original block missing in "
                                f"replacement: {', '.join(sorted(missing))}",
                                file=sys.stderr,
                                flush=True,
                            )
                        _check_failed = True

                # Build this group's edits (only if pre-edit checks passed).
                group_edits: List[Tuple[int, int, str]] = []
                candidate = ""
                if not _check_failed:
                    for seq, replacement in zip(group, call_replacements):
                        group_edits.append(
                            (seq.start_line - 1, seq.end_line, replacement)
                        )
                    first_seq = min(group, key=lambda s: s.start_line)
                    if placement.startswith("staticmethod:"):
                        # Insert inside the class body, one line after "class Foo:".
                        scope = placement.split(":", 1)[1]
                        insert_pos = _find_insertion_point(source, scope) + 1
                    else:
                        scope = first_seq.scope
                        insert_pos = _find_insertion_point(source, scope)
                    group_edits.append(
                        _build_helper_insertion(
                            source_lines, insert_pos, helper_source, placement
                        )
                    )
                    # Compile the per-group candidate independently so one bad
                    # extraction doesn't discard valid ones for the same file.
                    candidate = _apply_edits(source, group_edits)

                    # Check 9: assembled output is valid Python
                    try:
                        compile(candidate, "<rewritten>", "exec")
                    except SyntaxError as exc:
                        _failures.append(f"assembled edit not valid Python: {exc}")
                        if self.verbose:
                            print(
                                f"crispen: DuplicateExtractor: extraction FAILED — "
                                f"assembled edit not valid Python: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )
                            print(
                                f"crispen: DuplicateExtractor:   helper_source: "
                                f"{helper_source!r}",
                                file=sys.stderr,
                                flush=True,
                            )
                            print(
                                f"crispen: DuplicateExtractor:"
                                f"   call_site_replacements: "
                                f"{call_replacements!r}",
                                file=sys.stderr,
                                flush=True,
                            )
                        _check_failed = True

                    # Check 10: extracted function is actually called
                    if not _check_failed and not _has_call_to(func_name, candidate):
                        _failures.append(
                            f"'{func_name}' not called in candidate output"
                        )
                        if self.verbose:
                            print(
                                f"crispen: DuplicateExtractor: extraction FAILED — "
                                f"'{func_name}' not called in candidate output",
                                file=sys.stderr,
                                flush=True,
                            )
                        _check_failed = True

                    # Check 11: no new undefined names
                    if not _check_failed:
                        undef = _pyflakes_new_undefined_names(source, candidate)
                        if undef:
                            _failures.append(
                                f"undefined name(s) introduced by edit: "
                                f"{', '.join(sorted(undef))}"
                            )
                            if self.verbose:
                                print(
                                    f"crispen: DuplicateExtractor:"
                                    f" extraction FAILED — "
                                    f"undefined name(s) introduced by edit: "
                                    f"{', '.join(sorted(undef))}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            _check_failed = True

                # Retry decision for algorithmic failures
                if _check_failed:
                    if alg_retries_left > 0:
                        alg_retries_left -= 1
                        prev_failures = _failures
                        prev_output = None
                        if self.verbose:
                            print(
                                f"crispen: DuplicateExtractor:   → retrying"
                                f" extraction ({alg_retries_left} retries"
                                f" remaining after algorithmic failure)",
                                file=sys.stderr,
                                flush=True,
                            )
                        continue
                    break  # exhausted algorithmic retries — skip group

                # ---- LLM verification step ----
                try:
                    verify_ok, verify_issues = _run_with_timeout(
                        _llm_verify_extraction,
                        _API_HARD_TIMEOUT,
                        client,
                        group,
                        helper_source,
                        call_replacements,
                        source,
                        self._model,
                        self._provider,
                    )
                except _ApiTimeout:
                    if self.verbose:
                        print(
                            "crispen: DuplicateExtractor:   → verify timed out,"
                            " accepting extraction",
                            file=sys.stderr,
                            flush=True,
                        )
                    verify_ok, verify_issues = True, []

                if self.verbose:
                    v_status = "ACCEPTED" if verify_ok else "REJECTED"
                    print(
                        f"crispen: DuplicateExtractor:   → verify {v_status}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if not verify_ok:
                        for issue in verify_issues:
                            print(
                                f"crispen: DuplicateExtractor:" f"     issue: {issue}",
                                file=sys.stderr,
                                flush=True,
                            )

                if not verify_ok:
                    if llm_verify_retries_left > 0:
                        llm_verify_retries_left -= 1
                        prev_failures = [
                            f"LLM verification issue: {i}" for i in verify_issues
                        ]
                        prev_output = extraction
                        if self.verbose:
                            print(
                                f"crispen: DuplicateExtractor:   → retrying"
                                f" extraction after verify rejection"
                                f" ({llm_verify_retries_left} retries remaining)",
                                file=sys.stderr,
                                flush=True,
                            )
                        continue
                    break  # exhausted LLM verify retries — skip group

                # ---- All checks passed: accept this extraction ----
                used_names.add(func_name)
                if self.verbose:
                    print(
                        f"crispen: DuplicateExtractor: extracting '{func_name}'",
                        file=sys.stderr,
                        flush=True,
                    )
                extraction_groups.append(
                    (
                        func_name,
                        group_edits,
                        f"DuplicateExtractor: extracted '{func_name}' "
                        f"from {len(group)} duplicate blocks",
                    )
                )
                break  # done with this group

        # 18. Combine all accepted edits, verify all extracted functions are
        # actually called in the combined output, then write.
        all_edits = list(edits)
        for _, g_edits, _ in extraction_groups:
            all_edits.extend(g_edits)

        if all_edits:
            combined = _apply_edits(source, all_edits)

            # Drop any extraction group whose extracted function is not called
            # in the combined output.  This happens when call-site edits are
            # silently skipped by the overlap detector because they conflict
            # with edits from another group or from the func-match pass.
            uncalled = {
                name
                for name, _, _ in extraction_groups
                if not _has_call_to(name, combined)
            }
            if uncalled:
                for name in sorted(uncalled):
                    if self.verbose:
                        print(
                            f"crispen: DuplicateExtractor: extraction DROPPED — "
                            f"'{name}' not called in combined output "
                            f"(call-site edits overridden by overlapping edits)",
                            file=sys.stderr,
                            flush=True,
                        )
                extraction_groups = [
                    (n, g, m) for n, g, m in extraction_groups if n not in uncalled
                ]
                all_edits = list(edits)
                for _, g_edits, _ in extraction_groups:
                    all_edits.extend(g_edits)
                combined = _apply_edits(source, all_edits)

            all_pending = list(pending_changes)
            for _, _, msg in extraction_groups:
                all_pending.append(msg)

            if all_edits:
                self._new_source = combined
                self.changes_made.extend(all_pending)

    def get_rewritten_source(self) -> Optional[str]:
        return self._new_source
