"""Refactor: extract duplicate code blocks into helper functions using an LLM."""

from __future__ import annotations

import ast
import os
import re
import sys
import textwrap
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


# ---------------------------------------------------------------------------
# Sequence collector
# ---------------------------------------------------------------------------


class _SequenceCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, source_lines: List[str]) -> None:
        self.sequences: List[_SeqInfo] = []
        self._scope_stack: List[str] = ["<module>"]
        self._source_lines = source_lines

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
            for end_i in range(start_i + 1, min(start_i + _MAX_SEQ_LEN + 1, n + 1)):
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
# Duplicate group finding
# ---------------------------------------------------------------------------


def _overlaps_diff(seq: _SeqInfo, changed_ranges: List[Tuple[int, int]]) -> bool:
    return any(
        seq.start_line <= r_end and seq.end_line >= r_start
        for r_start, r_end in changed_ranges
    )


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
) -> Optional[dict]:
    blocks_text = "\n\n".join(
        f"Block {i + 1} (scope: {s.scope}, lines {s.start_line}-{s.end_line}):\n"
        f"```python\n{s.source.rstrip()}\n```"
        for i, s in enumerate(group)
    )
    snippet = full_source[:4000] if len(full_source) > 4000 else full_source
    prompt = (
        "Extract the following duplicate code blocks from this Python file into a "
        f"helper function.\n\nFile source:\n```python\n{snippet}\n```\n\n"
        f"Duplicate blocks:\n{blocks_text}\n\n"
        "Place the helper immediately before the enclosing function of its first use. "
        "If both call sites are inside the same class, use a @staticmethod. "
        "Return complete, valid Python for the helper and each call site replacement."
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


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _verify_extraction(helper_source: str, call_replacements: List[str]) -> bool:
    """Verify the extraction produces syntactically valid Python.

    Replacements are dedented before checking since they may be
    indented for their original context inside a function body.
    """
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


# ---------------------------------------------------------------------------
# Text editing
# ---------------------------------------------------------------------------


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
        try:
            tree = cst.parse_module(source)
        except cst.ParserSyntaxError:
            return

        source_lines = source.splitlines(keepends=True)
        collector = _SequenceCollector(source_lines)
        MetadataWrapper(tree).visit(collector)

        groups = _find_duplicate_groups(collector.sequences, self.changed_ranges)
        if not groups:
            return

        if self.verbose:
            print(
                f"crispen: DuplicateExtractor: found {len(groups)} duplicate group(s)",
                file=sys.stderr,
            )

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise CrispenAPIError(
                "DuplicateExtractor: ANTHROPIC_API_KEY is not set.\n"
                "Commit blocked. To skip all hooks: git commit --no-verify"
            )

        client = anthropic.Anthropic(api_key=api_key)
        edits: List[Tuple[int, int, str]] = []
        pending_changes: List[str] = []

        for group in groups:
            if self.verbose:
                ranges_str = ", ".join(
                    f"lines {s.start_line}-{s.end_line}" for s in group
                )
                print(
                    f"crispen: DuplicateExtractor: veto check — "
                    f"scope '{group[0].scope}': {ranges_str}",
                    file=sys.stderr,
                )
            is_valid, reason = _llm_veto(client, group)
            if self.verbose:
                status = "ACCEPTED" if is_valid else "VETOED"
                print(
                    f"crispen: DuplicateExtractor:   → {status}: {reason}",
                    file=sys.stderr,
                )
            if not is_valid:
                continue

            extraction = _llm_extract(client, group, source)
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
                )
            pending_changes.append(
                f"DuplicateExtractor: extracted '{func_name}' "
                f"from {len(group)} duplicate blocks"
            )

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
                    )
            else:
                self._new_source = candidate
                self.changes_made.extend(pending_changes)

    def get_rewritten_source(self) -> Optional[str]:
        return self._new_source
