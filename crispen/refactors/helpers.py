from __future__ import annotations
import ast
import re
import textwrap
from typing import List
import libcst as cst
from .extractors import _extract_import_names, _names_assigned_in
from .models import _SeqInfo


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


def _has_def(stmts: List[cst.BaseStatement]) -> bool:
    """Return True if any top-level statement is a function or class definition."""
    return any(isinstance(s, (cst.FunctionDef, cst.ClassDef)) for s in stmts)


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
        _extract_import_names(node, helper_imports)

    if not helper_imports:
        return False

    try:
        orig_tree = ast.parse(original_source)
    except SyntaxError:
        return False

    # Names already imported at the top level of the original file.
    orig_top_imports: set = set()
    for node in orig_tree.body:
        _extract_import_names(node, orig_top_imports)

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
