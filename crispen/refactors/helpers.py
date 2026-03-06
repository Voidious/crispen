from __future__ import annotations
import ast
import textwrap
import libcst as cst
from .sequence_models import _SeqInfo


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
