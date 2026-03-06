from __future__ import annotations
import textwrap
import threading
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
