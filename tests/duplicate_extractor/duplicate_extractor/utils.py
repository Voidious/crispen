from unittest.mock import MagicMock
import textwrap
from crispen.refactors.duplicate_extractor import _FunctionInfo, _SeqInfo


def _make_extract_response(data: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "extract_helper"
    block.input = data
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_verify_response(is_correct: bool, issues: list) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "verify_extraction"
    block.input = {"is_correct": is_correct, "issues": issues}
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_veto_response(is_valid: bool, reason: str = "test") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "evaluate_duplicate"
    block.input = {"is_valid_duplicate": is_valid, "reason": reason}
    resp = MagicMock()
    resp.content = [block]
    return resp


_DUP_SOURCE = textwrap.dedent(
    """\
    def foo():
        if debug:
            pass
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def bar():
        result = None
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_DUP_RANGES = [(10, 12)]  # overlaps bar's duplicate block (x/y/z lines)

# Source where foo's duplicate block assigns z, and foo uses z after the block.
# _has_escaping_vars should detect this and skip the extraction.
_ESC_SOURCE = textwrap.dedent(
    """\
    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
        assert z == expected

    def bar():
        result = None
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_ESC_RANGES = [(9, 11)]  # overlaps bar's duplicate block (x/y/z lines)


_POST_STEAL_SOURCE = textwrap.dedent(
    """\
    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
        return z

    def bar():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
        logger.info("done")
    """
)
_POST_STEAL_RANGES = [(8, 10)]  # overlaps bar's 3-statement block


# Source with two structurally distinct duplicate pairs so _find_duplicate_groups
# returns two separate groups.  The groups differ in argument count so that
# _ASTNormalizer produces different fingerprints for each group:
#   group 1 (foo/bar): 3-stmt bodies using 2-argument calls → fingerprint A
#   group 2 (baz/qux): 3-stmt bodies using 3-argument calls → fingerprint B
_TWO_PAIR_SOURCE = textwrap.dedent(
    """\
    import os

    def foo():
        if debug:
            pass
        x = compute(data, config)
        y = transform(x, scale)
        z = finalize(y, mode)

    def bar():
        result = None
        x = compute(data, config)
        y = transform(x, scale)
        z = finalize(y, mode)

    def baz():
        if debug:
            pass
        a = process(item, key, idx)
        b = convert(a, fmt, enc)
        c = export(b, path, opts)

    def qux():
        result = None
        a = process(item, key, idx)
        b = convert(a, fmt, enc)
        c = export(b, path, opts)
    """
)
_TWO_PAIR_RANGES = [(4, 30)]  # overlaps all duplicate sequences


# Source that already defines _helper AND has duplicate blocks.
_COLLISION_SOURCE = textwrap.dedent(
    """\
    def _helper(x):
        return x

    def foo():
        if debug:
            pass
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def bar():
        result = None
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_COLLISION_RANGES = [(12, 14)]  # overlaps bar's duplicate block


def _make_proxy_seq(stmts_count: int, scope: str, class_scope=None) -> _SeqInfo:
    """Build a _SeqInfo with a synthetic stmts list of the given length."""
    return _SeqInfo(
        stmts=[None] * stmts_count,  # type: ignore[list-item]
        start_line=1,
        end_line=stmts_count,
        scope=scope,
        source="",
        fingerprint="",
        class_scope=class_scope,
    )


def _make_proxy_func(
    name: str, body_stmt_count: int, scope: str = "<module>"
) -> _FunctionInfo:
    return _FunctionInfo(
        name=name,
        source=f"def {name}(): pass\n",
        scope=scope,
        body_source="    pass\n",
        body_stmt_count=body_stmt_count,
        params=[],
    )


_PROXY_SOURCE = textwrap.dedent(
    """\
    def foo():
        setup = prepare(data)
        x = compute(data)
        y = transform(x)
        z = finalize(y)
        return setup, z

    def bar():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
# overlaps foo: foo has 5 stmts but duplicate block is only 3 of them (not a proxy);
# bar has 3 stmts = its entire body (would become a proxy) → mixed → guard fires.
_PROXY_RANGES = [(1, 11)]
