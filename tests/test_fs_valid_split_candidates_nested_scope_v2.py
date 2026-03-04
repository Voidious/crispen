from __future__ import annotations
import textwrap
from crispen.refactors.function_splitter import _find_valid_splits
from tests.fs_ast_parse_helpers_v2 import _parse_func


def test_find_valid_splits_nested_funcdef_restricts_upper():
    # First nested funcdef at index 2 → valid splits only at indices ≤ 2.
    src = textwrap.dedent(
        """\
        def outer():
            a = 1
            b = 2
            def inner():
                pass
            c = 3
            d = 4
    """
    )
    stmts, positions, lines = _parse_func(src)
    # body_stmts: [a=1, b=2, def inner, c=3, d=4]
    # First nested funcdef at index 2 → upper=2 → range(2, 0, -1) = [2, 1]
    result = _find_valid_splits(stmts, positions, max_lines=1000)
    assert all(i <= 2 for i in result)
    assert 3 not in result
    assert 4 not in result
