from __future__ import annotations
import textwrap
from crispen.refactors.function_splitter import _choose_best_split
from tests.fs_ast_parse_helpers_v2 import _parse_func


def test_choose_best_split_self_in_tail_returns_instance_method():
    # Tail requires self → extracted as instance method, not static
    src = textwrap.dedent(
        """\
        class Foo:
            def method(self, x):
                a = 1
                b = self.value + a
        """
    )
    stmts, positions, lines = _parse_func(src)
    # split_idx=1: tail=[b = self.value + a] → free: [a, self] → instance method
    result = _choose_best_split(stmts, [1], lines, positions, ["self", "x"])
    assert result is not None
    split_idx, params, is_instance_method = result
    assert split_idx == 1
    assert is_instance_method is True
    assert "self" not in params  # self is implicit, not in params list
    assert "a" in params  # a is still a real param
