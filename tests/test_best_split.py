from __future__ import annotations
import textwrap
from crispen.refactors.function_splitter import _choose_best_split
from .test_valid_splits import _parse_func


def test_choose_best_split_fewest_params():
    # Two splits: one has free vars, one doesn't
    src = textwrap.dedent(
        """\
        def foo(external):
            a = 1
            b = external + 1
    """
    )
    stmts, positions, lines = _parse_func(src)
    # split_idx=1: tail=[b=external+1] → free vars: [external]
    # split_idx=2: tail=[] → but we need at least 1 stmt in tail,
    # so valid splits are [1] only for 2-stmt function
    # Let's use 3 stmts with different free var counts
    src2 = textwrap.dedent(
        """\
        def foo(ext):
            a = 1
            b = ext + 1
            c = a + b
    """
    )
    stmts2, positions2, lines2 = _parse_func(src2)
    # split_idx=1: tail=[b=ext+1, c=a+b] → free vars: [a, ext] (a from head)
    # Actually 'a' is assigned in head (split_idx=1 → head=[a=1]) and used in tail
    # So tail [b=ext+1, c=a+b] has free vars: [a, ext]
    # split_idx=2: tail=[c=a+b] → free vars: [a, b] (assigned in head)
    # Wait no, head=[a=1, b=ext+1] so tail=[c=a+b] has free vars: [a, b]
    # split_idx=3: not valid (needs at least 1 in tail)
    # So split_idx=1 has 2 free vars [a, ext], split_idx=2 has 2 free vars [a, b]
    # Tie → choose earliest in list = latest split = 2
    valid_splits = [2, 1]  # latest first
    split_idx, params, _ = _choose_best_split(
        stmts2, valid_splits, lines2, positions2, ["ext"]
    )
    # Both have 2 free vars, tie broken by latest (first in list) = 2
    assert split_idx == 2


def test_choose_best_split_fewer_params_wins():
    # Use a source where one split clearly has fewer params
    src = textwrap.dedent(
        """\
        def foo():
            a = 1
            b = 2
            c = a + b
    """
    )
    stmts, positions, lines = _parse_func(src)
    # split_idx=1: tail=[b=2, c=a+b] → free vars: [a] (1 free var)
    # split_idx=2: tail=[c=a+b] → free vars: [a, b] (2 free vars)
    valid_splits = [2, 1]
    split_idx, params, _ = _choose_best_split(stmts, valid_splits, lines, positions, [])
    # split_idx=1 has 1 free var (a) vs split_idx=2 has 2 free vars (a, b)
    assert split_idx == 1
    assert params == ["a"]


def test_choose_best_split_single_candidate():
    src = "def foo():\n    x = 1\n    y = 2\n"
    stmts, positions, lines = _parse_func(src)
    split_idx, params, _ = _choose_best_split(stmts, [1], lines, positions, [])
    assert split_idx == 1


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


def test_choose_best_split_empty_splits_returns_none():
    # No valid split candidates → None returned
    src = "def foo():\n    x = 1\n    y = 2\n"
    stmts, positions, lines = _parse_func(src)
    result = _choose_best_split(stmts, [], lines, positions, [])
    assert result is None


def test_choose_best_split_filters_module_globals():
    # Tail references a module-level import; it must not appear in params.
    src = textwrap.dedent(
        """\
        def foo():
            x = 1
            y = os.path.join("a", "b")
        """
    )
    stmts, positions, lines = _parse_func(src)
    # Without filtering: "os" would be a free var of the tail.
    # With module_globals={"os"}: "os" is filtered out → params = []
    result = _choose_best_split(stmts, [1], lines, positions, [], module_globals={"os"})
    assert result is not None
    _, params, _ = result
    assert "os" not in params
