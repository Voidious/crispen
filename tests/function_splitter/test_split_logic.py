from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import (
    _choose_best_split,
    _find_valid_splits,
    _func_in_changed_range,
    _generate_call,
    _generate_helper_source,
    _has_nested_funcdef,
    _has_new_undefined_names,
    _has_yield,
    _module_global_names,
)
from .utils import _make_func_info, _parse_func
import textwrap
import libcst as cst


def test_find_valid_splits_all_valid():
    src = "def foo():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n"
    stmts, positions, lines = _parse_func(src)
    # With a very loose limit, all splits should be valid
    result = _find_valid_splits(stmts, positions, max_lines=1000)
    assert len(result) > 0
    # Ordered latest first
    assert result == sorted(result, reverse=True)


def test_find_valid_splits_none_valid():
    # max_lines=1 means even a 1-stmt head (+ return call = 2 lines) is invalid
    src = "def foo():\n    a = 1\n    b = 2\n    c = 3\n"
    stmts, positions, lines = _parse_func(src)
    result = _find_valid_splits(stmts, positions, max_lines=1)
    assert result == []


def test_find_valid_splits_stops_at_max_candidates():
    # 7 statements → iterates from 6 down, stops after 5 valid candidates
    src = "def foo():\n" + "".join(f"    a{i} = {i}\n" for i in range(7))
    stmts, positions, lines = _parse_func(src)
    result = _find_valid_splits(stmts, positions, max_lines=1000)
    assert len(result) == 5


def test_find_valid_splits_fewer_than_max():
    # 4 statements → at most 3 valid splits (indices 3, 2, 1)
    src = "def foo():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n"
    stmts, positions, lines = _parse_func(src)
    result = _find_valid_splits(stmts, positions, max_lines=1000)
    assert 1 <= len(result) <= 3


def test_find_valid_splits_empty_body():
    # Should not crash with an empty list (though normally not called)
    result = _find_valid_splits([], {}, max_lines=1000)
    assert result == []


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


def test_module_global_names_imports():
    source = "import ast\nfrom pathlib import Path\nimport libcst as cst\n"
    result = _module_global_names(source)
    assert "ast" in result
    assert "Path" in result
    assert "cst" in result


def test_module_global_names_functions_and_classes():
    source = "def foo():\n    pass\n\nclass Bar:\n    pass\n"
    result = _module_global_names(source)
    assert "foo" in result
    assert "Bar" in result


def test_module_global_names_assignments():
    source = "_CONST = frozenset()\nVALUE: int = 42\n"
    result = _module_global_names(source)
    assert "_CONST" in result
    assert "VALUE" in result


def test_module_global_names_syntax_error():
    result = _module_global_names("def foo(")
    assert result == set()


def test_module_global_names_tuple_assign_target_not_collected():
    # Tuple-unpacking: Assign target is a Tuple node, not a Name → skipped
    source = "a, b = 1, 2\n"
    result = _module_global_names(source)
    assert "a" not in result
    assert "b" not in result


def test_module_global_names_ann_assign_non_name_target_skipped():
    # AnnAssign where target is an Attribute, not a Name → skipped
    source = "Foo.x: int\n"
    result = _module_global_names(source)
    assert "x" not in result


def test_generate_helper_source_with_staticmethod():
    result = _generate_helper_source(
        name="process",
        params=["x", "y"],
        tail_source="return x + y\n",
        func_indent="    ",
        is_static=True,
        add_docstring=False,
    )
    assert "@staticmethod" in result
    assert "def _process(x, y):" in result
    assert "return x + y" in result
    assert result.startswith("    @staticmethod")


def test_generate_helper_source_without_staticmethod():
    result = _generate_helper_source(
        name="process",
        params=["x"],
        tail_source="return x * 2\n",
        func_indent="",
        is_static=False,
        add_docstring=False,
    )
    assert "@staticmethod" not in result
    assert "def _process(x):" in result
    assert "return x * 2" in result


def test_generate_helper_source_with_docstring():
    result = _generate_helper_source(
        name="process",
        params=[],
        tail_source="return 42\n",
        func_indent="",
        is_static=False,
        add_docstring=True,
    )
    assert '"""' in result
    assert "return 42" in result


def test_generate_helper_source_instance_method():
    result = _generate_helper_source(
        name="process",
        params=["a"],
        tail_source="return self.x + a\n",
        func_indent="    ",
        is_static=False,
        add_docstring=False,
        is_instance_method=True,
    )
    assert "@staticmethod" not in result
    assert "def _process(self, a):" in result
    assert "return self.x + a" in result


def test_generate_helper_source_indentation_correct():
    result = _generate_helper_source(
        name="helper",
        params=[],
        tail_source="x = 1\ny = 2\n",
        func_indent="    ",
        is_static=False,
        add_docstring=False,
    )
    # Body should be indented by 8 spaces (func_indent=4 + body_indent=4)
    assert "        x = 1" in result
    assert "        y = 2" in result


def test_generate_call_with_class():
    result = _generate_call("helper", ["x", "y"], "MyClass", "    ")
    assert result == "    return MyClass._helper(x, y)"


def test_generate_call_module_level():
    result = _generate_call("helper", ["a"], None, "        ")
    assert result == "        return _helper(a)"


def test_generate_call_no_params():
    result = _generate_call("do_work", [], None, "    ")
    assert result == "    return _do_work()"


def test_generate_call_class_no_params():
    result = _generate_call("do_work", [], "Foo", "    ")
    assert result == "    return Foo._do_work()"


def test_generate_call_instance_method():
    result = _generate_call("process", ["a", "b"], "MyClass", "    ", True)
    assert result == "    return self._process(a, b)"


def test_generate_call_instance_method_no_params():
    result = _generate_call("process", [], "MyClass", "    ", True)
    assert result == "    return self._process()"


def test_has_yield_simple():
    src = "def gen():\n    yield 1\n"
    func = cst.parse_module(src).body[0]
    assert _has_yield(func) is True


def test_has_yield_from():
    src = "def gen():\n    yield from [1, 2]\n"
    func = cst.parse_module(src).body[0]
    assert _has_yield(func) is True


def test_has_yield_none():
    src = "def foo():\n    return 1\n"
    func = cst.parse_module(src).body[0]
    assert _has_yield(func) is False


def test_has_yield_nested_not_counted():
    src = textwrap.dedent(
        """\
        def foo():
            def inner():
                yield 1
            return inner
    """
    )
    func = cst.parse_module(src).body[0]
    # yield is inside nested function, should not count
    assert _has_yield(func) is False


def test_has_nested_funcdef_with_nested():
    src = textwrap.dedent(
        """\
        def outer():
            x = 1
            def inner():
                return x
            return inner
    """
    )
    func = cst.parse_module(src).body[0]
    assert _has_nested_funcdef(func) is True


def test_has_nested_funcdef_without_nested():
    src = "def foo():\n    x = 1\n    return x\n"
    func = cst.parse_module(src).body[0]
    assert _has_nested_funcdef(func) is False


def test_has_nested_funcdef_first_stmt():
    # Nested funcdef is the very first statement in the body
    src = textwrap.dedent(
        """\
        def outer():
            def inner():
                pass
            return inner()
    """
    )
    func = cst.parse_module(src).body[0]
    assert _has_nested_funcdef(func) is True


def test_func_in_changed_range_overlaps():
    fi = _make_func_info(5, 15)
    assert _func_in_changed_range(fi, [(1, 10)]) is True


def test_func_in_changed_range_no_overlap():
    fi = _make_func_info(5, 10)
    assert _func_in_changed_range(fi, [(20, 30)]) is False


def test_func_in_changed_range_adjacent():
    fi = _make_func_info(5, 10)
    assert _func_in_changed_range(fi, [(10, 20)]) is True


def test_has_new_undefined_names_no_new():
    """No new undefined names → returns False."""
    before = "x = 1\ny = x + 1\n"
    after = "x = 1\ny = x + 1\nz = y + 1\n"
    assert _has_new_undefined_names(before, after) is False


def test_has_new_undefined_names_introduced():
    """After introduces an undefined name that before didn't have → returns True."""
    before = "x = 1\n"
    after = "x = undefined_var\n"
    assert _has_new_undefined_names(before, after) is True


def test_has_new_undefined_names_non_undefined_warning():
    """Non-UndefinedName pyflakes warning (e.g. UnusedImport) → returns False."""
    # An unused import produces an UnusedImport warning, not UndefinedName.
    # This exercises the isinstance() False branch inside _Collector.flake.
    before = ""
    after = "import os\n"
    assert _has_new_undefined_names(before, after) is False


def test_has_new_undefined_names_exception():
    """If pyflakes raises an unexpected exception, returns False (safe default)."""
    with patch("pyflakes.api.check", side_effect=RuntimeError("boom")):
        assert _has_new_undefined_names("x = 1\n", "y = 1\n") is False
