"""Tests for function_splitter: 100% branch coverage."""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

import libcst as cst
import pytest
from libcst.metadata import MetadataWrapper, PositionProvider

from crispen.refactors.function_splitter import (
    _ApiTimeout,
    _FuncInfo,
    _FunctionCollector,
    _SplitTask,
    _choose_best_split,
    _count_body_lines,
    _cyclomatic_complexity,
    _extract_func_source,
    _find_free_vars,
    _find_valid_splits,
    _func_in_changed_range,
    _generate_call,
    _generate_helper_source,
    _has_yield,
    _head_effective_lines,
    _is_docstring_stmt,
    _llm_name_helpers,
    _run_with_timeout,
    _stmts_source,
    FunctionSplitter,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _parse_func(source: str):
    """Return (body_stmts, positions, source_lines) for the first function.

    Uses a CSTVisitor to capture body_stmts from the wrapper's internal copy,
    ensuring they match the keys in the positions dict.
    """
    tree = cst.parse_module(source)
    wrapper = MetadataWrapper(tree)
    positions = wrapper.resolve(PositionProvider)

    class _Getter(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def __init__(self):
            self.stmts: list = []

        def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
            if not self.stmts:  # first function only
                self.stmts = list(node.body.body)

    getter = _Getter()
    wrapper.visit(getter)
    source_lines = source.splitlines(keepends=True)
    return getter.stmts, positions, source_lines


def _make_mock_response(names_list):
    """Build a mock Anthropic message response for the name_helper_functions tool."""
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "name_helper_functions"
    mock_block.input = {
        "names": [{"id": str(i), "name": n} for i, n in enumerate(names_list)]
    }
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    return mock_response


# ---------------------------------------------------------------------------
# _is_docstring_stmt
# ---------------------------------------------------------------------------


def _parse_stmt(src: str) -> cst.BaseStatement:
    return cst.parse_module(src).body[0]


def test_is_docstring_triple_quoted():
    stmt = _parse_stmt('def f():\n    """doc"""\n').body.body[0]
    assert _is_docstring_stmt(stmt) is True


def test_is_docstring_single_quoted():
    stmt = _parse_stmt("def f():\n    'doc'\n").body.body[0]
    assert _is_docstring_stmt(stmt) is True


def test_is_docstring_concatenated():
    stmt = _parse_stmt('def f():\n    "foo" "bar"\n').body.body[0]
    assert _is_docstring_stmt(stmt) is True


def test_is_docstring_non_docstring_expr():
    # A numeric literal is not a docstring
    stmt = _parse_stmt("def f():\n    42\n").body.body[0]
    assert _is_docstring_stmt(stmt) is False


def test_is_docstring_import():
    stmt = _parse_stmt("import os\n")
    assert _is_docstring_stmt(stmt) is False


def test_is_docstring_assignment():
    stmt = _parse_stmt("x = 1\n")
    assert _is_docstring_stmt(stmt) is False


def test_is_docstring_two_stmts_on_line():
    # Two statements on one line — len(body) != 1
    stmt = _parse_stmt("x = 1; y = 2\n")
    assert _is_docstring_stmt(stmt) is False


# ---------------------------------------------------------------------------
# _count_body_lines
# ---------------------------------------------------------------------------


def test_count_body_lines_no_docstring():
    src = "def foo():\n    x = 1\n    y = 2\n    z = 3\n"
    assert _count_body_lines(src) == 3


def test_count_body_lines_with_docstring():
    src = 'def foo():\n    """doc"""\n    x = 1\n    y = 2\n'
    # docstring skipped; body is lines 2 (x=1) and 3 (y=2)
    assert _count_body_lines(src) == 2


def test_count_body_lines_multiline_docstring():
    src = 'def foo():\n    """line1\n    line2\n    """\n    x = 1\n'
    # docstring spans lines 2-4; body starts at x=1 (line 5)
    result = _count_body_lines(src)
    assert result == 1


def test_count_body_lines_only_docstring():
    # Body has only a docstring → effectively empty
    src = 'def foo():\n    """doc"""\n'
    assert _count_body_lines(src) == 0


def test_count_body_lines_parse_error():
    assert _count_body_lines("def f(\n  !!invalid") == 0


def test_count_body_lines_no_funcdef():
    # Module-level code, no function
    assert _count_body_lines("x = 1\n") == 0


# ---------------------------------------------------------------------------
# _cyclomatic_complexity
# ---------------------------------------------------------------------------


def test_cyclomatic_flat():
    src = "def foo():\n    x = 1\n    return x\n"
    assert _cyclomatic_complexity(src) == 1


def test_cyclomatic_if_elif():
    src = textwrap.dedent(
        """\
        def foo(x):
            if x > 0:
                return 1
            elif x < 0:
                return -1
            return 0
    """
    )
    # base=1, if=+1, elif is nested If=+1 → 3
    assert _cyclomatic_complexity(src) == 3


def test_cyclomatic_for_while():
    src = textwrap.dedent(
        """\
        def foo(items):
            for i in items:
                pass
            while True:
                break
    """
    )
    # base=1, for=+1, while=+1 → 3
    assert _cyclomatic_complexity(src) == 3


def test_cyclomatic_try_except():
    src = textwrap.dedent(
        """\
        def foo():
            try:
                x = 1
            except ValueError:
                pass
            except TypeError:
                pass
    """
    )
    # base=1, ExceptHandler x2 = +2 → 3
    assert _cyclomatic_complexity(src) == 3


def test_cyclomatic_ternary():
    src = "def foo(x):\n    y = 1 if x else 0\n"
    # base=1, IfExp=+1 → 2
    assert _cyclomatic_complexity(src) == 2


def test_cyclomatic_nested_functions_excluded():
    src = textwrap.dedent(
        """\
        def foo():
            def inner():
                if True:
                    pass
            return inner
    """
    )
    # base=1; inner's if is NOT counted → 1
    assert _cyclomatic_complexity(src) == 1


def test_cyclomatic_parse_error():
    assert _cyclomatic_complexity("def f(\n  !!") == 1


def test_cyclomatic_no_funcdef_uses_module_body():
    # When there's no function, use module body
    src = "if True:\n    pass\n"
    # base=1, If=+1 → 2
    assert _cyclomatic_complexity(src) == 2


# ---------------------------------------------------------------------------
# _find_free_vars
# ---------------------------------------------------------------------------


def test_find_free_vars_all_local():
    src = "x = 1\ny = x + 1\n"
    assert _find_free_vars(src) == []


def test_find_free_vars_one_free():
    src = "y = external_var + 1\n"
    result = _find_free_vars(src)
    assert "external_var" in result
    assert "y" not in result


def test_find_free_vars_builtins_excluded():
    src = "print(len([1, 2, 3]))\n"
    result = _find_free_vars(src)
    assert "print" not in result
    assert "len" not in result


def test_find_free_vars_nested_function_not_recursed():
    src = "def inner():\n    return outer_var\n"
    # outer_var is used inside nested function — not recursed into
    assert _find_free_vars(src) == []


def test_find_free_vars_nested_class_not_recursed():
    src = "class Inner:\n    x = class_var\n"
    # class_var inside nested class — not recursed
    assert _find_free_vars(src) == []


def test_find_free_vars_for_target_not_free():
    src = "for item in some_list:\n    pass\n"
    result = _find_free_vars(src)
    # item is a store, some_list is a load
    assert "item" not in result
    assert "some_list" in result


def test_find_free_vars_import_not_free():
    src = "import os\npath = os.getcwd()\n"
    result = _find_free_vars(src)
    # os is imported (stored), path is stored
    assert "os" not in result
    assert "path" not in result


def test_find_free_vars_import_from_not_free():
    src = "from os import path\nresult = path.join('a', 'b')\n"
    result = _find_free_vars(src)
    assert "path" not in result


def test_find_free_vars_parse_error():
    assert _find_free_vars("def f(\n  !!") == []


def test_find_free_vars_del_is_store():
    src = "del some_name\n"
    # some_name has Del context (not Load) — not treated as free
    result = _find_free_vars(src)
    assert "some_name" not in result


def test_find_free_vars_augassign_free():
    # weight += 1 reads weight before writing — weight must come from outside
    src = "weight += 1\n"
    result = _find_free_vars(src)
    assert "weight" in result


def test_find_free_vars_augassign_already_defined():
    # weight is unconditionally assigned first, so AugAssign doesn't need it free
    src = "weight = 0\nweight += 1\n"
    result = _find_free_vars(src)
    assert "weight" not in result


def test_find_free_vars_augassign_subscript():
    # data[0] += 1: target is a subscript, data is loaded
    src = "data[0] += 1\n"
    result = _find_free_vars(src)
    assert "data" in result


def test_find_free_vars_for_orelse():
    # for-else: orelse runs when loop completes normally
    src = "for item in data:\n    pass\nelse:\n    fallback()\n"
    result = _find_free_vars(src)
    assert "item" not in result  # for target is locally scoped
    assert "data" in result
    assert "fallback" in result  # used in orelse, not locally defined


def test_find_free_vars_with_target():
    # with-statement target is locally scoped inside the body
    src = "with open(filename) as fp:\n    content = fp.read()\n"
    result = _find_free_vars(src)
    assert "fp" not in result  # with target, locally scoped
    assert "filename" in result  # context_expr is free


def test_find_free_vars_with_no_target():
    # with-statement without 'as' clause
    src = "with ctx_mgr():\n    do_work()\n"
    result = _find_free_vars(src)
    assert "ctx_mgr" in result
    assert "do_work" in result


def test_find_free_vars_except_handler_name():
    # except-handler name is locally bound for the handler body
    src = "try:\n    risky()\nexcept ValueError as exc:\n    handle(exc)\n"
    result = _find_free_vars(src)
    assert "exc" not in result  # locally bound by except clause
    assert "risky" in result
    assert "handle" in result


def test_find_free_vars_except_no_name():
    # bare except without 'as' binding
    src = "try:\n    risky()\nexcept ValueError:\n    pass\n"
    result = _find_free_vars(src)
    assert "risky" in result


def test_find_free_vars_listcomp():
    # list comprehension: loop var is locally scoped
    src = "result = [x * 2 for x in data]\n"
    result = _find_free_vars(src)
    assert "x" not in result  # comprehension target, locally scoped
    assert "data" in result


def test_find_free_vars_listcomp_with_filter():
    # comprehension with 'if' guard: threshold must come from outside
    src = "result = [x for x in data if x > threshold]\n"
    result = _find_free_vars(src)
    assert "x" not in result
    assert "data" in result
    assert "threshold" in result


def test_find_free_vars_dictcomp():
    # dict comprehension: both key and value expressions are walked
    src = "result = {k: v for k, v in pairs}\n"
    result = _find_free_vars(src)
    assert "k" not in result  # tuple target of comprehension
    assert "v" not in result
    assert "pairs" in result


def test_find_free_vars_tuple_for_target():
    # tuple-unpacking for target: both names locally scoped
    src = "for a, b in pairs:\n    use(a, b)\n"
    result = _find_free_vars(src)
    assert "a" not in result
    assert "b" not in result
    assert "pairs" in result


def test_find_free_vars_subscript_assign_target():
    # subscript assignment target (e.g. data[0] = 1): _target_names returns {}
    # so nothing is added to definitely_defined, but data is loaded
    src = "data[0] = 1\n"
    result = _find_free_vars(src)
    assert "data" in result  # data is loaded as the subscript base


def test_find_free_vars_annassign_with_value():
    # annotated assignment with value: name is definitely defined afterwards
    src = "x: int = 5\ny = x + 1\n"
    result = _find_free_vars(src)
    assert "x" not in result
    assert "y" not in result


def test_find_free_vars_annassign_no_value():
    # annotation without assignment: x is NOT definitely defined
    src = "x: int\ny = x + 1\n"
    result = _find_free_vars(src)
    assert "x" in result  # not assigned, so it is free


def test_find_free_vars_annassign_non_name_target():
    # annotated assignment where target is not a plain Name
    src = "obj.attr: int = 5\n"
    result = _find_free_vars(src)
    assert "obj" in result  # obj is loaded to set the attribute


def test_find_free_vars_conditional_store_is_free():
    # variables only assigned inside a conditional block remain free
    src = "for i in xs:\n    result = f(i)\nprint(result)\n"
    result = _find_free_vars(src)
    assert "result" in result  # conditionally assigned → still free after loop


# ---------------------------------------------------------------------------
# _stmts_source
# ---------------------------------------------------------------------------


def test_stmts_source_basic():
    src = "def foo():\n    x = 1\n    y = 2\n    z = 3\n"
    stmts, positions, lines = _parse_func(src)
    result = _stmts_source(stmts[:2], lines, positions)
    assert "x = 1" in result
    assert "y = 2" in result
    assert "z = 3" not in result


def test_stmts_source_empty():
    src = "def foo():\n    x = 1\n"
    _, positions, lines = _parse_func(src)
    assert _stmts_source([], lines, positions) == ""


def test_stmts_source_dedented():
    src = "def foo():\n    x = 1\n    y = 2\n"
    stmts, positions, lines = _parse_func(src)
    result = _stmts_source(stmts, lines, positions)
    # Should be dedented (no leading 4-space indent)
    assert result.startswith("x = 1") or result.startswith("x = 1\n")


# ---------------------------------------------------------------------------
# _head_effective_lines
# ---------------------------------------------------------------------------


def test_head_effective_lines_no_docstring():
    src = "def foo():\n    x = 1\n    y = 2\n    z = 3\n"
    stmts, positions, lines = _parse_func(src)
    # split_idx=2: head=[x,y], last=y at line 3, first=x at line 2 → 3-2+2=3
    result = _head_effective_lines(stmts, 2, positions, False)
    assert result == 3


def test_head_effective_lines_with_docstring_normal():
    src = 'def foo():\n    """doc"""\n    x = 1\n    y = 2\n    z = 3\n'
    stmts, positions, lines = _parse_func(src)
    # split_idx=3: head=[doc, x, y], first_non_doc=x at line 3, last=y at line 4
    # 4-3+2=3
    result = _head_effective_lines(stmts, 3, positions, True)
    assert result == 3


def test_head_effective_lines_only_docstring_in_head():
    # split_idx=1 with docstring: first_non_doc_idx=1 >= split_idx=1 → returns 1
    src = 'def foo():\n    """doc"""\n    x = 1\n    y = 2\n'
    stmts, positions, lines = _parse_func(src)
    result = _head_effective_lines(stmts, 1, positions, True)
    assert result == 1


# ---------------------------------------------------------------------------
# _find_valid_splits
# ---------------------------------------------------------------------------


def test_find_valid_splits_all_valid():
    src = "def foo():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n"
    stmts, positions, lines = _parse_func(src)
    # With very loose limits, all splits should be valid
    result = _find_valid_splits(
        stmts, positions, lines, max_lines=1000, max_complexity=1000
    )
    assert len(result) > 0
    # Ordered latest first
    assert result == sorted(result, reverse=True)


def test_find_valid_splits_none_valid():
    # max_lines=1 means even a 1-stmt head (+ return call = 2 lines) is invalid
    src = "def foo():\n    a = 1\n    b = 2\n    c = 3\n"
    stmts, positions, lines = _parse_func(src)
    result = _find_valid_splits(
        stmts, positions, lines, max_lines=1, max_complexity=1000
    )
    assert result == []


def test_find_valid_splits_stops_at_max_candidates():
    # 7 statements → iterates from 6 down, stops after 5 valid candidates
    src = "def foo():\n" + "".join(f"    a{i} = {i}\n" for i in range(7))
    stmts, positions, lines = _parse_func(src)
    result = _find_valid_splits(
        stmts, positions, lines, max_lines=1000, max_complexity=1000
    )
    assert len(result) == 5


def test_find_valid_splits_fewer_than_max():
    # 4 statements → at most 3 valid splits (indices 3, 2, 1)
    src = "def foo():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n"
    stmts, positions, lines = _parse_func(src)
    result = _find_valid_splits(
        stmts, positions, lines, max_lines=1000, max_complexity=1000
    )
    assert 1 <= len(result) <= 3


def test_find_valid_splits_empty_body():
    # Should not crash with an empty list (though normally not called)
    stmts = []
    result = _find_valid_splits(stmts, {}, [], max_lines=1000, max_complexity=1000)
    assert result == []


def test_find_valid_splits_complexity_filter():
    # A head with high complexity is filtered out; only the simple head passes
    # Source: simple stmt first, then complex nested-if block, then trivial last stmt
    src = textwrap.dedent(
        """\
        def foo(x):
            y = 1
            if x > 0:
                if x > 1:
                    if x > 2:
                        pass
            z = 2
    """
    )
    stmts, positions, lines = _parse_func(src)
    # stmts = [y=1, if_block, z=2]
    # i=2: head=[y=1, if_block], complexity of nested ifs > 2 → filtered
    # i=1: head=[y=1], complexity=1 ≤ 2 → valid
    result = _find_valid_splits(
        stmts, positions, lines, max_lines=1000, max_complexity=2
    )
    assert result == [1]


# ---------------------------------------------------------------------------
# _choose_best_split
# ---------------------------------------------------------------------------


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
    split_idx, params = _choose_best_split(
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
    split_idx, params = _choose_best_split(stmts, valid_splits, lines, positions, [])
    # split_idx=1 has 1 free var (a) vs split_idx=2 has 2 free vars (a, b)
    assert split_idx == 1
    assert params == ["a"]


def test_choose_best_split_single_candidate():
    src = "def foo():\n    x = 1\n    y = 2\n"
    stmts, positions, lines = _parse_func(src)
    split_idx, params = _choose_best_split(stmts, [1], lines, positions, [])
    assert split_idx == 1


# ---------------------------------------------------------------------------
# _generate_helper_source
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _generate_call
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _has_yield
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _run_with_timeout
# ---------------------------------------------------------------------------


def test_run_with_timeout_success():
    result = _run_with_timeout(lambda x: x * 2, 5, 21)
    assert result == 42


def test_run_with_timeout_exceeds():
    import time

    with pytest.raises(_ApiTimeout):
        _run_with_timeout(lambda: time.sleep(10), timeout=0.05)


def test_run_with_timeout_propagates_exception():
    def _raise():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        _run_with_timeout(_raise, 5)


# ---------------------------------------------------------------------------
# _func_in_changed_range / _extract_func_source
# ---------------------------------------------------------------------------


def _make_func_info(start, end):
    """Create a minimal _FuncInfo for range tests."""
    mock_node = MagicMock()
    return _FuncInfo(
        node=mock_node,
        start_line=start,
        end_line=end,
        class_name=None,
        indent="",
        original_params=[],
    )


def test_func_in_changed_range_overlaps():
    fi = _make_func_info(5, 15)
    assert _func_in_changed_range(fi, [(1, 10)]) is True


def test_func_in_changed_range_no_overlap():
    fi = _make_func_info(5, 10)
    assert _func_in_changed_range(fi, [(20, 30)]) is False


def test_func_in_changed_range_adjacent():
    fi = _make_func_info(5, 10)
    assert _func_in_changed_range(fi, [(10, 20)]) is True


def test_extract_func_source():
    lines = ["line1\n", "line2\n", "line3\n", "line4\n"]
    fi = _make_func_info(2, 3)
    result = _extract_func_source(fi, lines)
    assert result == "line2\nline3\n"


# ---------------------------------------------------------------------------
# _FunctionCollector
# ---------------------------------------------------------------------------


def test_function_collector_module_level():
    src = "def foo():\n    x = 1\n"
    tree = cst.parse_module(src)
    wrapper = MetadataWrapper(tree)
    collector = _FunctionCollector()
    wrapper.visit(collector)
    assert len(collector.functions) == 1
    assert collector.functions[0].node.name.value == "foo"
    assert collector.functions[0].class_name is None
    assert collector.functions[0].indent == ""


def test_function_collector_class_method():
    src = textwrap.dedent(
        """\
        class Foo:
            def bar(self):
                pass
    """
    )
    tree = cst.parse_module(src)
    wrapper = MetadataWrapper(tree)
    collector = _FunctionCollector()
    wrapper.visit(collector)
    assert len(collector.functions) == 1
    assert collector.functions[0].class_name == "Foo"
    assert collector.functions[0].indent == "    "


def test_function_collector_skips_async():
    src = "async def foo():\n    pass\n"
    tree = cst.parse_module(src)
    wrapper = MetadataWrapper(tree)
    collector = _FunctionCollector()
    wrapper.visit(collector)
    assert len(collector.functions) == 0


def test_function_collector_skips_generator():
    src = "def gen():\n    yield 1\n"
    tree = cst.parse_module(src)
    wrapper = MetadataWrapper(tree)
    collector = _FunctionCollector()
    wrapper.visit(collector)
    assert len(collector.functions) == 0


def test_function_collector_skips_nested_functions():
    # Only module-level and class methods are collected
    src = textwrap.dedent(
        """\
        def outer():
            def inner():
                pass
            return inner
    """
    )
    tree = cst.parse_module(src)
    wrapper = MetadataWrapper(tree)
    collector = _FunctionCollector()
    wrapper.visit(collector)
    # Only outer is collected (inner is nested inside outer)
    assert len(collector.functions) == 1
    assert collector.functions[0].node.name.value == "outer"


def test_function_collector_captures_params():
    src = "def foo(a, b, c):\n    pass\n"
    tree = cst.parse_module(src)
    wrapper = MetadataWrapper(tree)
    collector = _FunctionCollector()
    wrapper.visit(collector)
    assert collector.functions[0].original_params == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# _llm_name_helpers
# ---------------------------------------------------------------------------


def _make_task(func_name, params=None, tail_source="return 0\n"):
    """Create a minimal _SplitTask for testing _llm_name_helpers."""
    mock_node = MagicMock()
    mock_node.name.value = func_name
    fi = _FuncInfo(
        node=mock_node,
        start_line=1,
        end_line=5,
        class_name=None,
        indent="",
        original_params=[],
    )
    return _SplitTask(fi, 1, params or [], tail_source=tail_source)


@patch("crispen.llm_client.anthropic")
def test_llm_name_helpers_success(mock_anthropic):
    mock_response = _make_mock_response(["process_tail"])
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    tasks = [_make_task("my_func")]
    client = mock_anthropic.Anthropic.return_value
    result = _llm_name_helpers(client, "claude-sonnet-4-6", "anthropic", tasks)
    assert result == ["process_tail"]


@patch("crispen.llm_client.anthropic")
def test_llm_name_helpers_result_none(mock_anthropic):
    # LLM returns no tool use block
    mock_response = MagicMock()
    mock_response.content = []
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    tasks = [_make_task("my_func")]
    client = mock_anthropic.Anthropic.return_value
    result = _llm_name_helpers(client, "claude-sonnet-4-6", "anthropic", tasks)
    # Falls back to "my_func_helper"
    assert result == ["my_func_helper"]


@patch("crispen.llm_client.anthropic")
def test_llm_name_helpers_no_names_key(mock_anthropic):
    # LLM returns tool use but without "names" key
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "name_helper_functions"
    mock_block.input = {"something_else": []}
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    tasks = [_make_task("my_func")]
    client = mock_anthropic.Anthropic.return_value
    result = _llm_name_helpers(client, "claude-sonnet-4-6", "anthropic", tasks)
    assert result == ["my_func_helper"]


@patch("crispen.llm_client.anthropic")
def test_llm_name_helpers_strips_leading_underscore(mock_anthropic):
    mock_response = _make_mock_response(["__private_name"])
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    tasks = [_make_task("foo")]
    client = mock_anthropic.Anthropic.return_value
    result = _llm_name_helpers(client, "claude-sonnet-4-6", "anthropic", tasks)
    assert result == ["private_name"]


@patch("crispen.llm_client.anthropic")
def test_llm_name_helpers_all_underscores_uses_helper(mock_anthropic):
    mock_response = _make_mock_response(["___"])
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    tasks = [_make_task("foo")]
    client = mock_anthropic.Anthropic.return_value
    result = _llm_name_helpers(client, "claude-sonnet-4-6", "anthropic", tasks)
    assert result == ["helper"]


@patch("crispen.llm_client.anthropic")
def test_llm_name_helpers_bad_item_skipped(mock_anthropic):
    # One item has a TypeError (e.g. name is not a string)
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "name_helper_functions"
    mock_block.input = {
        "names": [{"id": "0", "name": None}]  # None.lstrip() raises AttributeError
    }
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    tasks = [_make_task("foo")]
    client = mock_anthropic.Anthropic.return_value
    result = _llm_name_helpers(client, "claude-sonnet-4-6", "anthropic", tasks)
    # Falls back to "foo_helper" because item had AttributeError
    assert result == ["foo_helper"]


@patch("crispen.llm_client.anthropic")
def test_llm_name_helpers_with_class_name(mock_anthropic):
    mock_response = _make_mock_response(["process"])
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

    mock_node = MagicMock()
    mock_node.name.value = "method"
    fi = _FuncInfo(
        node=mock_node,
        start_line=1,
        end_line=5,
        class_name="MyClass",
        indent="    ",
        original_params=[],
    )
    task = _SplitTask(fi, 1, [], tail_source="return 0\n")
    client = mock_anthropic.Anthropic.return_value
    result = _llm_name_helpers(client, "claude-sonnet-4-6", "anthropic", [task])
    assert result == ["process"]


# ---------------------------------------------------------------------------
# FunctionSplitter — integration tests
# ---------------------------------------------------------------------------


def _make_long_func(n_stmts: int, func_name: str = "long_func") -> str:
    """Build a function with n_stmts independent assignments."""
    lines = [f"def {func_name}():\n"]
    for i in range(n_stmts):
        lines.append(f"    a{i} = {i}\n")
    lines.append("    return 0\n")
    return "".join(lines)


def test_function_splitter_under_limits_no_op():
    # A small function should not be split
    src = "def small():\n    x = 1\n    return x\n"
    splitter = FunctionSplitter([(1, 10)], source=src, verbose=False)
    assert splitter.get_rewritten_source() is None


def test_function_splitter_parse_error_no_crash():
    # Invalid source should not crash
    splitter = FunctionSplitter([(1, 10)], source="def f(\n  !!invalid", verbose=False)
    assert splitter.get_rewritten_source() is None


def test_function_splitter_out_of_range_no_op():
    # Function exists but is outside changed ranges
    src = _make_long_func(80)
    splitter = FunctionSplitter([(200, 300)], source=src, verbose=False, max_lines=10)
    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_over_line_limit(mock_anthropic):
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["process_tail"])
    )
    src = _make_long_func(80)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],
            source=src,
            verbose=False,
            max_lines=50,
            max_complexity=1000,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    assert "_process_tail" in result
    assert "return _process_tail(" in result
    assert len(splitter.changes_made) >= 1


@patch("crispen.llm_client.anthropic")
def test_function_splitter_over_complexity_limit(mock_anthropic):
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["handle_tail"])
    )
    # Build a function with high cyclomatic complexity
    lines = ["def complex_func(x):\n"]
    for i in range(15):
        lines.append(f"    if x > {i}:\n        pass\n")
    lines.append("    return 0\n")
    src = "".join(lines)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],
            source=src,
            verbose=False,
            max_lines=1000,  # no line limit
            max_complexity=5,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    assert "_handle_tail" in result


@patch("crispen.llm_client.anthropic")
def test_function_splitter_async_skipped(mock_anthropic):
    # Async functions should not be split
    src = (
        "async def foo():\n"
        + "".join(f"    a{i} = {i}\n" for i in range(80))
        + "    return 0\n"
    )

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=10
        )

    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_generator_skipped(mock_anthropic):
    # Generator functions should not be split
    src = (
        "def gen():\n"
        + "".join(f"    a{i} = {i}\n" for i in range(80))
        + "    yield 0\n"
    )

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=10
        )

    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_llm_fallback_on_api_error(mock_anthropic):
    # API key not set → get_api_key raises CrispenAPIError → fallback names used
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["tail"])
    )
    src = _make_long_func(60, "my_func")

    # No ANTHROPIC_API_KEY → get_api_key raises → fallback to "my_func_helper"
    with patch.dict("os.environ", {}, clear=True):
        # Remove any existing API key
        import os

        os.environ.pop("ANTHROPIC_API_KEY", None)
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=30, max_complexity=1000
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    # Fallback name used: "my_func_helper"
    assert "_my_func_helper" in result


@patch("crispen.llm_client.anthropic")
def test_function_splitter_recursive_split(mock_anthropic):
    # With small max_lines and broad changed_ranges, triggers multiple iterations
    # First call names helper for first function, second call for helper
    mock_anthropic.Anthropic.return_value.messages.create.side_effect = [
        _make_mock_response(["part1"]),
        _make_mock_response(["part2"]),
        _make_mock_response(["part3"]),
    ]

    # 13 body statements → with max_lines=5, needs multiple splits
    src = _make_long_func(13, "func")

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],  # broad range covers all helpers too
            source=src,
            verbose=False,
            max_lines=5,
            max_complexity=1000,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    # Multiple splits occurred
    assert len(splitter.changes_made) >= 2


@patch("crispen.llm_client.anthropic")
def test_function_splitter_syntax_error_in_output_is_skipped(mock_anthropic):
    # If the assembled edit is invalid Python, the change is not applied
    # We simulate this by making _generate_call return something invalid
    # Instead, test the path via a function with 1-stmt body (no valid split)
    src = "def foo():\n    x = 1\n"  # only 1 stmt → can't split
    splitter = FunctionSplitter(
        [(1, 10)], source=src, verbose=False, max_lines=0, max_complexity=0
    )
    # Either no valid splits found OR already under limit (0 means everything is over)
    # body lines=1 > 0=max_lines → tries to split but len(body_stmts)=1 < 2 → skip
    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_no_valid_split_skipped(mock_anthropic):
    # max_lines=1 → even a head with 1 stmt (+return call=2) > max_lines=1
    # So no valid splits → no change
    src = _make_long_func(5, "foo")

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=1, max_complexity=1000
        )

    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_with_helper_docstrings(mock_anthropic):
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["process"])
    )
    src = _make_long_func(80)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],
            source=src,
            verbose=False,
            max_lines=50,
            helper_docstrings=True,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    assert '"""' in result


@patch("crispen.llm_client.anthropic")
def test_function_splitter_class_method(mock_anthropic):
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["tail_work"])
    )
    lines = ["class Foo:\n", "    def method(self):\n"]
    for i in range(80):
        lines.append(f"        a{i} = {i}\n")
    lines.append("        return 0\n")
    src = "".join(lines)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],
            source=src,
            verbose=False,
            max_lines=50,
            max_complexity=1000,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    # Class methods use staticmethod and ClassName._ call
    assert "@staticmethod" in result
    assert "Foo._tail_work(" in result


@patch("crispen.llm_client.anthropic")
def test_function_splitter_llm_timeout_fallback(mock_anthropic):
    # LLM call times out → fallback names
    import time

    def _slow(*args, **kwargs):
        time.sleep(10)

    mock_anthropic.Anthropic.return_value.messages.create.side_effect = _slow
    src = _make_long_func(60, "slow_func")

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("crispen.refactors.function_splitter._API_HARD_TIMEOUT", 0.05):
            splitter = FunctionSplitter(
                [(1, 1000)],
                source=src,
                verbose=False,
                max_lines=30,
            )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    # Fallback name "slow_func_helper" used
    assert "_slow_func_helper" in result


# ---------------------------------------------------------------------------
# FunctionSplitter — additional branch coverage tests
# ---------------------------------------------------------------------------


def test_function_splitter_empty_source():
    """FunctionSplitter created with no source does nothing."""
    splitter = FunctionSplitter([(1, 10)])
    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_max_iterations_loop_exhausted(mock_anthropic):
    """Loop runs to completion (no break) when max iterations reached."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["helper"])
    )
    src = _make_long_func(80, "foo")

    # Patch _MAX_SPLIT_ITERATIONS to 1 → loop runs exactly once without breaking
    # (break only occurs at START of next iteration when tasks=[])
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("crispen.refactors.function_splitter._MAX_SPLIT_ITERATIONS", 1):
            splitter = FunctionSplitter(
                [(1, 1000)], source=src, verbose=False, max_lines=50
            )

    result = splitter.get_rewritten_source()
    assert result is not None
    assert len(splitter.changes_made) == 1


@patch("crispen.llm_client.anthropic")
def test_function_splitter_syntax_error_in_generated_output(mock_anthropic):
    """If assembled output fails compile(), the change is not applied."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["helper"])
    )
    src = _make_long_func(80, "foo")

    import builtins as _builtins

    orig_compile = _builtins.compile

    def _selective_compile(source, filename, mode, *args, **kwargs):
        if filename == "<string>":
            raise SyntaxError("mocked error for test")
        return orig_compile(source, filename, mode, *args, **kwargs)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("builtins.compile", side_effect=_selective_compile):
            splitter = FunctionSplitter(
                [(1, 1000)], source=src, verbose=False, max_lines=50
            )

    assert splitter.get_rewritten_source() is None


def test_find_free_vars_del_context():
    """del statement adds name to stores (else branch for non-Load contexts)."""
    src = "del my_var\n"
    result = _find_free_vars(src)
    assert "my_var" not in result


# ---------------------------------------------------------------------------
# Engine integration: FunctionSplitter branch is exercised
# ---------------------------------------------------------------------------


def test_engine_includes_function_splitter_no_op(tmp_path):
    """FunctionSplitter is in _REFACTORS and runs without error for simple files."""
    from crispen.engine import run_engine
    from crispen.config import CrispenConfig

    py_file = tmp_path / "sample.py"
    py_file.write_text("def foo():\n    return 1\n")
    config = CrispenConfig(max_function_length=75, max_complexity=10)
    msgs = list(run_engine({str(py_file): [(1, 2)]}, verbose=False, config=config))
    # No split needed — no messages expected (or just no errors)
    assert all("FunctionSplitter" not in m for m in msgs)
