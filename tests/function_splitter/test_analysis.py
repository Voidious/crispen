from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import (
    _choose_best_split,
    _count_body_lines,
    _find_free_vars,
    _has_new_undefined_names,
    _head_effective_lines,
    _is_docstring_stmt,
    _module_global_names,
    _stmts_source,
)
from .test_splitter import _parse_func
import textwrap
import libcst as cst


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


def test_is_docstring_compound_stmt():
    # A compound statement (If) is not a SimpleStatementLine
    src = "def f():\n    if True:\n        pass\n"
    stmt = cst.parse_module(src).body[0].body.body[0]
    assert _is_docstring_stmt(stmt) is False


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


def test_find_free_vars_for_body_sequential():
    # a variable assigned then used in the same for-body iteration is not free
    src = "for alias in names:\n    name = alias.asname\n    result.add(name)\n"
    result = _find_free_vars(src)
    assert "name" not in result  # assigned before used in same loop body
    assert "names" in result
    assert "result" in result


def test_find_free_vars_if_branch():
    # if-body assignments do not propagate to after the if block
    src = "if cond:\n    x = 1\nelse:\n    y = 2\nz = x + y\n"
    result = _find_free_vars(src)
    assert "cond" in result
    assert "x" in result  # only conditionally defined in if body
    assert "y" in result  # only conditionally defined in else body


def test_find_free_vars_while_loop():
    # while condition is free; while-else is walked
    src = "while running:\n    do_work()\nelse:\n    finalize()\n"
    result = _find_free_vars(src)
    assert "running" in result
    assert "do_work" in result
    assert "finalize" in result


def test_find_free_vars_try_propagates():
    # variables assigned in a try body propagate to code after the try block
    src = textwrap.dedent(
        """\
        try:
            lineno = compute()
        except ValueError:
            return
        use(lineno)
    """
    )
    result = _find_free_vars(src)
    assert "lineno" not in result  # defined in try body, propagated outward
    assert "compute" in result
    assert "use" in result


def test_find_free_vars_try_orelse():
    # try-else clause is walked with the try-body scope (x is defined there)
    src = textwrap.dedent(
        """\
        try:
            x = compute()
        except ValueError:
            return
        else:
            use(x)
    """
    )
    result = _find_free_vars(src)
    assert "x" not in result  # defined in try body, visible in else clause
    assert "use" in result
    assert "compute" in result


def test_find_free_vars_try_finally():
    # try with finally and no handlers: handlers loop is empty
    src = "try:\n    x = compute()\nfinally:\n    cleanup()\n"
    result = _find_free_vars(src)
    assert "compute" in result
    assert "cleanup" in result
    assert "x" not in result  # defined in try body, propagated


def test_find_free_vars_bare_except():
    # bare 'except:' has node.type = None (covers the None branch)
    src = "try:\n    risky()\nexcept:\n    pass\n"
    result = _find_free_vars(src)
    assert "risky" in result


def test_find_free_vars_lambda_param_not_free():
    # lambda parameter must not appear as a free variable
    src = "result = sorted(tasks, key=lambda t: t.name)\n"
    result = _find_free_vars(src)
    assert "t" not in result
    assert "tasks" in result


def test_find_free_vars_lambda_vararg_not_free():
    # *args in lambda body — args is the vararg, not free
    src = "f = lambda *args: list(args)\n"
    result = _find_free_vars(src)
    assert "args" not in result


def test_find_free_vars_lambda_kwarg_not_free():
    # **kw in lambda body — kw is the kwarg, not free
    src = "f = lambda **kw: kw\n"
    result = _find_free_vars(src)
    assert "kw" not in result


def test_find_free_vars_lambda_default_outer_scope():
    # Default values are evaluated in the enclosing scope, not the lambda scope.
    src = "f = lambda x=outer_val: x\n"
    result = _find_free_vars(src)
    assert "outer_val" in result  # evaluated in outer scope → free
    assert "x" not in result  # lambda param → not free


def test_find_free_vars_lambda_kw_default_none_entry():
    # keyword-only param without a default: kw_defaults has a None entry
    # lambda *, x, y=outer_val: x+y → kw_defaults=[None, outer_val_node]
    src = "f = lambda *, x, y=outer_val: x + y\n"
    result = _find_free_vars(src)
    assert "x" not in result  # kwonly param → not free
    assert "y" not in result  # kwonly param → not free
    assert "outer_val" in result  # kw_default evaluated in outer scope → free


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


def test_find_free_vars_del_context():
    """del statement adds name to stores (else branch for non-Load contexts)."""
    src = "del my_var\n"
    result = _find_free_vars(src)
    assert "my_var" not in result


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
