from __future__ import annotations
import textwrap
from crispen.refactors.function_splitter import _find_free_vars


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
