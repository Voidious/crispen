from __future__ import annotations
import textwrap
from crispen.refactors.function_splitter import _find_free_vars


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
