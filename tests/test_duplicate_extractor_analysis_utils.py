import textwrap
from crispen.refactors.duplicate_extractor import (
    _extract_defined_names,
    _names_assigned_in,
    _missing_free_vars,
)


def test_missing_free_vars_syntax_error_in_block_returns_empty():
    assert (
        _missing_free_vars("not valid python!!!", ["x = 1\n"], "def f(): pass\n", "")
        == set()
    )


def test_missing_free_vars_syntax_error_in_replacement_returns_empty():
    source = "def run():\n    a = 1\n"
    assert (
        _missing_free_vars("x = a\n", ["not valid!!!\n"], "def f(): pass\n", source)
        == set()
    )


def test_missing_free_vars_syntax_error_in_source_returns_empty():
    assert (
        _missing_free_vars("x = a\n", ["y = a\n"], "def f(a): pass\n", "not valid!!!")
        == set()
    )


def test_missing_free_vars_empty_block_returns_empty():
    # A block with no reads has no free vars → nothing can be missing.
    source = "def run():\n    x = 1\n"
    block_src = "    x = 1\n"
    call_src = "    _h()\n"
    helper_src = "def _h():\n    x = 1\n"
    assert _missing_free_vars(block_src, [call_src], helper_src, source) == set()


def test_missing_free_vars_function_parameter_is_caught():
    # A function parameter that's free in the block must appear in the
    # replacement — parameters are local to the function and cannot be
    # accessed by a helper without being passed as an argument.
    source = "def run(verbose):\n    msg = verbose\n"
    block_src = "    msg = verbose\n"
    call_src = "    msg = _h()\n"
    helper_src = "def _h():\n    pass\n"
    assert "verbose" in _missing_free_vars(block_src, [call_src], helper_src, source)


def test_names_assigned_in_simple():
    assert _names_assigned_in("x = 1\n") == {"x"}


def test_names_assigned_in_tuple_unpack():
    assert _names_assigned_in("x, y = f()\n") == {"x", "y"}


def test_names_assigned_in_augassign():
    assert _names_assigned_in("x += 1\n") == {"x"}


def test_names_assigned_in_no_assign():
    assert _names_assigned_in("f()\n") == set()


def test_names_assigned_in_syntax_error():
    assert _names_assigned_in("def (\n") == set()


def test_extract_defined_names_basic():
    source = textwrap.dedent(
        """\
        def foo():
            pass

        async def bar():
            pass

        class Baz:
            pass
        """
    )
    assert _extract_defined_names(source) == {"foo", "bar", "Baz"}


def test_extract_defined_names_syntax_error():
    assert _extract_defined_names("def (\n") == set()
