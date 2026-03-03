from tests.test_function_helpers import _collect_functions


def test_function_collector_collects_stmt_count():
    source = "def foo():\n    pass\n"
    funcs = _collect_functions(source)
    assert funcs[0].body_stmt_count == 1
