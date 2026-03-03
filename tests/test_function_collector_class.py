from tests.test_function_helpers import _collect_functions


def test_function_collector_class_level():
    source = "class C:\n    def method(self):\n        pass\n"
    funcs = _collect_functions(source)
    assert len(funcs) == 1
    assert funcs[0].name == "method"
    assert funcs[0].scope == "C"
    assert funcs[0].body_stmt_count == 1
    assert funcs[0].params == ["self"]
