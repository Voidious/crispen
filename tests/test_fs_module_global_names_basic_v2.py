from __future__ import annotations
from crispen.refactors.function_splitter import _module_global_names


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
