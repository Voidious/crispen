from __future__ import annotations
from crispen.file_limiter.code_gen import (
    _extract_module_docstring,
    _strip_module_docstring,
)


def test_extract_module_docstring_present():
    src = '"""My module."""\n\nimport os\n'
    assert _extract_module_docstring(src) == '"""My module."""'


def test_extract_module_docstring_absent():
    src = "import os\n\ndef foo():\n    pass\n"
    assert _extract_module_docstring(src) is None


def test_extract_module_docstring_syntax_error():
    assert _extract_module_docstring("def (\n") is None


def test_extract_module_docstring_non_string_expr():
    # First statement is an expression but not a string constant.
    src = "1 + 1\n\ndef foo():\n    pass\n"
    assert _extract_module_docstring(src) is None


def test_strip_module_docstring_removes_docstring():
    src = '"""My module."""\n\n_CONST = 1\n'
    result = _strip_module_docstring(src)
    assert '"""My module."""' not in result
    assert "_CONST = 1" in result


def test_strip_module_docstring_no_docstring():
    src = "_CONST = 1\n"
    assert _strip_module_docstring(src) == src


def test_strip_module_docstring_syntax_error():
    src = "def (\n"
    assert _strip_module_docstring(src) == src
