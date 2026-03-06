from __future__ import annotations
from crispen.file_limiter.runner import _strip_imports_by_line


def test_strip_imports_no_imports():
    src = "def foo():\n    return 1\n"
    assert _strip_imports_by_line(src) == src


def test_strip_imports_single_line():
    src = "import os\nx = 1\n"
    assert _strip_imports_by_line(src) == "x = 1\n"


def test_strip_imports_multiline():
    src = "from os import (\n    path,\n    getcwd,\n)\nx = 1\n"
    assert _strip_imports_by_line(src) == "x = 1\n"


def test_strip_imports_inner_import():
    # Imports inside a function body are also stripped.
    src = "def foo():\n    import os\n    return os.getcwd()\n"
    assert _strip_imports_by_line(src) == "def foo():\n    return os.getcwd()\n"


def test_strip_imports_syntax_error_returns_unchanged():
    src = "def foo(:\n    pass\n"
    assert _strip_imports_by_line(src) == src
