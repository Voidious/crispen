from __future__ import annotations
from crispen.file_limiter.code_gen import _strip_top_level_import_lines


def test_strip_top_level_import_lines_removes_imports():
    src = "import os\nfrom typing import List\n\n_CONST = 1\n"
    result = _strip_top_level_import_lines(src)
    assert "import os" not in result
    assert "from typing import List" not in result
    assert "_CONST = 1" in result


def test_strip_top_level_import_lines_no_imports():
    src = "_CONST = 1\n"
    assert _strip_top_level_import_lines(src) == src


def test_strip_top_level_import_lines_syntax_error():
    src = "def (\n"
    assert _strip_top_level_import_lines(src) == src
