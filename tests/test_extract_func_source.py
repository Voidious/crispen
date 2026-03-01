from __future__ import annotations
from crispen.refactors.function_splitter import _extract_func_source
from .func_info_utilities import _make_func_info


def test_extract_func_source():
    lines = ["line1\n", "line2\n", "line3\n", "line4\n"]
    fi = _make_func_info(2, 3)
    result = _extract_func_source(fi, lines)
    assert result == "line2\nline3\n"
