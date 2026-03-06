from __future__ import annotations
from unittest.mock import MagicMock
from crispen.refactors.function_splitter import (
    _FuncInfo,
    _extract_func_source,
    _func_in_changed_range,
)


def _make_func_info(start, end):
    """Create a minimal _FuncInfo for range tests."""
    mock_node = MagicMock()
    return _FuncInfo(
        node=mock_node,
        start_line=start,
        end_line=end,
        class_name=None,
        indent="",
        original_params=[],
    )


def test_func_in_changed_range_overlaps():
    fi = _make_func_info(5, 15)
    assert _func_in_changed_range(fi, [(1, 10)]) is True


def test_func_in_changed_range_no_overlap():
    fi = _make_func_info(5, 10)
    assert _func_in_changed_range(fi, [(20, 30)]) is False


def test_func_in_changed_range_adjacent():
    fi = _make_func_info(5, 10)
    assert _func_in_changed_range(fi, [(10, 20)]) is True


def test_extract_func_source():
    lines = ["line1\n", "line2\n", "line3\n", "line4\n"]
    fi = _make_func_info(2, 3)
    result = _extract_func_source(fi, lines)
    assert result == "line2\nline3\n"
