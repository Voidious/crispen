from __future__ import annotations
from unittest.mock import MagicMock
from crispen.refactors.function_splitter import _FuncInfo


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
