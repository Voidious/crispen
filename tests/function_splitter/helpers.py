from __future__ import annotations
from unittest.mock import MagicMock
from crispen.refactors.function_splitter import _FuncInfo, _SplitTask


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


def _make_task(func_name, params=None, tail_source="return 0\n"):
    """Create a minimal _SplitTask for testing _llm_name_helpers."""
    mock_node = MagicMock()
    mock_node.name.value = func_name
    fi = _FuncInfo(
        node=mock_node,
        start_line=1,
        end_line=5,
        class_name=None,
        indent="",
        original_params=[],
    )
    return _SplitTask(fi, 1, params or [], tail_source=tail_source)


def _make_long_func(n_stmts: int, func_name: str = "long_func") -> str:
    """Build a function with n_stmts independent assignments."""
    lines = [f"def {func_name}():\n"]
    for i in range(n_stmts):
        lines.append(f"    a{i} = {i}\n")
    lines.append("    return 0\n")
    return "".join(lines)
