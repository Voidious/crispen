from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import FunctionSplitter


def _run_function_splitter_with_patch(src):
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=10
        )

    assert splitter.get_rewritten_source() is None


def _create_function_splitter_with_env(src, max_lines=50):
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=max_lines
        )
    return splitter
