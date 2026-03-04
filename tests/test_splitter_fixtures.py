from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import FunctionSplitter


def _make_splitter_with_env(source: str) -> FunctionSplitter:
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        return FunctionSplitter([(1, 1000)], source=source, verbose=False, max_lines=50)
