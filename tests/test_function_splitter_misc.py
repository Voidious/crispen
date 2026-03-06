from __future__ import annotations
from crispen.refactors.function_splitter import FunctionSplitter


def test_function_splitter_empty_source():
    """FunctionSplitter created with no source does nothing."""
    splitter = FunctionSplitter([(1, 10)])
    assert splitter.get_rewritten_source() is None
