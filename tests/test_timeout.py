from __future__ import annotations
import pytest
from crispen.refactors.function_splitter import _run_with_timeout


def test_run_with_timeout_propagates_exception():
    def _raise():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        _run_with_timeout(_raise, 5)
