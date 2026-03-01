from __future__ import annotations
import pytest
from crispen.refactors.function_splitter import _ApiTimeout, _run_with_timeout


def test_run_with_timeout_success():
    result = _run_with_timeout(lambda x: x * 2, 5, 21)
    assert result == 42


def test_run_with_timeout_exceeds():
    import time

    with pytest.raises(_ApiTimeout):
        _run_with_timeout(lambda: time.sleep(10), timeout=0.05)


def test_run_with_timeout_propagates_exception():
    def _raise():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        _run_with_timeout(_raise, 5)
