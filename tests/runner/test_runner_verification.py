from __future__ import annotations
from unittest.mock import patch
from crispen.file_limiter.code_gen import SplitResult
from crispen.file_limiter.runner import run_file_limiter
from .helpers import (
    _CONFIG,
    _PATCH_ADVISE,
    _PATCH_CLASSIFY,
    _PATCH_GEN,
    _make_classified,
    _make_entity,
    _plan_with,
)


@patch(_PATCH_GEN)
@patch(_PATCH_ADVISE)
@patch(_PATCH_CLASSIFY)
def test_runner_verification_fails(mock_classify, mock_advise, mock_gen):
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    mock_classify.return_value = _make_classified(entities=[entity])
    mock_advise.return_value = _plan_with(["foo"], "utils.py")
    # Return a split where foo's source is NOT present anywhere.
    mock_gen.return_value = SplitResult(
        new_files={"utils.py": "# empty placeholder"},
        original_source="# empty original",
        abort=False,
    )

    result = run_file_limiter("big.py", "", source, [], _CONFIG)

    assert result.abort is True
    assert result.original_source == source
    assert any("verification failed" in m for m in result.messages)
