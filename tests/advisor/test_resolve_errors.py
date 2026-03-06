from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import resolve_naming_conflicts
from .helpers import (
    _CONFLICTING_PLACEMENTS,
    _PATCH_CALL,
    _PATCH_CLIENT,
    _PATCH_KEY,
    _classified,
)


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_llm_none_returns_none(mock_key, mock_client, mock_call):
    """LLM returns None → resolve returns None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = None
    c = _classified()
    result = resolve_naming_conflicts(
        _CONFLICTING_PLACEMENTS,
        c,
        "src/big.py",
        frozenset(),
        frozenset(),
        CrispenConfig(file_limiter_retries=0),
    )
    assert result is None


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_forbidden_target_returns_none(mock_key, mock_client, mock_call):
    """LLM picks a target that is in forbidden_files → resolve returns None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    # "helpers.py" is a non-conflicting target → included in forbidden_files.
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "helpers.py"},
            {"group_id": 1, "target_file": "services.py"},
        ]
    }
    c = _classified()
    result = resolve_naming_conflicts(
        _CONFLICTING_PLACEMENTS,
        c,
        "src/big.py",
        frozenset(),
        frozenset(),
        CrispenConfig(file_limiter_retries=0),
    )
    assert result is None


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_incomplete_response_returns_none(mock_key, mock_client, mock_call):
    """LLM returns fewer placements than groups → len mismatch → None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "models.py"}]  # only 1 of 2
    }
    c = _classified()
    result = resolve_naming_conflicts(
        _CONFLICTING_PLACEMENTS,
        c,
        "src/big.py",
        frozenset(),
        frozenset(),
        CrispenConfig(file_limiter_retries=0),
    )
    assert result is None
