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
def test_resolve_non_int_group_id(mock_key, mock_client, mock_call):
    """Non-integer group_id → isinstance check fails → skipped → len mismatch → None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": "zero", "target_file": "models.py"},
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
def test_resolve_out_of_range_group_id(mock_key, mock_client, mock_call):
    """Out-of-range group_id → range check fails → skipped → len mismatch → None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 99, "target_file": "models.py"},
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
def test_resolve_duplicate_group_id(mock_key, mock_client, mock_call):
    """Duplicate group_id → second entry skipped → len mismatch → None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "models.py"},
            {"group_id": 0, "target_file": "other.py"},  # duplicate
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
def test_resolve_empty_target(mock_key, mock_client, mock_call):
    """Empty target_file → falsy check fails → skipped → len mismatch → None."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": ""},
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
