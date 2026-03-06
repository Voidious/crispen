from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from crispen.config import CrispenConfig
from crispen.errors import CrispenAPIError
from crispen.file_limiter.advisor import resolve_naming_conflicts
from .helpers import (
    _CONFIG,
    _CONFLICTING_PLACEMENTS,
    _PATCH_CALL,
    _PATCH_CLIENT,
    _PATCH_KEY,
    _classified,
    _make_entity,
)


def test_resolve_api_key_error_propagates(monkeypatch):
    """Missing API key raises CrispenAPIError before any LLM call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = _classified(entities=[_make_entity("foo", 1, 5)])
    with pytest.raises(CrispenAPIError):
        resolve_naming_conflicts(
            _CONFLICTING_PLACEMENTS, c, "src/big.py", frozenset(), frozenset(), _CONFIG
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
