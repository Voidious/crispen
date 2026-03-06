from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import resolve_naming_conflicts
from .helpers import (
    _CLEAN_PLACEMENTS,
    _CONFIG,
    _CONFLICTING_PLACEMENTS,
    _PATCH_CALL,
    _PATCH_CLIENT,
    _PATCH_KEY,
    _classified,
    _make_entity,
)


def test_resolve_no_conflicts_returns_unchanged():
    """No conflicts → returns a copy of the input list; no LLM calls needed."""
    c = _classified(entities=[_make_entity("foo", 1, 5), _make_entity("bar", 6, 10)])
    result = resolve_naming_conflicts(
        _CLEAN_PLACEMENTS, c, "src/big.py", frozenset(), frozenset(), _CONFIG
    )
    assert result == _CLEAN_PLACEMENTS
    assert result is not _CLEAN_PLACEMENTS


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_success(mock_key, mock_client, mock_call):
    """Happy path: forbidden_dir_stems and existing_file_stems both non-empty;
    prev_failure is False on the first (successful) attempt."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "models.py"},
            {"group_id": 1, "target_file": "services.py"},
        ]
    }
    c = _classified(
        entities=[_make_entity("foo", 1, 5), _make_entity("bar", 6, 10)],
    )
    result = resolve_naming_conflicts(
        _CONFLICTING_PLACEMENTS,
        c,
        "src/big.py",
        existing_files=frozenset({"other.py"}),  # non-empty → existing_file_stems
        existing_dirs=frozenset({"mydir"}),  # non-empty → forbidden_dir_stems
        config=_CONFIG,
    )
    assert result is not None
    assert result[0].target_file == "models.py"
    assert result[1].target_file == "services.py"
    assert result[2].target_file == "helpers.py"  # non-conflicting, unchanged
    assert mock_call.call_count == 1


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_retry_succeeds(mock_key, mock_client, mock_call):
    """First attempt None, second succeeds; covers if prev_failure: True branch."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        None,
        {
            "placements": [
                {"group_id": 0, "target_file": "models.py"},
                {"group_id": 1, "target_file": "services.py"},
            ]
        },
    ]
    c = _classified()
    result = resolve_naming_conflicts(
        _CONFLICTING_PLACEMENTS,
        c,
        "src/big.py",
        frozenset(),
        frozenset(),
        CrispenConfig(file_limiter_retries=1),
    )
    assert result is not None
    assert result[0].target_file == "models.py"
    assert mock_call.call_count == 2


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_empty_forbidden_dir_stems(mock_key, mock_client, mock_call):
    """existing_dirs empty → forbidden_dir_stems empty → branch False."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "models.py"},
            {"group_id": 1, "target_file": "services.py"},
        ]
    }
    c = _classified()
    result = resolve_naming_conflicts(
        _CONFLICTING_PLACEMENTS,
        c,
        "src/big.py",
        existing_files=frozenset({"other.py"}),  # file_stems non-empty
        existing_dirs=frozenset(),  # dir_stems empty
        config=_CONFIG,
    )
    assert result is not None


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_empty_existing_file_stems(mock_key, mock_client, mock_call):
    """existing_files=frozenset() → file_stems empty → if existing_file_stems: False."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "models.py"},
            {"group_id": 1, "target_file": "services.py"},
        ]
    }
    c = _classified()
    result = resolve_naming_conflicts(
        _CONFLICTING_PLACEMENTS,
        c,
        "src/big.py",
        existing_files=frozenset(),  # file_stems empty
        existing_dirs=frozenset({"mydir"}),  # dir_stems non-empty
        config=_CONFIG,
    )
    assert result is not None
