from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.file_limiter.advisor import resolve_naming_conflicts
from .helpers import (
    _CONFIG,
    _CONFLICTING_PLACEMENTS,
    _PATCH_CALL,
    _PATCH_CLIENT,
    _PATCH_KEY,
    _classified,
)


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
