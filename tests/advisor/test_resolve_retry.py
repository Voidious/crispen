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
