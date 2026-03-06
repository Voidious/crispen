from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import advise_file_limiter
from .helpers import (
    _CONFIG,
    _PATCH_CALL,
    _PATCH_CLIENT,
    _PATCH_KEY,
    _classified,
    _make_entity,
)


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_call_returns_none_aborts(mock_key, mock_client, mock_call):
    """Call 1 (set3 advice) returns None → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = None

    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)
    assert plan.abort is True


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_placement_call_returns_none_aborts(mock_key, mock_client, mock_call):
    """Placement chunk exhausts all per-chunk retries → abort."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    # file_limiter_retries=0 → 1 attempt per chunk; set3 call + 1 placement attempt.
    mock_call.side_effect = [
        {"decisions": [{"group_id": 0, "action": "migrate"}]},
        None,
    ]
    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", CrispenConfig(file_limiter_retries=0))
    assert plan.abort is True
