from __future__ import annotations
from unittest.mock import MagicMock, patch
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
def test_plan_set2_only_skips_set3_call(mock_key, mock_client, mock_call):
    """set_2 groups only: call 1 is skipped, call 2 assigns placement."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [{"group_id": 0, "target_file": "utils.py"}]
    }
    c = _classified(
        entities=[_make_entity("foo", 1, 10)],
        set_2_groups=[["foo"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)

    assert plan.abort is False
    assert plan.set3_migrate == []
    assert len(plan.placements) == 1
    assert plan.placements[0].group == ["foo"]
    assert plan.placements[0].target_file == "utils.py"
    assert mock_call.call_count == 1  # only placement call


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_all_stay_no_placement(mock_key, mock_client, mock_call):
    """All Set 3 groups stay → no placement call, empty plan."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {"decisions": [{"group_id": 0, "action": "stay"}]}

    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)

    assert plan.abort is False
    assert plan.set3_migrate == []
    assert plan.placements == []
    assert mock_call.call_count == 1  # only set3 advice call


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set3_migrate(mock_key, mock_client, mock_call):
    """Set 3 group migrates → two LLM calls, placement returned."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        {"decisions": [{"group_id": 0, "action": "migrate"}]},
        {"placements": [{"group_id": 0, "target_file": "helpers.py"}]},
    ]
    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)

    assert plan.abort is False
    assert plan.set3_migrate == [["bar"]]
    assert len(plan.placements) == 1
    assert plan.placements[0].group == ["bar"]
    assert plan.placements[0].target_file == "helpers.py"
    assert mock_call.call_count == 2


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_plan_set2_and_set3_migrate(mock_key, mock_client, mock_call):
    """set_2 + migrating set_3 → both groups in placement call."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.side_effect = [
        {"decisions": [{"group_id": 0, "action": "migrate"}]},
        {
            "placements": [
                {"group_id": 0, "target_file": "new_stuff.py"},
                {"group_id": 1, "target_file": "changed.py"},
            ]
        },
    ]
    c = _classified(
        entities=[_make_entity("foo", 1, 5), _make_entity("bar", 6, 15)],
        set_2_groups=[["foo"]],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG)

    assert plan.abort is False
    assert plan.set3_migrate == [["bar"]]
    assert len(plan.placements) == 2
    targets = {p.target_file for p in plan.placements}
    assert targets == {"new_stuff.py", "changed.py"}
