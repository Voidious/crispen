from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.file_limiter.advisor import (
    advise_file_limiter,
    GroupPlacement,
    resolve_naming_conflicts,
)
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
def test_advise_verbose_set3_and_placement(mock_key, mock_client, mock_call, capsys):
    """verbose=True exercises the print + _counter branches in _advise_set3
    and _assign_placements_chunk."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    # First call → set-3 decision; second call → placement.
    mock_call.side_effect = [
        {"decisions": [{"group_id": 0, "action": "migrate"}]},
        {"placements": [{"group_id": 0, "target_file": "utils.py"}]},
    ]
    c = _classified(
        entities=[_make_entity("bar", 1, 10)],
        set_3_groups=[["bar"]],
    )
    plan = advise_file_limiter(c, "src/big.py", _CONFIG, verbose=True)

    assert plan.abort is False
    assert plan.llm_calls == 2
    err = capsys.readouterr().err
    assert "set-3 group" in err
    assert "file placements" in err


@patch(_PATCH_CALL)
@patch(_PATCH_CLIENT)
@patch(_PATCH_KEY)
def test_resolve_verbose(mock_key, mock_client, mock_call, capsys):
    """verbose=True exercises the print + _counter branches in
    _rename_conflicting_chunk (with _counter passed to cover the increment)."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    # Both placements conflict (utils.py vs utils/io.py share stem "utils"),
    # so the chunk sent to LLM has 2 groups; return both renamed.
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "models.py"},
            {"group_id": 1, "target_file": "helpers.py"},
        ]
    }
    entity = _make_entity("foo", 1, 5)
    c = _classified(entities=[entity])
    placements = [
        GroupPlacement(group=["foo"], target_file="utils.py"),
        GroupPlacement(group=["bar"], target_file="utils/io.py"),  # conflict
    ]
    counter = [0]
    result = resolve_naming_conflicts(
        placements,
        c,
        "src/big.py",
        frozenset(),
        frozenset(),
        _CONFIG,
        verbose=True,
        _counter=counter,
    )

    assert result is not None
    assert counter[0] == 1  # one LLM call was counted
    err = capsys.readouterr().err
    assert "naming conflicts" in err
