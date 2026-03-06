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
def test_placement_prompt_includes_mermaid_when_deps_exist(
    mock_key, mock_client, mock_call
):
    """When inter-group deps exist, Mermaid diagram is included in the prompt."""
    mock_key.return_value = "key"
    mock_client.return_value = MagicMock()
    mock_call.return_value = {
        "placements": [
            {"group_id": 0, "target_file": "utils.py"},
            {"group_id": 1, "target_file": "models.py"},
        ]
    }
    c = _classified(
        entities=[_make_entity("foo", 1, 5), _make_entity("bar", 6, 10)],
        set_2_groups=[["foo"], ["bar"]],
        graph={"foo": {"bar"}, "bar": set()},
    )
    advise_file_limiter(c, "src/big.py", _CONFIG)

    messages = mock_call.call_args[0][6]
    assert "```mermaid" in messages[0]["content"]
    assert "G0 --> G1" in messages[0]["content"]
