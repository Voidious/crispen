from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.file_limiter.advisor import _build_group_mermaid, advise_file_limiter
from .test_advisor_plan_core import (
    _CONFIG,
    _PATCH_CALL,
    _PATCH_CLIENT,
    _PATCH_KEY,
    _classified,
    _make_entity,
)


def test_build_group_mermaid_no_edges():
    """Empty graph → no inter-group deps → returns empty string."""
    c = _classified(entities=[], set_2_groups=[["foo"], ["bar"]])
    result = _build_group_mermaid([["foo"], ["bar"]], c)
    assert result == ""


def test_build_group_mermaid_with_inter_group_dep():
    """G0 depends on G1 → Mermaid text with that edge is returned."""
    c = _classified(graph={"foo": {"bar"}, "bar": set()})
    result = _build_group_mermaid([["foo"], ["bar"]], c)
    assert "```mermaid" in result
    assert "G0 --> G1" in result


def test_build_group_mermaid_dep_outside_chunk():
    """Dep to entity outside chunk → dep_gid is None → not added → empty."""
    c = _classified(graph={"foo": {"external"}, "bar": set()})
    result = _build_group_mermaid([["foo"], ["bar"]], c)
    assert result == ""


def test_build_group_mermaid_intra_group_dep():
    """Dep within same SCC group → dep_gid == gid → not added as edge."""
    # foo and baz are in the same group; foo depends on baz (intra-SCC edge)
    c = _classified(graph={"foo": {"baz"}, "baz": {"foo"}, "bar": set()})
    result = _build_group_mermaid([["foo", "baz"], ["bar"]], c)
    assert result == ""


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
