from __future__ import annotations
from unittest.mock import MagicMock, patch
from crispen.config import CrispenConfig
from crispen.llm_client import LLMCallResult
from crispen.file_limiter.advisor import (
    GroupPlacement,
    _LLMAccumulator,
    _assign_placements_chunk,
    _build_group_mermaid,
    _compute_projected_lines,
    _group_summary,
    _propose_files_step,
    _refine_merge_tiny,
)
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.entity_parser import Entity, EntityKind


def _make_entity(
    name: str,
    start: int,
    end: int,
    *,
    docstring=None,
    params=None,
) -> Entity:
    return Entity(
        EntityKind.FUNCTION,
        name,
        start,
        end,
        [name],
        docstring=docstring,
        params=params or [],
    )


def _classified(
    *,
    entities=None,
    entity_class=None,
    graph=None,
    set_1=None,
    set_2_groups=None,
    set_3_groups=None,
    abort=False,
) -> ClassifiedEntities:
    return ClassifiedEntities(
        entities=entities or [],
        entity_class=entity_class or {},
        graph=graph if graph is not None else {},
        set_1=set_1 or [],
        set_2_groups=set_2_groups or [],
        set_3_groups=set_3_groups or [],
        abort=abort,
    )


def _make_llm_result(tool_input) -> LLMCallResult:
    """Wrap a dict (or None) in LLMCallResult for mock_call.return_value."""
    return LLMCallResult(
        tool_input=tool_input, elapsed=0.01, input_tokens=10, output_tokens=5
    )


def _propose_ok(*filenames: str) -> LLMCallResult:
    """Return a valid propose_output_files LLM response for the given filenames."""
    return _make_llm_result(
        {"files": [{"filename": f, "description": "auto-generated"} for f in filenames]}
    )


_CONFIG = CrispenConfig()
_PATCH_KEY = "crispen.file_limiter.advisor.get_api_key"
_PATCH_CLIENT = "crispen.file_limiter.advisor.make_client"
_PATCH_CALL = "crispen.file_limiter.advisor.placement.call_with_tool"


def test_group_summary_with_docstring_and_params():
    """Entity with docstring and params → both appear in summary."""
    ent = _make_entity(
        "foo",
        1,
        10,
        docstring="Parse the config file. More details here.",
        params=["path: str", "strict: bool"],
    )
    summary = _group_summary(["foo"], {"foo": ent})
    assert "foo (10 lines)" in summary
    assert '"Parse the config file."' in summary
    assert "params: path: str, strict: bool" in summary


def test_group_summary_with_params_only():
    """Entity with params but no docstring → params appear, no docstring quote."""
    ent = _make_entity("bar", 1, 5, params=["x: int", "y"])
    summary = _group_summary(["bar"], {"bar": ent})
    assert "params: x: int, y" in summary
    assert '"' not in summary


def test_group_summary_docstring_no_period():
    """Docstring with no period → full text used as first sentence."""
    ent = _make_entity("baz", 1, 3, docstring="No period here")
    summary = _group_summary(["baz"], {"baz": ent})
    assert '"No period here"' in summary


def test_group_summary_with_section_header():
    """Entity with section_header → section appears first in extras."""

    ent = Entity(
        EntityKind.FUNCTION,
        "foo",
        1,
        5,
        ["foo"],
        section_header="Helpers",
    )
    summary = _group_summary(["foo"], {"foo": ent})
    assert 'section: "Helpers"' in summary


def test_group_summary_no_section_header():
    """Entity without section_header → no 'section:' in summary."""
    ent = _make_entity("bar", 1, 5)
    summary = _group_summary(["bar"], {"bar": ent})
    assert "section:" not in summary


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
def test_assign_placements_chunk_constrained_success(mock_call):
    """Constrained mode: target in proposed_filenames → placement accepted."""
    mock_call.return_value = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "utils.py"}]}
    )
    c = _classified(entities=[_make_entity("foo", 1, 5)])
    proposed = [("utils.py", "general utilities"), ("models.py", "data models")]
    result = _assign_placements_chunk(
        [["foo"]],
        c,
        "src/big.py",
        frozenset(),
        MagicMock(),
        _CONFIG,
        proposed_files=proposed,
    )
    assert result is not None
    assert result[0].target_file == "utils.py"
    # Prompt should list proposed files and instruct constrained choice.
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "Proposed output files" in prompt
    assert "utils.py" in prompt
    assert "models.py" in prompt


@patch(_PATCH_CALL)
def test_assign_placements_chunk_constrained_invalid_target(mock_call):
    """Constrained mode: target not in proposed_filenames → immediate None return."""
    mock_call.return_value = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "rogue_file.py"}]}
    )
    c = _classified(entities=[_make_entity("foo", 1, 5)])
    proposed = [("utils.py", "general utilities")]
    result = _assign_placements_chunk(
        [["foo"]],
        c,
        "src/big.py",
        frozenset(),
        MagicMock(),
        _CONFIG,
        proposed_files=proposed,
    )
    assert result is None


@patch(_PATCH_CALL)
def test_propose_files_step_success(mock_call):
    """Basic success: valid filenames are returned."""
    mock_call.return_value = _make_llm_result(
        {
            "files": [
                {"filename": "utils.py", "description": "utility functions"},
                {"filename": "models.py", "description": "data models"},
            ]
        }
    )
    c = _classified(entities=[_make_entity("foo", 1, 50)])
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), MagicMock(), _CONFIG
    )
    assert result is not None
    assert len(result) == 2
    assert result[0] == ("utils.py", "utility functions")
    assert result[1] == ("models.py", "data models")


@patch(_PATCH_CALL)
def test_propose_files_step_llm_none(mock_call):
    """call_with_tool returns None → _propose_files_step returns None."""
    mock_call.return_value = _make_llm_result(None)
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), MagicMock(), _CONFIG
    )
    assert result is None


@patch(_PATCH_CALL)
def test_propose_files_step_empty_files_list(mock_call):
    """LLM returns empty files list → returns None (not proposed)."""
    mock_call.return_value = _make_llm_result({"files": []})
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), MagicMock(), _CONFIG
    )
    assert result is None


@patch(_PATCH_CALL)
def test_propose_files_step_strips_existing_files(mock_call):
    """Filename in existing_files is stripped; remaining valid ones returned."""
    mock_call.return_value = _make_llm_result(
        {
            "files": [
                {"filename": "taken.py", "description": "already exists"},
                {"filename": "utils.py", "description": "new file"},
            ]
        }
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]],
        c,
        "src/big.py",
        2,
        frozenset({"taken.py"}),
        MagicMock(),
        _CONFIG,
    )
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == "utils.py"


@patch(_PATCH_CALL)
def test_propose_files_step_all_in_existing_files(mock_call):
    """All proposed filenames are in existing_files → stripped → returns None."""
    mock_call.return_value = _make_llm_result(
        {"files": [{"filename": "taken.py", "description": "existing"}]}
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]],
        c,
        "src/big.py",
        2,
        frozenset({"taken.py"}),
        MagicMock(),
        _CONFIG,
    )
    assert result is None


@patch(_PATCH_CALL)
def test_propose_files_step_strips_duplicates(mock_call):
    """Duplicate filenames are stripped; only first occurrence kept."""
    mock_call.return_value = _make_llm_result(
        {
            "files": [
                {"filename": "utils.py", "description": "first"},
                {"filename": "utils.py", "description": "duplicate"},
            ]
        }
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), MagicMock(), _CONFIG
    )
    assert result is not None
    assert len(result) == 1
    assert result[0] == ("utils.py", "first")


@patch(_PATCH_CALL)
def test_propose_files_step_strips_empty_filename(mock_call):
    """Empty filename string is skipped; valid ones returned."""
    mock_call.return_value = _make_llm_result(
        {
            "files": [
                {"filename": "", "description": "empty"},
                {"filename": "utils.py", "description": "valid"},
            ]
        }
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), MagicMock(), _CONFIG
    )
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == "utils.py"


@patch(_PATCH_CALL)
def test_propose_files_step_verbose(mock_call, capsys):
    """verbose=True prints propose message to stderr and increments counter."""
    mock_call.return_value = _make_llm_result(
        {"files": [{"filename": "utils.py", "description": "utilities"}]}
    )
    c = _classified()
    acc = _LLMAccumulator()
    result = _propose_files_step(
        [["foo"]],
        c,
        "src/big.py",
        2,
        frozenset(),
        MagicMock(),
        _CONFIG,
        verbose=True,
        _acc=acc,
    )
    assert result is not None
    assert acc.calls == 1
    err = capsys.readouterr().err
    assert "propose" in err.lower()


@patch(_PATCH_CALL)
def test_propose_files_step_no_counter(mock_call):
    """_counter=None covers the None-counter branch (no increment)."""
    mock_call.return_value = _make_llm_result(
        {"files": [{"filename": "utils.py", "description": "utilities"}]}
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), MagicMock(), _CONFIG
    )
    assert result is not None


@patch(_PATCH_CALL)
def test_propose_files_step_subdir_name(mock_call):
    """subdir_name triggers the subdir placement_rule branch in the prompt."""
    mock_call.return_value = _make_llm_result(
        {"files": [{"filename": "handlers.py", "description": "request handlers"}]}
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]],
        c,
        "src/service.py",
        2,
        frozenset(),
        MagicMock(),
        _CONFIG,
        subdir_name="service",
    )
    assert result is not None
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "service/" in prompt


@patch(_PATCH_CALL)
def test_propose_files_step_no_existing_files(mock_call):
    """existing_files=frozenset() → exclude_section empty (branch False)."""
    mock_call.return_value = _make_llm_result(
        {"files": [{"filename": "utils.py", "description": "utilities"}]}
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]], c, "src/big.py", 2, frozenset(), MagicMock(), _CONFIG
    )
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "already exist" not in prompt
    assert result is not None


@patch(_PATCH_CALL)
def test_propose_files_step_with_existing_files(mock_call):
    """existing_files non-empty → exclude_section added to prompt (branch True)."""
    mock_call.return_value = _make_llm_result(
        {"files": [{"filename": "utils.py", "description": "utilities"}]}
    )
    c = _classified()
    result = _propose_files_step(
        [["foo"]],
        c,
        "src/big.py",
        2,
        frozenset({"other.py"}),
        MagicMock(),
        _CONFIG,
    )
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "already exist" in prompt
    assert result is not None


@patch(_PATCH_CALL)
def test_propose_files_step_prev_failure(mock_call):
    """prev_failure is appended to the propose prompt."""
    mock_call.return_value = _make_llm_result(
        {"files": [{"filename": "utils.py", "description": "utilities"}]}
    )
    c = _classified()
    _propose_files_step(
        [["foo"]],
        c,
        "src/big.py",
        2,
        frozenset(),
        MagicMock(),
        _CONFIG,
        prev_failure="sentinel_propose_failure",
    )
    prompt = mock_call.call_args[0][6][0]["content"]
    assert "sentinel_propose_failure" in prompt


def test_compute_projected_lines_basic():
    """Entities found in map → lines counted per target file."""
    entity_a = _make_entity("func_a", 1, 50)  # 50 lines
    entity_b = _make_entity("func_b", 51, 100)  # 50 lines
    entity_map = {"func_a": entity_a, "func_b": entity_b}
    placements = [
        GroupPlacement(group=["func_a"], target_file="utils.py"),
        GroupPlacement(group=["func_b"], target_file="utils.py"),
    ]
    projected = _compute_projected_lines(placements, entity_map)
    assert projected == {"utils.py": 100}


def test_compute_projected_lines_unknown_entity():
    """Entity name not in map → no lines added for that entity (skipped)."""
    entity_map = {}  # nothing in the map
    placements = [GroupPlacement(group=["ghost"], target_file="utils.py")]
    projected = _compute_projected_lines(placements, entity_map)
    assert projected == {}


def test_compute_projected_lines_multiple_files():
    """Entities across multiple target files → separate line counts."""
    entity_a = _make_entity("func_a", 1, 100)  # 100 lines
    entity_b = _make_entity("func_b", 101, 200)  # 100 lines
    entity_map = {"func_a": entity_a, "func_b": entity_b}
    placements = [
        GroupPlacement(group=["func_a"], target_file="module_a.py"),
        GroupPlacement(group=["func_b"], target_file="module_b.py"),
    ]
    projected = _compute_projected_lines(placements, entity_map)
    assert projected == {"module_a.py": 100, "module_b.py": 100}


def test_refine_merge_tiny_no_tiny_files():
    """All projected files are above the tiny threshold → no merge, return unchanged."""
    # Entity with 300 lines is well above min_size (200 for 1000-line limit).
    entity = _make_entity("large_func", 1, 300)
    c = _classified(entities=[entity])
    placements = [GroupPlacement(group=["large_func"], target_file="utils.py")]
    proposed_files = [("utils.py", "large functions"), ("models.py", "models")]

    result = _refine_merge_tiny(
        placements, proposed_files, c, "src/big.py", MagicMock(), _CONFIG
    )
    assert result == placements
    assert result is not placements


def test_refine_merge_tiny_no_ok_proposed():
    """All proposed files are tiny → ok_proposed is empty → return unchanged."""
    # Two tiny entities, both below threshold.
    entity_a = _make_entity("tiny_a", 1, 10)
    entity_b = _make_entity("tiny_b", 11, 20)
    c = _classified(entities=[entity_a, entity_b])
    placements = [
        GroupPlacement(group=["tiny_a"], target_file="a.py"),
        GroupPlacement(group=["tiny_b"], target_file="b.py"),
    ]
    proposed_files = [("a.py", "tiny a"), ("b.py", "tiny b")]
    # Both files are tiny (10 and 10 lines < 200); no ok_proposed → no merge.

    result = _refine_merge_tiny(
        placements, proposed_files, c, "src/big.py", MagicMock(), _CONFIG
    )
    assert result == placements


@patch(_PATCH_CALL)
def test_refine_merge_tiny_success(mock_call):
    """Tiny file group is merged into a larger file successfully."""
    # LLM reassigns the tiny group to the large file.
    mock_call.return_value = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "large.py"}]}
    )

    entity_small = _make_entity("small_func", 1, 10)  # 10 lines (tiny)
    entity_large = _make_entity("large_func", 11, 310)  # 300 lines (not tiny)
    c = _classified(entities=[entity_small, entity_large])

    placements = [
        GroupPlacement(group=["small_func"], target_file="small.py"),
        GroupPlacement(group=["large_func"], target_file="large.py"),
    ]
    proposed_files = [("small.py", "small"), ("large.py", "large")]
    # small.py: 10 lines (tiny <200); large.py: 300 lines (not tiny).
    # ok_proposed = [("large.py", "large")]; refinement merges small.py into large.py.

    result = _refine_merge_tiny(
        placements, proposed_files, c, "src/big.py", MagicMock(), _CONFIG
    )
    assert len(result) == 2
    # small_func should now be in large.py.
    small_placement = next(r for r in result if "small_func" in r.group)
    assert small_placement.target_file == "large.py"
    # large_func remains in large.py.
    large_placement = next(r for r in result if "large_func" in r.group)
    assert large_placement.target_file == "large.py"


@patch(_PATCH_CALL)
def test_refine_merge_tiny_llm_fails(mock_call):
    """Reassignment LLM returns None → original placements returned (best-effort)."""
    mock_call.return_value = _make_llm_result(None)  # LLM fails

    entity_small = _make_entity("small_func", 1, 10)
    entity_large = _make_entity("large_func", 11, 310)
    c = _classified(entities=[entity_small, entity_large])

    placements = [
        GroupPlacement(group=["small_func"], target_file="small.py"),
        GroupPlacement(group=["large_func"], target_file="large.py"),
    ]
    proposed_files = [("small.py", "small"), ("large.py", "large")]

    result = _refine_merge_tiny(
        placements, proposed_files, c, "src/big.py", MagicMock(), _CONFIG
    )
    # Best-effort: return original placements unchanged.
    assert len(result) == 2
    assert result[0].target_file == "small.py"
    assert result[1].target_file == "large.py"


@patch(_PATCH_CALL)
def test_refine_merge_tiny_verbose(mock_call, capsys):
    """verbose=True prints the refining message to stderr."""
    mock_call.return_value = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "large.py"}]}
    )

    entity_small = _make_entity("small_func", 1, 10)
    entity_large = _make_entity("large_func", 11, 310)
    c = _classified(entities=[entity_small, entity_large])

    placements = [
        GroupPlacement(group=["small_func"], target_file="small.py"),
        GroupPlacement(group=["large_func"], target_file="large.py"),
    ]
    proposed_files = [("small.py", "small"), ("large.py", "large")]

    _refine_merge_tiny(
        placements,
        proposed_files,
        c,
        "src/big.py",
        MagicMock(),
        _CONFIG,
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "refining" in err.lower()


@patch(_PATCH_CALL)
def test_assign_placements_chunk_existing_files_exclude_section(mock_call):
    """Free-form mode with non-empty existing_files builds the exclude section."""
    mock_call.return_value = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "helpers.py"}]}
    )
    entity = _make_entity("foo", 1, 10)
    c = _classified(entities=[entity], set_2_groups=[["foo"]])
    result = _assign_placements_chunk(
        [["foo"]],
        c,
        "src/big.py",
        frozenset({"existing.py"}),  # non-empty existing_files
        MagicMock(),
        _CONFIG,
        proposed_files=None,  # free-form mode
    )
    assert result is not None
    assert result[0].target_file == "helpers.py"


@patch(_PATCH_CALL)
def test_assign_placements_chunk_target_in_existing_files_returns_none(mock_call):
    """Free-form mode: target_file in existing_files → return None (line 589)."""
    mock_call.return_value = _make_llm_result(
        {"placements": [{"group_id": 0, "target_file": "existing.py"}]}
    )
    entity = _make_entity("foo", 1, 10)
    c = _classified(entities=[entity], set_2_groups=[["foo"]])
    result = _assign_placements_chunk(
        [["foo"]],
        c,
        "src/big.py",
        frozenset({"existing.py"}),  # target collides with existing file
        MagicMock(),
        _CONFIG,
        proposed_files=None,  # free-form mode
    )
    assert result is None
