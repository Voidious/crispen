from crispen.engine import _add_fl_context
from crispen.file_limiter.runner import FileLimiterResult


def test_add_fl_context_no_module_path():
    """When module path cannot be determined, _add_fl_context does nothing."""
    fl_list = []
    fl_result = FileLimiterResult(
        original_source="",
        new_files={},
        abort=False,
        entity_to_target={"X": "a.py"},
    )
    # A path with no ancestor containing pyproject.toml / .git → returns None.
    _add_fl_context(fl_list, "/no/project/root/here/file.py", "", fl_result, {})
    assert fl_list == []


def test_add_fl_context_no_forking(tmp_path):
    """When all entities are already in combined_patch_map, nothing is appended."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    fl_result = FileLimiterResult(
        original_source="",
        new_files={},
        abort=False,
        entity_to_target={"X": "a.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    # Entity already covered by combined_patch_map → forking_old_paths is empty.
    # No _block_N entities → nothing appended.
    _add_fl_context(fl_list, filepath, "", fl_result, {"mypkg.big.X": "mypkg.a.X"})
    assert fl_list == []


def test_add_fl_context_block_entity_uses_specific_names(tmp_path):
    """When a _block_N entity was moved and all named entities are mapped,
    the block-internal names (vars, imports) from the target file are used as
    specific scan keys — NOT the broad module path — so already-updated strings
    like ``old_module.sub.run_engine`` are not re-sent to the LLM."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    fl_result = FileLimiterResult(
        original_source="modified\n",
        new_files={
            "core.py": "_REFACTORS = []\nimport libcst as cst\n\ndef X(): pass\n"
        },
        abort=False,
        entity_to_target={"_block_1": "core.py", "X": "core.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    # Both entities are in the patch map → forking_old_paths would be empty.
    # _block_1 is a TOP_LEVEL block → scan core.py for block-internal names.
    # X is in entity_to_target so it's excluded; _REFACTORS and cst are not.
    combined = {
        "mypkg.big._block_1": "mypkg.core._block_1",
        "mypkg.big.X": "mypkg.core.X",
    }
    _add_fl_context(fl_list, filepath, "original\n", fl_result, combined)
    assert len(fl_list) == 1
    # Only _REFACTORS and cst are block-internal; X is excluded (named entity).
    assert fl_list[0].forking_old_paths == {"mypkg.big._REFACTORS", "mypkg.big.cst"}
    assert fl_list[0].old_module == "mypkg.big"


def test_add_fl_context_block_entity_no_new_names(tmp_path):
    """When a _block_N entity was moved but the target file contains no names
    beyond those already in entity_to_target, nothing is appended."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    fl_result = FileLimiterResult(
        original_source="modified\n",
        # core.py only defines X, which is already in entity_to_target.
        new_files={"core.py": "def X(): pass\n"},
        abort=False,
        entity_to_target={"_block_1": "core.py", "X": "core.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    combined = {
        "mypkg.big._block_1": "mypkg.core._block_1",
        "mypkg.big.X": "mypkg.core.X",
    }
    _add_fl_context(fl_list, filepath, "original\n", fl_result, combined)
    assert fl_list == []


def test_add_fl_context_forking_and_block_combined(tmp_path):
    """Forking entities AND block-internal names are both added to forking_old_paths."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    fl_result = FileLimiterResult(
        original_source="modified\n",
        new_files={"core.py": "_TIMEOUT = 30\ndef Y(): pass\n"},
        abort=False,
        # Y is forking (not in combined_patch_map); _block_1 moved with _TIMEOUT inside.
        entity_to_target={"_block_1": "core.py", "Y": "core.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    # Only _block_1 is in combined_patch_map; Y is not (forking).
    combined = {"mypkg.big._block_1": "mypkg.core._block_1"}
    _add_fl_context(fl_list, filepath, "original\n", fl_result, combined)
    assert len(fl_list) == 1
    # Y is a forking entity; _TIMEOUT is block-internal; Y in new file is excluded
    # (it's in all_entity_names).
    assert fl_list[0].forking_old_paths == {"mypkg.big.Y", "mypkg.big._TIMEOUT"}


def test_add_fl_context_normal(tmp_path):
    """Forking entity not in combined_patch_map → appended to fl_all_contexts."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    fl_result = FileLimiterResult(
        original_source="modified\n",
        new_files={"utils.py": "class X: pass\n"},
        abort=False,
        entity_to_target={"X": "utils.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    _add_fl_context(fl_list, filepath, "original\n", fl_result, {})
    assert len(fl_list) == 1
    assert fl_list[0].forking_old_paths == {"mypkg.big.X"}
    assert fl_list[0].old_module == "mypkg.big"
    assert fl_list[0].original_source == "original\n"
    assert fl_list[0].modified_source == "modified\n"


def test_add_fl_context_forked_import_alias_added(tmp_path):
    """Import aliases forked across multiple new files are added to forking_old_paths.

    When the original file imports ``call_with_tool`` and multiple new sub-files
    also import it, basic mode skips it (forking).  _add_fl_context must still
    add ``old_module.call_with_tool`` to forking_old_paths so the LLM rewrite
    step can detect and update @patch decorators that reference it.
    """
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    # Original file imports call_with_tool; both new sub-files also import it
    # (forking) so basic mode left it out of combined_patch_map.
    pre_split = "from external import call_with_tool\ndef F(): pass\n"
    fl_result = FileLimiterResult(
        original_source="modified\n",
        new_files={
            "a.py": (
                "from external import call_with_tool\ncall_with_tool()\ndef F(): pass\n"
            ),
            "b.py": "from external import call_with_tool\ncall_with_tool()\n",
        },
        abort=False,
        entity_to_target={"F": "a.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    # F is already in combined_patch_map (non-forking entity); call_with_tool
    # is NOT in combined_patch_map (forked, skipped by basic mode).
    combined = {"mypkg.big.F": "mypkg.a.F"}
    _add_fl_context(fl_list, filepath, pre_split, fl_result, combined)
    assert len(fl_list) == 1
    # call_with_tool must be in forking_old_paths despite F being already mapped.
    assert "mypkg.big.call_with_tool" in fl_list[0].forking_old_paths


def test_add_fl_context_forked_import_alias_entity_name_skipped(tmp_path):
    """Import alias that is also an entity name is skipped by the alias loop's continue.

    Helper is in entity_to_target (not in combined_patch_map) so the entity
    section already adds it to forking_old_paths.  The alias loop hits the
    ``continue`` branch and does not process it again.  ``other`` (import alias
    only, not an entity) is picked up by the alias loop instead.
    """
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    pre_split = "from ext import Helper, other\ndef F(): pass\n"
    fl_result = FileLimiterResult(
        original_source="modified\n",
        new_files={"a.py": "from ext import other\nother()\ndef F(): pass\n"},
        abort=False,
        # Helper is both an imported alias and a named entity (forking entity).
        entity_to_target={"Helper": "a.py", "F": "a.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    # Neither entity is in combined_patch_map → both are forking.
    _add_fl_context(fl_list, filepath, pre_split, fl_result, {})
    assert len(fl_list) == 1
    # Helper was added by the entity section; other was added by the alias loop.
    assert "mypkg.big.Helper" in fl_list[0].forking_old_paths
    assert "mypkg.big.other" in fl_list[0].forking_old_paths


def test_add_fl_context_forked_import_alias_already_mapped_skipped(tmp_path):
    """Import alias already in combined_patch_map is not re-added."""
    (tmp_path / "pyproject.toml").write_text("[tool.crispen]\n", encoding="utf-8")
    (tmp_path / "mypkg").mkdir()
    fl_list = []
    # pre_split imports Helper (already mapped by basic) and call_with_tool (forked).
    pre_split = "from ext import Helper, call_with_tool\ndef F(): pass\n"
    fl_result = FileLimiterResult(
        original_source="modified\n",
        new_files={
            "a.py": "from ext import call_with_tool\ncall_with_tool()\ndef F(): pass\n",
            "b.py": "from ext import call_with_tool\ncall_with_tool()\n",
        },
        abort=False,
        entity_to_target={"F": "a.py"},
    )
    filepath = str(tmp_path / "mypkg" / "big.py")
    # Helper is already in combined_patch_map (basic mapped it); call_with_tool is not.
    combined = {
        "mypkg.big.F": "mypkg.a.F",
        "mypkg.big.Helper": "mypkg.a.Helper",
    }
    _add_fl_context(fl_list, filepath, pre_split, fl_result, combined)
    assert len(fl_list) == 1
    # Helper is already mapped → not added again; call_with_tool is forked → added.
    assert "mypkg.big.Helper" not in fl_list[0].forking_old_paths
    assert "mypkg.big.call_with_tool" in fl_list[0].forking_old_paths


_REWRITE_PATCH = "crispen.engine.apply_patch_rewrite"
