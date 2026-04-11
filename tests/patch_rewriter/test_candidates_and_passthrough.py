from __future__ import annotations
from unittest.mock import MagicMock, patch as mock_patch
from crispen.config import CrispenConfig
from crispen.patch_rewriter import (
    _FLContext,
    _apply_cross_file_const_updates,
    _build_rename_guard_sets,
    _callgraph_update_file,
    _candidates_check,
    _is_bad_rename,
    _process_file_source,
    _rewrite_candidates_check,
)
from .test_patch_detection import _CFG, _PATCH_CALL_TOOL, _VERIFY_OK, _make_fl_ctx, _ok
from .test_callgraph_resolution import _make_cuf_contexts, _make_cuf_index


def test_is_bad_rename_pattern_a_shallowing_moved_out():
    # advisor.placement.call_with_tool → advisor.call_with_tool
    # call_with_tool is moved out; new_depth < old_depth → bad
    assert _is_bad_rename(
        "crispen.advisor.placement.call_with_tool",
        "crispen.advisor.call_with_tool",
        moved_out_names={"call_with_tool"},
        still_imported=set(),
        orig_users_map={},
        test_text="",
    )


def test_is_bad_rename_pattern_a_deepening_moved_out_ok():
    # Deepening a moved-out name is fine (not shallowing).
    assert not _is_bad_rename(
        "crispen.advisor.call_with_tool",
        "crispen.advisor.placement.call_with_tool",
        moved_out_names={"call_with_tool"},
        still_imported=set(),
        orig_users_map={},
        test_text="",
    )


def test_is_bad_rename_pattern_b_deepening_still_in_with_orig_user_in_test():
    # advisor.make_client → advisor.placement.make_client
    # make_client is still_imported, orig user advise_file_limiter is in test body → bad
    assert _is_bad_rename(
        "crispen.advisor.make_client",
        "crispen.advisor.placement.make_client",
        moved_out_names=set(),
        still_imported={"make_client"},
        orig_users_map={"make_client": ["advise_file_limiter"]},
        test_text="def test_foo():\n    advise_file_limiter(src)\n",
    )


def test_is_bad_rename_pattern_b_deepening_still_in_no_orig_user_in_test():
    # Same deepening but test body doesn't contain advise_file_limiter → ok
    assert not _is_bad_rename(
        "crispen.advisor.make_client",
        "crispen.advisor.placement.make_client",
        moved_out_names=set(),
        still_imported={"make_client"},
        orig_users_map={"make_client": ["advise_file_limiter"]},
        test_text="def test_foo():\n    _propose_files_step(src)\n",
    )


def test_is_bad_rename_pattern_b_deepening_no_orig_users_map():
    # Name is still_imported but not in orig_users_map → not blocked
    assert not _is_bad_rename(
        "crispen.advisor.make_client",
        "crispen.advisor.placement.make_client",
        moved_out_names=set(),
        still_imported={"make_client"},
        orig_users_map={},
        test_text="def test_foo():\n    advise_file_limiter(src)\n",
    )


def test_is_bad_rename_not_bad_when_no_relevant_sets():
    assert not _is_bad_rename(
        "a.b.foo",
        "a.b.c.foo",
        moved_out_names=set(),
        still_imported=set(),
        orig_users_map={},
        test_text="",
    )


def test_is_bad_rename_pattern_c_target_module_missing_name():
    # Target module "pkg.advisor.placement" exists in new_module_imports
    # but doesn't import call_with_tool; name is in moved_out_names → bad rename.
    assert _is_bad_rename(
        "pkg.advisor.call_with_tool",
        "pkg.advisor.placement.call_with_tool",
        moved_out_names={"call_with_tool"},
        still_imported=set(),
        orig_users_map={},
        test_text="",
        new_module_imports={"pkg.advisor.placement": {"make_client"}},
    )


def test_is_bad_rename_pattern_c_target_module_has_name():
    # Target module imports the name → not blocked by Pattern C.
    assert not _is_bad_rename(
        "pkg.advisor.call_with_tool",
        "pkg.advisor.placement.call_with_tool",
        moved_out_names={"call_with_tool"},
        still_imported=set(),
        orig_users_map={},
        test_text="",
        new_module_imports={"pkg.advisor.placement": {"call_with_tool"}},
    )


def test_is_bad_rename_pattern_c_target_module_unknown():
    # Target module not in new_module_imports (unknown module) → not blocked.
    assert not _is_bad_rename(
        "pkg.advisor.call_with_tool",
        "pkg.advisor.placement.call_with_tool",
        moved_out_names={"call_with_tool"},
        still_imported=set(),
        orig_users_map={},
        test_text="",
        new_module_imports={"pkg.advisor.schemas": {"call_with_tool"}},
    )


def test_is_bad_rename_pattern_c_name_not_tracked():
    # Name is not in moved_out_names or still_imported → Pattern C skipped
    # even if the target module doesn't import it (locally-defined symbols).
    assert not _is_bad_rename(
        "pkg.big.A",
        "pkg.sub_a.A",
        moved_out_names=set(),
        still_imported=set(),
        orig_users_map={},
        test_text="",
        new_module_imports={"pkg.sub_a": set()},
    )


def test_is_bad_rename_pattern_c_none_new_module_imports():
    # new_module_imports=None (not passed) → Pattern C skipped entirely.
    assert not _is_bad_rename(
        "pkg.advisor.call_with_tool",
        "pkg.advisor.placement.call_with_tool",
        moved_out_names={"call_with_tool"},
        still_imported=set(),
        orig_users_map={},
        test_text="",
        new_module_imports=None,
    )


def test_build_rename_guard_sets_new_module_imports():
    # new_files with known module paths populate new_module_imports correctly.
    ctx = _make_fl_ctx(
        original_source="from ...llm_client import call_with_tool, make_client\n",
        modified_source="from .placement import call_with_tool\n",
        new_files={
            "placement.py": "from ...llm_client import call_with_tool\n",
            "schemas.py": "from ...llm_client import make_client\n",
        },
        new_module_paths={
            "placement.py": "pkg.advisor.placement",
            "schemas.py": "pkg.advisor.schemas",
        },
    )
    _, _, _, new_mod_imports = _build_rename_guard_sets([ctx])
    assert new_mod_imports["pkg.advisor.placement"] == {"call_with_tool"}
    assert new_mod_imports["pkg.advisor.schemas"] == {"make_client"}


def test_cross_file_empty_proposals():
    msgs = list(_apply_cross_file_const_updates({}, {}))
    assert msgs == []


def test_cross_file_conflicting_proposals(tmp_path):
    """Multiple new values for the same constant → resolved is empty → skip."""
    f = tmp_path / "helpers.py"
    f.write_text('TARGET = "old.val"\n', encoding="utf-8")
    proposals = {str(f.resolve()): {"old.val": {"new.val1", "new.val2"}}}
    msgs = list(_apply_cross_file_const_updates(proposals, {}))
    assert msgs == []
    # File unchanged.
    assert f.read_text(encoding="utf-8") == 'TARGET = "old.val"\n'


def test_cross_file_per_file_entry_updated(tmp_path):
    """Const source file is in per_file → updates in-memory source, no disk write."""
    f = tmp_path / "helpers.py"
    f.write_text('TARGET = "old.val"\n', encoding="utf-8")
    per_file = {str(f): {"source": 'TARGET = "old.val"\n', "msgs": []}}
    proposals = {str(f.resolve()): {"old.val": {"new.val"}}}
    msgs = list(_apply_cross_file_const_updates(proposals, per_file))
    assert msgs == []
    assert '"new.val"' in per_file[str(f)]["source"]
    assert any("constant definition" in m for m in per_file[str(f)]["msgs"])
    # Disk file unchanged.
    assert f.read_text(encoding="utf-8") == 'TARGET = "old.val"\n'


def test_cross_file_per_file_entry_no_change(tmp_path):
    """Resolved new value equals old → apply_patch_strings makes no change → no msg."""
    f = tmp_path / "helpers.py"
    src = 'TARGET = "new.val"\n'  # already has new value
    per_file = {str(f): {"source": src, "msgs": []}}
    proposals = {str(f.resolve()): {"old.val": {"new.val"}}}
    # apply_patch_strings("TARGET = "new.val"\n", {"old.val": "new.val"}) → unchanged
    msgs = list(_apply_cross_file_const_updates(proposals, per_file))
    assert msgs == []
    assert per_file[str(f)]["msgs"] == []


def test_cross_file_disk_file_updated(tmp_path):
    """Const source file is a disk file → written, message yielded."""
    f = tmp_path / "helpers.py"
    f.write_text('TARGET = "old.val"\n', encoding="utf-8")
    proposals = {str(f.resolve()): {"old.val": {"new.val"}}}
    msgs = list(_apply_cross_file_const_updates(proposals, {}))
    assert len(msgs) == 1
    assert "constant definition" in msgs[0]
    assert '"new.val"' in f.read_text(encoding="utf-8")


def test_cross_file_disk_file_no_change(tmp_path):
    """Disk file already has the new value → no write, no message."""
    f = tmp_path / "helpers.py"
    f.write_text('TARGET = "new.val"\n', encoding="utf-8")
    proposals = {str(f.resolve()): {"old.val": {"new.val"}}}
    msgs = list(_apply_cross_file_const_updates(proposals, {}))
    assert msgs == []


def test_cross_file_disk_oserror(tmp_path):
    """OSError reading disk file → skipped silently."""
    f = tmp_path / "helpers.py"
    f.write_text('TARGET = "old.val"\n', encoding="utf-8")
    f.chmod(0o000)
    try:
        proposals = {str(f.resolve()): {"old.val": {"new.val"}}}
        msgs = list(_apply_cross_file_const_updates(proposals, {}))
        assert msgs == []
    finally:
        f.chmod(0o644)


@mock_patch(_PATCH_CALL_TOOL)
def test_process_passthrough_single_proposal_updates_const(mock_call, tmp_path):
    """One test passes through a constant (no rename), another proposes one value.
    Expected: since all proposals agree on one new value, the constant definition
    is updated — no per-function inlining needed.  The passthrough function
    continues using the const ref (which now points to the updated value).

    Covers:
      - same_file_passthrough path (new_val is None → continue, no add)
      - same_file_const_map populated for single-proposal (len==1) const
        even when some tests pass through unchanged
    """
    src = (
        'TARGET = "crispen.before.X"\n'
        'TARGET2 = "crispen.before.Y"\n'
        "\n"
        "@patch(TARGET)\n"
        "@patch(TARGET2)\n"
        "def test_a(mock_y, mock_x):\n"
        "    pass\n"
        "\n"
        "@patch(TARGET)\n"
        "def test_b(mock_x):\n"
        "    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # test_a renames Y but NOT X  → X passthrough from test_a's perspective.
    # test_b renames X            → single proposal for X (no disagreement).
    # With passthrough no longer blocking: both X and Y go to same_file_const_map.
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.Y": "crispen.after.Y"},
            }
        ),
        _ok(_VERIFY_OK),
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.after.X"},
            }
        ),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src,
        {"crispen.before.X", "crispen.before.Y"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
    )
    assert changed is True
    # Both constants updated via same_file_const_map.
    assert 'TARGET = "crispen.after.X"' in result
    assert 'TARGET2 = "crispen.after.Y"' in result
    # Decorators stay as const refs — no inlining of literals.
    assert "@patch(TARGET)" in result
    assert '@patch("crispen.after.X")' not in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_conflict_two_renames_existing_splice(mock_call, tmp_path):
    """Two tests rename the same constant to *different* targets → conflict.
    test_a also renames a literal patch → it gets a func_splice from string_swap.
    Expected: both functions get inlined with their respective literals;
    test_a's existing splice is *updated in place* (existing_idx path).

    Covers:
      - lines 1763-1772  (loop, build inline_subs)
      - line 1787-False  (inlined != base_text)
      - line 1789-True   (existing_idx not None → update splice)
      - line 1792        (existing_idx is None → append splice, for test_b)
    """
    src = (
        'TARGET = "crispen.before.X"\n'
        "\n"
        "@patch(TARGET)\n"
        '@patch("crispen.before.Z")\n'
        "def test_a(mock_z, mock_x):\n"
        "    pass\n"
        "\n"
        "@patch(TARGET)\n"
        "def test_b(mock_x):\n"
        "    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # test_a renames X → after_a.X and Z → after.Z.
    # test_b renames X → after_b.X.
    # Two different targets for X → conflict → inline each function individually.
    # test_a's Z literal rename creates an existing func_splice; the inline step
    # must update that existing splice rather than appending a new one.
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {
                    "crispen.before.X": "crispen.after_a.X",
                    "crispen.before.Z": "crispen.after.Z",
                },
            }
        ),
        _ok(_VERIFY_OK),
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.after_b.X"},
            }
        ),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src,
        {"crispen.before.X", "crispen.before.Z"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
    )
    assert changed is True
    # The shared TARGET constant must NOT be updated (conflict).
    assert 'TARGET = "crispen.before.X"' in result
    # test_a: Z literal renamed, X constant inlined.
    assert '@patch("crispen.after_a.X")' in result
    assert '@patch("crispen.after.Z")' in result
    # test_b: X constant inlined with its own target.
    assert '@patch("crispen.after_b.X")' in result
    # No original constant-style decorator survives.
    assert "@patch(TARGET)" not in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_conflict_two_proposals_passthrough_function_continue(
    mock_call, tmp_path
):
    """Two functions propose *different* values for TARGET → conflicting_old_vals.
    A third function also uses TARGET but only renames a different const (TARGET_Y).
    Expected: the third function is in string_swap_results but triggers the
    ``continue`` branch in the conflicting_old_vals inline loop (inline_subs
    empty for X); the other two get their decorators inlined individually.

    Covers the ``if not inline_subs: continue`` branch inside the
    ``if conflicting_old_vals:`` block (via two sub-paths):
      - ref.resolved_value NOT in conflicting_old_vals (Y ref → loop continues)
      - ref.resolved_value in conflicting_old_vals but new_val is None (X ref)
    """
    src = (
        'TARGET = "crispen.before.X"\n'
        'TARGET_Y = "crispen.before.Y"\n'
        "\n"
        "@patch(TARGET)\n"
        "def test_a(mock_x):\n"
        "    pass\n"
        "\n"
        "@patch(TARGET)\n"
        "def test_b(mock_x):\n"
        "    pass\n"
        "\n"
        "@patch(TARGET)\n"
        "@patch(TARGET_Y)\n"
        "def test_c(mock_y, mock_x):\n"
        "    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # test_a → after_a.X; test_b → after_b.X (two different proposals → conflicting)
    # test_c → renames Y only (not X) → in string_swap_results but inline_subs empty
    #   for X → continue.  Y gets a single proposal → same_file_const_map update.
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.after_a.X"},
            }
        ),
        _ok(_VERIFY_OK),
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.after_b.X"},
            }
        ),
        _ok(_VERIFY_OK),
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.Y": "crispen.after.Y"},
            }
        ),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src,
        {"crispen.before.X", "crispen.before.Y"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
    )
    assert changed is True
    # X: conflicting (two proposals) → const unchanged, test_a and test_b inlined.
    assert 'TARGET = "crispen.before.X"' in result
    assert '@patch("crispen.after_a.X")' in result
    assert '@patch("crispen.after_b.X")' in result
    # Y: single proposal → const updated via same_file_const_map.
    assert 'TARGET_Y = "crispen.after.Y"' in result
    # test_c: in string_swap_results (renamed Y) but X inline_subs empty → continue.
    assert "@patch(TARGET)" in result


def test_candidates_check_no_candidates():
    # No candidates for any path → None.
    assert _candidates_check({"pkg.orig.A": "pkg.sub.A"}, ["pkg.orig.A"], {}) is None


def test_candidates_check_rename_valid():
    # Rename is in candidates → None.
    cands = {"pkg.orig.A": ["pkg.placement.A", "pkg.helpers.A"]}
    assert (
        _candidates_check({"pkg.orig.A": "pkg.placement.A"}, ["pkg.orig.A"], cands)
        is None
    )


def test_candidates_check_rename_invalid():
    # Rename proposes a path not in candidates → error message.
    cands = {"pkg.orig.A": ["pkg.placement.A"]}
    result = _candidates_check({"pkg.orig.A": "pkg.wrong.A"}, ["pkg.orig.A"], cands)
    assert result is not None
    assert "pkg.wrong.A" in result
    assert "pkg.placement.A" in result


def test_candidates_check_no_change_with_candidates():
    # No rename proposed for a path that has candidates → error message.
    cands = {"pkg.orig.A": ["pkg.placement.A"]}
    result = _candidates_check({}, ["pkg.orig.A"], cands)
    assert result is not None
    assert "pkg.orig.A" in result
    assert "pkg.placement.A" in result


def test_candidates_check_path_not_in_candidates():
    # Another path has no candidates → passes; only paths with candidates are checked.
    cands = {"pkg.orig.A": ["pkg.placement.A"]}
    # pkg.orig.B has no candidates; even though no rename proposed → None
    assert _candidates_check({}, ["pkg.orig.B"], cands) is None


def test_callgraph_update_file_multiple_candidates_saved(tmp_path):
    # Both placement and conflict are reachable → 2 candidates → saved.
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(m):\n"
        "    helper()\n"
        "    resolve()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        _make_cuf_contexts(),
        scan_file=scan,
        index=index,
    )
    assert not changed  # ambiguous → no update
    assert "test_f" in unresolved
    assert "pkg.orig.use_fn" in unresolved["test_f"]
    cands = unresolved["test_f"]["pkg.orig.use_fn"]
    assert sorted(cands) == ["pkg.conflict.use_fn", "pkg.placement.use_fn"]


def test_callgraph_update_file_resolved_clears_candidates(tmp_path):
    # Single ctx with unique resolution → no candidates saved.
    placement_src = "from external import use_fn\ndef helper(): use_fn()\n"
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={"placement.py": placement_src},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(m):\n"
        "    helper()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        [ctx],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert "test_f" not in unresolved  # unique resolution → no candidates saved


def test_callgraph_update_file_resolved_clears_function_entry(tmp_path):
    # ctx_ambig gives 2 candidates (saves to unresolved); ctx_uniq resolves uniquely →
    # unresolved entry for the function is deleted (line 2695).
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(m):\n"
        "    helper()\n"
        "    resolve()\n"
    )
    ctx_ambig = _make_cuf_contexts()[0]  # both placement and conflict → 2 candidates
    ctx_uniq = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="from .placement import helper\n",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
        },
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        [ctx_ambig, ctx_uniq],
        scan_file=scan,
        index=index,
    )
    assert changed
    assert "test_f" not in unresolved  # ctx_uniq resolved → entry deleted


def test_rewrite_candidates_check_no_candidates():
    # No candidates for any path → None.
    text = '@patch("pkg.mod.A")\ndef test_f(m): pass\n'
    assert _rewrite_candidates_check(["pkg.mod.A"], text, {}) is None


def test_rewrite_candidates_check_valid_rename():
    # Old path absent, one candidate present → None.
    text = '@patch("pkg.placement.A")\ndef test_f(m): pass\n'
    cands = {"pkg.mod.A": ["pkg.placement.A", "pkg.other.A"]}
    assert _rewrite_candidates_check(["pkg.mod.A"], text, cands) is None


def test_rewrite_candidates_check_old_still_present():
    # Old path still present even though candidates exist → error.
    text = '@patch("pkg.mod.A")\ndef test_f(m): pass\n'
    cands = {"pkg.mod.A": ["pkg.placement.A"]}
    result = _rewrite_candidates_check(["pkg.mod.A"], text, cands)
    assert result is not None
    assert "pkg.mod.A" in result
    assert "pkg.placement.A" in result


def test_rewrite_candidates_check_renamed_to_unknown():
    # Old path absent but none of the candidates appear → error.
    text = '@patch("pkg.wrong.A")\ndef test_f(m): pass\n'
    cands = {"pkg.mod.A": ["pkg.placement.A", "pkg.other.A"]}
    result = _rewrite_candidates_check(["pkg.mod.A"], text, cands)
    assert result is not None
    assert "pkg.placement.A" in result


def test_rewrite_candidates_check_path_without_candidates_ignored():
    # A path with no candidates in the dict → skip it.
    text = '@patch("pkg.mod.B")\ndef test_f(m): pass\n'
    cands = {"pkg.mod.A": ["pkg.placement.A"]}  # A has candidates, B does not
    assert _rewrite_candidates_check(["pkg.mod.B"], text, cands) is None


def _make_process_cfg():
    return CrispenConfig(patch_update_retries=1, llm_verify_retries=0)
