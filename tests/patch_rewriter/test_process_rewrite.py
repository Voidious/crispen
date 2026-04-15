from __future__ import annotations
from unittest.mock import MagicMock, patch as mock_patch
from crispen.config import CrispenConfig
from crispen.llm_client import LLMCallResult
from crispen.patch_rewriter import (
    RewriteAccumulator,
    _FLContext,
    _build_context_message,
    _compiles,
    _process_file_source,
)
from .context_builders import (
    _CFG,
    _CFG_NO_LLM_VERIFY,
    _CLASSIFY_REWRITE,
    _FORKING_PATHS,
    _PATCH_CALL_TOOL,
    _REWRITE_VERIFY_OK,
    _REWRITE_VERIFY_REJECT,
    _SRC_WITH_PATCH,
    _VERIFY_OK,
    _ok,
)
from .process_rewrite import _VALID_REWRITE
from .const_resolution import _SRC_WITH_CONST
from . import process_basic
from .process_basic import _PATCH_CALL_PR, _PATCH_GET_KEY_PR, _make_process_cfg


def test_compiles_valid():
    assert _compiles("x = 1\n") is True


def test_compiles_invalid():
    assert _compiles("def f(:\n    pass\n") is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_success(mock_call):
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is True
    assert "crispen.after.X" in result
    assert "crispen.after.Y" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_tool_none(mock_call):
    # Rewrite call returns tool_input=None → no update.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok(None),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_empty_text(mock_call):
    # Rewrite returns empty string → no update.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": ""}),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_non_string(mock_call):
    # Rewrite returns non-string value → no update.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": 42}),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_compile_error_retry(mock_call):
    # First rewrite has syntax error; second is valid.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": "def f(:\n    pass\n"}),  # invalid
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is True
    assert "crispen.after.X" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_compile_error_exhausted(mock_call):
    # Both rewrite attempts fail to compile → no update.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": "def f(:\n    pass\n"}),  # invalid
        _ok({"rewritten_function": "def f(:\n    pass\n"}),  # still invalid
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_verify_none_accept(mock_call):
    # Verify returns tool_input=None → accept the rewrite.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(None),  # verify returns None → accept
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1
    )
    assert changed is True
    assert "crispen.after.X" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_verify_rejected_then_accept(mock_call):
    # Verify rejects first rewrite; second rewrite+verify is accepted.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_REJECT),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 2
    )
    assert changed is True


@mock_patch(_PATCH_CALL_TOOL)
def test_process_needs_rewrite_verify_rejected_exhausted(mock_call):
    # Verify rejects with llm_verify_retries=0 → no update.
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_REJECT),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG_NO_LLM_VERIFY, 1
    )
    assert result == _SRC_WITH_PATCH
    assert changed is False


@mock_patch(_PATCH_CALL_TOOL)
def test_process_acc_rewrite_accumulates(mock_call):
    """Full rewrite path accumulates classify, rewrite, and verify calls."""
    mock_call.side_effect = [
        LLMCallResult(
            tool_input=_CLASSIFY_REWRITE,
            elapsed=0.5,
            input_tokens=100,
            output_tokens=10,
        ),
        LLMCallResult(
            tool_input={"rewritten_function": _VALID_REWRITE},
            elapsed=1.5,
            input_tokens=300,
            output_tokens=60,
        ),
        LLMCallResult(
            tool_input=_REWRITE_VERIFY_OK,
            elapsed=0.2,
            input_tokens=80,
            output_tokens=5,
        ),
    ]
    acc = RewriteAccumulator()
    _process_file_source(
        _SRC_WITH_PATCH, _FORKING_PATHS, "ctx", MagicMock(), _CFG, 1, _acc=acc
    )
    assert acc.calls == 3
    assert abs(acc.elapsed - 2.2) < 1e-9


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_rewrite_path(mock_call, capsys):
    """verbose=True prints 'rewriting', 'verifying rewrite', and 'rewrote'."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "rewriting" in err
    assert "verifying rewrite" in err
    assert "rewrote" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_rewrite_verify_rejected(mock_call, capsys):
    """verbose=True prints 'REJECTED' and issue when rewrite verify fails."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_REJECT),
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        _CFG,
        2,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "REJECTED" in err
    assert "wrong mock setup" in err
    assert "ACCEPTED" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_verbose_rewrite_compile_retry(mock_call, capsys):
    """verbose=True prints '(retry)' when rewrite compile fails."""
    mock_call.side_effect = [
        _ok(_CLASSIFY_REWRITE),
        _ok({"rewritten_function": "def f(:\n    pass\n"}),  # invalid
        _ok({"rewritten_function": _VALID_REWRITE}),
        _ok(_REWRITE_VERIFY_OK),
    ]
    cfg = CrispenConfig(patch_update_retries=1, timing="detailed")
    _process_file_source(
        _SRC_WITH_PATCH,
        _FORKING_PATHS,
        "ctx",
        MagicMock(),
        cfg,
        2,
        scan_file="tests/test_foo.py",
        verbose=True,
    )
    err = capsys.readouterr().err
    assert "rewriting" in err
    assert "(retry)" in err


@mock_patch(_PATCH_CALL_TOOL)
def test_process_rewrite_restores_unchanged_const_ref(mock_call):
    """After full rewrite, @patch("value") left unchanged → reverted to @patch(NAME)."""
    src = (
        'STABLE = "pkg.stable.X"\n'
        'TARGET = "pkg.big.A"\n\n'
        "@patch(STABLE)\n"
        "@patch(TARGET)\n"
        "def test_f(mock_stable, mock_target):\n"
        "    pass\n"
    )
    # LLM updates TARGET but leaves STABLE's substituted literal unchanged.
    rewritten = (
        '@patch("pkg.stable.X")\n'
        '@patch("pkg.sub_a.A")\n'
        "def test_f(mock_stable, mock_target):\n"
        "    pass\n"
    )
    mock_call.side_effect = [
        _ok({"needs_rewrite": True}),
        _ok({"rewritten_function": rewritten}),
        _ok({"correct": True, "issue": ""}),
    ]
    result, changed, _ = _process_file_source(
        src,
        {"pkg.big.A"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file="tests/test_foo.py",
    )
    assert changed is True
    # STABLE decorator reverted to named constant form.
    assert "@patch(STABLE)" in result
    assert '@patch("pkg.stable.X")' not in result
    # TARGET decorator keeps the LLM's updated literal value.
    assert '@patch("pkg.sub_a.A")' in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_const_same_file_update(mock_call, tmp_path):
    """Same-file const ref → same_file_const_map updates the const definition."""
    scan = str(tmp_path / "test_foo.py")
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.after.X"},
            }
        ),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        _SRC_WITH_CONST,
        {"crispen.before.X"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
    )
    assert changed is True
    # apply_patch_strings updates the const definition.
    assert '"crispen.after.X"' in result
    assert '"crispen.before.X"' not in result
    # No cross-file updates for same-file const.
    assert cross == {}


@mock_patch(_PATCH_CALL_TOOL)
def test_process_const_cross_file_update(mock_call, tmp_path):
    """Const ref from imported file → cross_file_patch_maps returned."""
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "crispen.before.X"\n', encoding="utf-8")
    src = "from .helpers import TARGET\n\n@patch(TARGET)\ndef test_f(mock):\n    pass\n"
    scan = str(tmp_path / "test_foo.py")
    mock_call.side_effect = [
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
        {"crispen.before.X"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
        repo_root=str(tmp_path),
    )
    helpers_abs = str(helpers.resolve())
    assert helpers_abs in cross
    assert cross[helpers_abs] == {"crispen.before.X": "crispen.after.X"}


@mock_patch(
    _PATCH_CALL_TOOL, return_value=_ok({"needs_rewrite": False, "patch_renames": {}})
)
def test_process_const_no_change_no_cross(mock_call, tmp_path):
    """LLM returns no renames → no change, cross is empty."""
    scan = str(tmp_path / "test_foo.py")
    result, changed, cross = _process_file_source(
        _SRC_WITH_CONST,
        {"crispen.before.X"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
    )
    assert changed is False
    assert cross == {}


@mock_patch(_PATCH_CALL_TOOL)
def test_process_cross_file_const_ref_not_in_renames(mock_call, tmp_path):
    """Cross-file const whose patch path is not in accepted renames → skipped.

    Scenario: function has two @patch decorators with different old paths. One
    is a cross-file const ref (path A) and the other is a literal (path B).
    Classify returns rename only for B; A is not in accepted renames.
    The const ref for A should be skipped.
    """
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET_A = "crispen.before.A"\n', encoding="utf-8")
    src = (
        "from .helpers import TARGET_A\n\n"
        '@patch(TARGET_A)\n@patch("crispen.before.B")\n'
        "def test_f(m1, m2):\n    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # Classify: only rename crispen.before.B → crispen.after.B;
    # crispen.before.A unchanged.
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.B": "crispen.after.B"},
            }
        ),
        _ok(_VERIFY_OK),
    ]
    result, changed, cross = _process_file_source(
        src,
        {"crispen.before.A", "crispen.before.B"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
        repo_root=str(tmp_path),
    )
    assert changed is True
    assert "crispen.after.B" in result
    # crispen.before.A not in accepted renames → no cross-file update for helpers.py.
    helpers_abs = str(helpers.resolve())
    assert helpers_abs not in cross


@mock_patch(_PATCH_CALL_TOOL)
def test_process_no_scan_file_no_const_processing(mock_call):
    """scan_file="" → const_map is empty, const post-processing skipped."""
    # Even with a const-ref style source, no scan_file means no const resolution.
    src = 'TARGET = "crispen.before.X"\n\n@patch(TARGET)\ndef test_f(m):\n    pass\n'
    # With scan_file="", const_map is empty, @patch(TARGET) is not collected.
    result, changed, cross = _process_file_source(
        src, {"crispen.before.X"}, "ctx", MagicMock(), _CFG, 1
    )
    assert result == src
    assert changed is False
    assert cross == {}
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL)
def test_process_attr_const_cross_file_update(mock_call, tmp_path):
    """@patch(constants.TARGET) resolved via import → cross-file proposal returned."""
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('TARGET = "crispen.before.X"\n', encoding="utf-8")
    src = (
        "import constants\n\n"
        "@patch(constants.TARGET)\n"
        "def test_f(mock_x):\n"
        "    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    mock_call.side_effect = [
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
        {"crispen.before.X"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
        repo_root=str(tmp_path),
    )
    # Cross-file proposal recorded for constants.py.
    constants_abs = str(constants_file.resolve())
    assert constants_abs in cross
    assert cross[constants_abs] == {"crispen.before.X": "crispen.after.X"}


@mock_patch(_PATCH_CALL_TOOL)
def test_process_passthrough_votes_conflict_with_rename_proposal(mock_call, tmp_path):
    """One test (A) renames Y but not X → casts "keep old" vote for X.
    Another test (B) renames X → casts "rename" vote for X.
    "keep old" + "rename" → conflicting proposals → inline test_b with new value;
    test_a's decorator unchanged.  TARGET2 (Y) has a single rename vote → updated
    via same_file_const_map.

    Covers:
      - "keep old" vote (new_val is None) entered into same_file_proposals
      - conflict detection (len > 1) → conflicting_old_vals
      - per-function inline for test_b (existing_idx is None → append)
      - test_a in conflicting inline loop with new_val=None → inline_subs empty
        → continue
      - single-proposal for TARGET2 (value != old) → same_file_const_map update
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
    # test_a renames Y but NOT X → X gets a "keep old" vote, Y gets a rename vote.
    # test_b renames X → X gets a "rename to after.X" vote.
    # X proposals: {old, after.X} → conflicting → inline test_b, test_a unchanged.
    # Y proposals: {after.Y}      → single, != old → same_file_const_map update.
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
    # X has conflicting votes → TARGET NOT updated globally.
    assert 'TARGET = "crispen.before.X"' in result
    # Y has single vote → TARGET2 updated via same_file_const_map.
    assert 'TARGET2 = "crispen.after.Y"' in result
    # test_b's X decorator is inlined individually.
    assert '@patch("crispen.after.X")' in result
    # test_a's decorator unchanged (its inline_subs were empty).
    assert "@patch(TARGET)" in result


@mock_patch(_PATCH_CALL_TOOL)
def test_process_passthrough_identity_proposal_skipped(mock_call, tmp_path):
    """One test renames Y but not X.  X only receives a "keep old" identity vote.
    Expected: TARGET not updated (identity guard: proposed == old); TARGET2 updated.

    Covers the ``next(iter(new_set)) != old`` identity guard in same_file_const_map
    that drops entries where the sole proposal equals the existing value.
    """
    src = (
        'TARGET = "crispen.before.X"\n'
        'TARGET2 = "crispen.before.Y"\n'
        "\n"
        "@patch(TARGET)\n"
        "@patch(TARGET2)\n"
        "def test_a(mock_y, mock_x):\n"
        "    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # test_a: renames Y → after.Y, does not rename X.
    # X proposals: {"crispen.before.X"} → len==1, value==old → identity skip.
    # Y proposals: {"crispen.after.Y"}  → len==1, value!=old → const_map update.
    mock_call.side_effect = [
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
    # X got only an identity vote → not in same_file_const_map → TARGET unchanged.
    assert 'TARGET = "crispen.before.X"' in result
    # Y got a rename vote → TARGET2 updated.
    assert 'TARGET2 = "crispen.after.Y"' in result


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


@mock_patch(_PATCH_CALL_PR)
@mock_patch(process_basic._PATCH_MAKE_CLIENT)
@mock_patch(_PATCH_GET_KEY_PR, return_value="key")
def test_process_file_source_rewrite_candidates_reject_and_retry(
    mock_key, mock_client, mock_call
):
    # Rewrite returns old path still present → rewrite candidates check rejects
    # without calling verify → retry; second rewrite uses valid candidate → accepted.
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    complex_logic()\n'
    ctx = _FLContext(
        filepath="/repo/pkg/big.py",
        old_module="pkg.big",
        original_source="from external import A\ndef f(): A()\n",
        modified_source="from .sub_a import f\n",
        new_files={"sub_a.py": "from external import A\ndef f(): A()\n"},
        new_module_paths={"sub_a.py": "pkg.sub_a"},
        entity_to_target={"f": "sub_a.py"},
        forking_old_paths={"pkg.big.A"},
    )
    context_msg = _build_context_message([ctx])
    bad_rewrite = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    complex_logic()\n'
    good_rewrite = '@patch("pkg.sub_a.A")\ndef test_f(mock_a):\n    complex_logic()\n'
    mock_call.side_effect = [
        # classify → needs rewrite
        LLMCallResult(
            tool_input={"needs_rewrite": True},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # rewrite 1: old path still present → rejected by _rewrite_candidates_check
        LLMCallResult(
            tool_input={"rewritten_function": bad_rewrite},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # rewrite 2: valid candidate
        LLMCallResult(
            tool_input={"rewritten_function": good_rewrite},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # verify
        LLMCallResult(
            tool_input={"correct": True, "issue": ""},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
    ]
    cg_candidates = {"test_f": {"pkg.big.A": ["pkg.sub_a.A"]}}
    new_src, changed, _ = _process_file_source(
        src,
        {"pkg.big.A"},
        context_msg,
        mock_client.return_value,
        _make_process_cfg(),
        max_attempts=2,
        cg_candidates=cg_candidates,
    )
    assert changed
    assert "pkg.sub_a.A" in new_src
    assert mock_call.call_count == 4  # classify + bad_rw + good_rw + verify


@mock_patch(_PATCH_CALL_PR)
@mock_patch(process_basic._PATCH_MAKE_CLIENT)
@mock_patch(_PATCH_GET_KEY_PR, return_value="key")
def test_process_file_source_candidates_all_retries_escalates_to_rewrite(
    mock_key, mock_client, mock_call, capsys
):
    # All classify retries exhausted with persistent candidates check rejections →
    # escalate to full rewrite rather than silently leaving the test broken.
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    ctx = _FLContext(
        filepath="/repo/pkg/big.py",
        old_module="pkg.big",
        original_source="from external import A\ndef f(): A()\n",
        modified_source="from .sub_a import f\n",
        new_files={"sub_a.py": "from external import A\ndef f(): A()\n"},
        new_module_paths={"sub_a.py": "pkg.sub_a"},
        entity_to_target={"f": "sub_a.py"},
        forking_old_paths={"pkg.big.A"},
    )
    context_msg = _build_context_message([ctx])
    good_rewrite = '@patch("pkg.sub_a.A")\ndef test_f(mock_a):\n    pass\n'
    mock_call.side_effect = [
        # First classify: no rename → rejected by candidates check (not last attempt).
        LLMCallResult(
            tool_input={"needs_rewrite": False, "patch_renames": {}},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Second classify: still no rename → last attempt → escalate to rewrite.
        LLMCallResult(
            tool_input={"needs_rewrite": False, "patch_renames": {}},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Rewrite (escalated from candidates check failure):
        LLMCallResult(
            tool_input={"rewritten_function": good_rewrite},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
        # Rewrite verify:
        LLMCallResult(
            tool_input={"correct": True, "issue": ""},
            elapsed=0.1,
            input_tokens=10,
            output_tokens=5,
        ),
    ]
    # Two candidates → ambiguous → LLM keeps returning no_change.
    cg_candidates = {"test_f": {"pkg.big.A": ["pkg.sub_a.A", "pkg.sub_b.A"]}}
    new_src, changed, _ = _process_file_source(
        src,
        {"pkg.big.A"},
        context_msg,
        mock_client.return_value,
        _make_process_cfg(),
        max_attempts=2,
        cg_candidates=cg_candidates,
        verbose=True,
    )
    assert changed
    assert "pkg.sub_a.A" in new_src
    assert mock_call.call_count == 4  # classify x2 + rewrite + verify
    err = capsys.readouterr().err
    assert "candidates check retries exhausted" in err
