from __future__ import annotations
from unittest.mock import MagicMock, patch as mock_patch
from crispen.llm_client import LLMCallResult
from crispen.patch_rewriter import (
    _FLContext,
    _callgraph_update_file,
    _get_const_votes_from_rewrite,
    _process_file_source,
)
from ..helpers import (
    _CFG,
    _PATCH_CALL_TOOL,
    _VERIFY_OK,
    _make_cuf_index,
    _make_ref,
    _ok,
)


def test_get_const_votes_empty_refs():
    """No const_refs → empty dict, no parsing needed."""
    assert _get_const_votes_from_rewrite("def test_f(): pass\n", []) == {}


def test_get_const_votes_syntax_error():
    """Unparseable func_text → empty dict (SyntaxError branch)."""
    refs = [_make_ref("TARGET", "pkg.old.X")]
    assert _get_const_votes_from_rewrite("def f(:\n", refs) == {}


def test_get_const_votes_no_function_in_body():
    """Valid Python but no FunctionDef/AsyncFunctionDef → empty dict."""
    refs = [_make_ref("TARGET", "pkg.old.X")]
    result = _get_const_votes_from_rewrite("x = 1\n", refs)
    assert result == {}


def test_get_const_votes_non_call_decorator_skipped():
    """A bare-name decorator (not a Call node) is skipped without error."""
    code = "@pytest.mark.slow\n@patch(TARGET)\ndef test_f(m): pass\n"
    refs = [_make_ref("TARGET", "pkg.old.X")]
    # TARGET still present as Name → no vote entry (const unchanged).
    result = _get_const_votes_from_rewrite(code, refs)
    assert result == {}


def test_get_const_votes_non_patch_call_skipped():
    """A Call decorator whose func is not 'patch' is skipped."""
    code = "@other_decorator('pkg.old.X')\ndef test_f(m): pass\n"
    refs = [_make_ref("TARGET", "pkg.old.X")]
    result = _get_const_votes_from_rewrite(code, refs)
    assert result == {}


def test_get_const_votes_no_args_decorator_skipped():
    """@patch() with no args → skipped (no args branch)."""
    code = "@patch()\ndef test_f(): pass\n"
    refs = [_make_ref("TARGET", "pkg.old.X")]
    result = _get_const_votes_from_rewrite(code, refs)
    assert result == {}


def test_get_const_votes_module_attr_const_name():
    """@patch(module.CONST) style (Attribute node) → const name recorded correctly."""
    # Attribute form used when const is module-aliased after _restore_const_refs.
    code = "@patch(module.TARGET)\ndef test_f(m): pass\n"
    refs = [_make_ref("module.TARGET", "pkg.old.X")]
    result = _get_const_votes_from_rewrite(code, refs)
    # const still present as module.TARGET → no vote entry.
    assert result == {}


def test_get_const_votes_successful_vote():
    """LLM updated the path → new literal collected, vote returned."""
    refs = [_make_ref("TARGET", "pkg.mod.X")]
    code = '@patch("pkg.mod.sub.X")\ndef test_f(m): pass\n'
    result = _get_const_votes_from_rewrite(code, refs)
    assert result == {"pkg.mod.X": "pkg.mod.sub.X"}


def test_get_const_votes_deeply_nested_attr_skipped():
    """@patch(module.sub.CONST) where arg0 is Attribute(Attribute) — falls through
    all elif branches (663->647 coverage: the third elif is False for this form)."""
    # module.sub.CONST: arg0.value is Attribute, not Name → elif at 661 is False;
    # arg0 is not Constant → elif at 663 is False → no match, loop continues.
    code = "@patch(module.sub.CONST)\ndef test_f(m): pass\n"
    refs = [_make_ref("TARGET", "pkg.mod.X")]
    result = _get_const_votes_from_rewrite(code, refs)
    # No string literal collected, TARGET still absent → no vote.
    assert result == {}


@mock_patch(_PATCH_CALL_TOOL)
def test_rewrite_non_participant_casts_keep_old_vote(mock_call, tmp_path):
    """A function that fails the rewrite (edit_failure) still casts a keep-old
    vote, preventing a const from being updated when only one of two users
    successfully renamed it.

    Scenario:
      test_a: classify → rename X → after.X; verify OK → string_swap_results.
      test_b: classify → needs_rewrite → rewrite → LLM returns None (failure)
              → edit_failure → NOT in string_swap_results.

    Without the keep-old fix: X proposals = {"after.X"} (single) → const updated.
    With the keep-old fix:    X proposals = {"after.X", "old.X"} → conflicting →
                              test_a inlined, const definition unchanged.
    """
    src = (
        'TARGET = "crispen.before.X"\n'
        "\n"
        "@patch(TARGET)\n"
        "def test_a(mock_x):\n"
        "    pass\n"
        "\n"
        "@patch(TARGET)\n"
        "def test_b(mock_x):\n"
        "    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # test_a: classify → rename → verify OK.
    # test_b: classify → needs_rewrite → rewrite attempt → None response (failure).
    mock_call.side_effect = [
        _ok(
            {
                "needs_rewrite": False,
                "patch_renames": {"crispen.before.X": "crispen.after.X"},
            }
        ),
        _ok(_VERIFY_OK),
        _ok({"needs_rewrite": True}),
        LLMCallResult(tool_input=None, elapsed=0.0, input_tokens=0, output_tokens=0),
    ]
    result, changed, cross = _process_file_source(
        src,
        {"crispen.before.X"},
        "ctx",
        MagicMock(),
        _CFG,
        1,
        scan_file=scan,
    )
    # test_b failed → keep-old vote → conflicting → const NOT updated.
    assert 'TARGET = "crispen.before.X"' in result
    # test_a's decorator inlined individually.
    assert '@patch("crispen.after.X")' in result


@mock_patch(_PATCH_CALL_TOOL)
def test_rewrite_non_participant_cross_file_ref_skipped(mock_call, tmp_path):
    """Non-participant with a cross-file const ref: the ref.source_file != scan_file_abs
    branch is False, so no same-file keep-old vote is cast (3171->3170 branch).

    test_a: succeeds (in string_swap_results).
    test_b: fails (not in string_swap_results).  test_b's const is defined in
            helpers.py (cross-file) so the keep-old loop skips it — no
            same_file_proposals entry for that ref.
    """
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "crispen.before.X"\n', encoding="utf-8")
    src = (
        "from .helpers import TARGET\n"
        "\n"
        "@patch(TARGET)\n"
        "def test_a(mock_x):\n"
        "    pass\n"
        "\n"
        "@patch(TARGET)\n"
        "def test_b(mock_x):\n"
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
        _ok({"needs_rewrite": True}),
        LLMCallResult(tool_input=None, elapsed=0.0, input_tokens=0, output_tokens=0),
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
    # Cross-file ref → no same-file conflict → cross updated (test_a's rename wins).
    helpers_abs = str(helpers.resolve())
    assert helpers_abs in cross
    assert cross[helpers_abs] == {"crispen.before.X": "crispen.after.X"}


def test_callgraph_const_ref_no_scan_file_skips_keep_old(tmp_path):
    """scan_file=None → scan_file_abs="" (falsy) → the keep-old block is skipped
    entirely (3420->3426 branch). BFS ambiguous functions don't cast any vote.
    The string literal path is still updated normally.
    """
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
            "conflict.py": "from external import use_fn\ndef resolve(): use_fn()\n",
        },
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '@patch("pkg.orig.use_fn")\n'
        "def test_f(m):\n"
        "    helper()\n"
        "    resolve()\n"
    )
    # scan_file=None → scan_file_abs="" → keep-old block skipped; string literal
    # unchanged because BFS is ambiguous (no resolved result).
    result, changed, _unresolved = _callgraph_update_file(
        test_src,
        {"pkg.orig.use_fn"},
        [ctx],
        scan_file=None,
        index=None,
    )
    assert not changed


def test_callgraph_const_ref_ambiguous_casts_keep_old_vote(tmp_path):
    """When BFS finds multiple candidates for a const-backed path (ambiguous),
    the function casts a keep-old vote so a shared constant isn't updated to a
    value that is wrong for the ambiguous function.

    test_a: calls helper() → placement (single BFS candidate) → vote "placement".
    test_b: calls helper() + resolve() → placement AND conflict (2 BFS candidates
            for use_fn) → ambiguous → keep-old vote.
    Proposals for _PATCH_USE: {"pkg.placement.use_fn", "pkg.orig.use_fn"} → conflict
    → constant NOT updated; test_a gets its decorator inlined individually.
    """
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="from external import use_fn\n",
        modified_source="",
        new_files={
            "placement.py": "from external import use_fn\ndef helper(): use_fn()\n",
            "conflict.py": "from external import use_fn\ndef resolve(): use_fn()\n",
        },
        new_module_paths={
            "placement.py": "pkg.placement",
            "conflict.py": "pkg.conflict",
        },
        entity_to_target={},
        forking_old_paths={"pkg.orig.use_fn"},
    )
    test_src = (
        "from pkg.placement import helper\n"
        "from pkg.conflict import resolve\n"
        '_PATCH_USE = "pkg.orig.use_fn"\n'
        "@patch(_PATCH_USE)\n"
        "def test_a(m):\n"
        "    helper()\n"
        "\n"
        "@patch(_PATCH_USE)\n"
        "def test_b(m):\n"
        "    helper()\n"
        "    resolve()\n"
    )
    scan = str(tmp_path / "test_foo.py")
    index = _make_cuf_index(scan, test_src)
    result, changed, _unresolved = _callgraph_update_file(
        test_src, {"pkg.orig.use_fn"}, [ctx], scan_file=scan, index=index
    )
    # test_b is ambiguous → keep-old vote → conflict with test_a's rename vote.
    # Constant definition must NOT be updated.
    assert '_PATCH_USE = "pkg.orig.use_fn"' in result
    # test_a's decorator IS inlined (it had a resolved rename).
    assert '@patch("pkg.placement.use_fn")' in result
