from __future__ import annotations
from unittest.mock import MagicMock, patch as mock_patch
from crispen.llm_client import LLMCallResult
from crispen.patch_rewriter import RewriteAccumulator, apply_patch_rewrite
from .helpers import (
    _CFG,
    _PATCH_CALL_TOOL,
    _PATCH_GET_KEY,
    _VERIFY_OK,
    _make_fl_ctx,
    _ok,
)
from . import helpers


def test_rewrite_empty_contexts():
    msgs = list(apply_patch_rewrite([], {}, "/repo", _CFG))
    assert msgs == []


def test_rewrite_no_forking_paths():
    ctx = _make_fl_ctx(forking_old_paths=set())
    msgs = list(apply_patch_rewrite([ctx], {}, "/repo", _CFG))
    assert msgs == []


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(helpers._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_per_file_update(mock_key, mock_client, mock_call):
    mock_call.side_effect = [
        _ok({"needs_rewrite": False, "patch_renames": {"pkg.big.A": "pkg.sub_a.A"}}),
        _ok(_VERIFY_OK),
    ]
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    per_file = {"/repo/tests/test_big.py": {"source": src, "msgs": []}}
    ctx = _make_fl_ctx()
    msgs = list(apply_patch_rewrite([ctx], per_file, None, _CFG))
    updated = per_file["/repo/tests/test_big.py"]["source"]
    assert "pkg.sub_a.A" in updated
    assert any("patch_update" in m for m in per_file["/repo/tests/test_big.py"]["msgs"])
    assert msgs == []  # no disk messages since repo_root=None


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(helpers._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_no_repo_root_no_disk_scan(mock_key, mock_client, mock_call):
    # repo_root=None → exits after per_file; empty per_file → no LLM calls.
    msgs = list(apply_patch_rewrite([_make_fl_ctx()], {}, None, _CFG))
    assert msgs == []
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(helpers._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_disk_file_update(mock_key, mock_client, mock_call, tmp_path):
    test_file = tmp_path / "test_big.py"
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    test_file.write_text(src, encoding="utf-8")
    mock_call.side_effect = [
        _ok({"needs_rewrite": False, "patch_renames": {"pkg.big.A": "pkg.sub_a.A"}}),
        _ok(_VERIFY_OK),
    ]
    msgs = list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG))
    assert "pkg.sub_a.A" in test_file.read_text(encoding="utf-8")
    assert len(msgs) == 1
    assert "patch_update" in msgs[0]


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(helpers._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_skip_excluded_dir(mock_key, mock_client, mock_call, tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()
    f = venv / "test_big.py"
    f.write_text('@patch("pkg.big.A")\ndef test_f(): pass\n', encoding="utf-8")
    msgs = list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG))
    assert msgs == []
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(helpers._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_skip_per_file_abs(mock_key, mock_client, mock_call, tmp_path):
    # A file already in per_file should NOT be re-processed from disk.
    test_file = tmp_path / "test_big.py"
    test_file.write_text('@patch("pkg.big.A")\ndef test_f(): pass\n', encoding="utf-8")
    original_disk = test_file.read_text(encoding="utf-8")
    # per_file entry uses a source without matching patches (no LLM call needed).
    per_file = {str(test_file): {"source": "# no patches\n", "msgs": []}}
    list(apply_patch_rewrite([_make_fl_ctx()], per_file, str(tmp_path), _CFG))
    # Disk file untouched since it was in per_file_abs.
    assert test_file.read_text(encoding="utf-8") == original_disk
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(helpers._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_oserror_skipped(mock_key, mock_client, mock_call, tmp_path):
    test_file = tmp_path / "test_big.py"
    test_file.write_text('@patch("pkg.big.A")\ndef test_f(): pass\n', encoding="utf-8")
    test_file.chmod(0o000)
    try:
        msgs = list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG))
        assert msgs == []
        mock_call.assert_not_called()
    finally:
        test_file.chmod(0o644)


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(helpers._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_disk_file_no_match_not_updated(
    mock_key, mock_client, mock_call, tmp_path
):
    # Disk file exists but has no matching @patch decorators → changed=False,
    # file is not written, no yield message (covers the `if changed: False` branch).
    test_file = tmp_path / "no_patches.py"
    test_file.write_text("def test_unrelated(): pass\n", encoding="utf-8")
    msgs = list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG))
    assert msgs == []
    assert test_file.read_text(encoding="utf-8") == "def test_unrelated(): pass\n"
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(helpers._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_no_py_files_in_repo(mock_key, mock_client, mock_call, tmp_path):
    # tmp_path has no .py files → disk scan loop body never executes.
    msgs = list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG))
    assert msgs == []
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(helpers._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_acc_tracks_calls_and_files(mock_key, mock_client, mock_call, tmp_path):
    """RewriteAccumulator is populated with call counts and files_updated."""
    test_file = tmp_path / "test_big.py"
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    test_file.write_text(src, encoding="utf-8")
    mock_call.side_effect = [
        LLMCallResult(
            tool_input={
                "needs_rewrite": False,
                "patch_renames": {"pkg.big.A": "pkg.sub_a.A"},
            },
            elapsed=1.5,
            input_tokens=100,
            output_tokens=50,
        ),
        LLMCallResult(
            tool_input=_VERIFY_OK,
            elapsed=0.5,
            input_tokens=80,
            output_tokens=10,
        ),
    ]
    acc = RewriteAccumulator()
    list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG, _acc=acc))
    assert acc.calls == 2
    assert acc.elapsed == 2.0
    assert acc.input_tokens == 180
    assert acc.output_tokens == 60
    assert acc.files_updated == 1


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(helpers._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_acc_per_file_files_updated(mock_key, mock_client, mock_call):
    """files_updated is incremented for in-memory per_file changes."""
    src = '@patch("pkg.big.A")\ndef test_f(mock_a):\n    pass\n'
    per_file = {"/repo/tests/test_big.py": {"source": src, "msgs": []}}
    mock_call.side_effect = [
        _ok({"needs_rewrite": False, "patch_renames": {"pkg.big.A": "pkg.sub_a.A"}}),
        _ok(_VERIFY_OK),
    ]
    acc = RewriteAccumulator()
    list(apply_patch_rewrite([_make_fl_ctx()], per_file, None, _CFG, _acc=acc))
    assert acc.files_updated == 1
