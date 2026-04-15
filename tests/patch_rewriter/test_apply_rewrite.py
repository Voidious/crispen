from __future__ import annotations
from unittest.mock import MagicMock, patch as mock_patch
from crispen.llm_client import LLMCallResult
from crispen.patch_rewriter import (
    RewriteAccumulator,
    _splice_function,
    apply_patch_rewrite,
)
from .context_builders import (
    _CFG,
    _PATCH_CALL_TOOL,
    _PATCH_GET_KEY,
    _VERIFY_OK,
    _make_fl_ctx,
    _ok,
)
from . import process_basic


def test_splice_function_basic():
    source = "line1\nline2\nline3\nline4\n"
    result = _splice_function(source, 2, 3, "new2\nnew3\n")
    assert result == "line1\nnew2\nnew3\nline4\n"


def test_splice_function_single_line():
    source = "line1\nline2\nline3\n"
    result = _splice_function(source, 2, 2, "replacement\n")
    assert result == "line1\nreplacement\nline3\n"


def test_splice_function_size_change():
    # Replace 1 line with 3 lines.
    source = "a\nb\nc\n"
    result = _splice_function(source, 2, 2, "x\ny\nz\n")
    assert result == "a\nx\ny\nz\nc\n"


def test_splice_function_no_trailing_newline():
    # new_func_text without trailing newline gets one added.
    source = "a\nb\nc\n"
    result = _splice_function(source, 2, 2, "replacement")
    assert result == "a\nreplacement\nc\n"


def test_splice_function_empty_new_text():
    # Empty string: no trailing newline added (falsy check), splitlines gives [].
    source = "a\nb\nc\n"
    result = _splice_function(source, 2, 2, "")
    assert result == "a\nc\n"


def test_rewrite_empty_contexts():
    msgs = list(apply_patch_rewrite([], {}, "/repo", _CFG))
    assert msgs == []


def test_rewrite_no_forking_paths():
    ctx = _make_fl_ctx(forking_old_paths=set())
    msgs = list(apply_patch_rewrite([ctx], {}, "/repo", _CFG))
    assert msgs == []


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(process_basic._PATCH_MAKE_CLIENT, return_value=MagicMock())
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
@mock_patch(process_basic._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_no_repo_root_no_disk_scan(mock_key, mock_client, mock_call):
    # repo_root=None → exits after per_file; empty per_file → no LLM calls.
    msgs = list(apply_patch_rewrite([_make_fl_ctx()], {}, None, _CFG))
    assert msgs == []
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(process_basic._PATCH_MAKE_CLIENT, return_value=MagicMock())
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
@mock_patch(process_basic._PATCH_MAKE_CLIENT, return_value=MagicMock())
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
@mock_patch(process_basic._PATCH_MAKE_CLIENT, return_value=MagicMock())
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
@mock_patch(process_basic._PATCH_MAKE_CLIENT, return_value=MagicMock())
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
@mock_patch(process_basic._PATCH_MAKE_CLIENT, return_value=MagicMock())
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
@mock_patch(process_basic._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_no_py_files_in_repo(mock_key, mock_client, mock_call, tmp_path):
    # tmp_path has no .py files → disk scan loop body never executes.
    msgs = list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG))
    assert msgs == []
    mock_call.assert_not_called()


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(process_basic._PATCH_MAKE_CLIENT, return_value=MagicMock())
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
@mock_patch(process_basic._PATCH_MAKE_CLIENT, return_value=MagicMock())
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


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(process_basic._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_cross_file_const_per_file(mock_key, mock_client, mock_call, tmp_path):
    """Cross-file const whose source is in per_file gets updated in-memory."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "pkg.big.A"\n', encoding="utf-8")
    # test_foo.py imports TARGET from helpers and uses it in @patch.
    test_src = (
        "from .helpers import TARGET\n\n@patch(TARGET)\ndef test_f(m):\n    pass\n"
    )
    helpers_state = {"source": 'TARGET = "pkg.big.A"\n', "msgs": []}
    per_file = {
        str(tmp_path / "test_foo.py"): {"source": test_src, "msgs": []},
        str(helpers): helpers_state,
    }
    mock_call.side_effect = [
        _ok({"needs_rewrite": False, "patch_renames": {"pkg.big.A": "pkg.sub_a.A"}}),
        _ok(_VERIFY_OK),
    ]
    list(apply_patch_rewrite([_make_fl_ctx()], per_file, None, _CFG))
    # The constant definition in helpers.py (per_file entry) should be updated.
    assert '"pkg.sub_a.A"' in helpers_state["source"]


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(process_basic._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_cross_file_const_disk(mock_key, mock_client, mock_call, tmp_path):
    """Cross-file const on disk (not in per_file) gets written directly."""
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "pkg.big.A"\n', encoding="utf-8")
    test_src = (
        "from .helpers import TARGET\n\n@patch(TARGET)\ndef test_f(m):\n    pass\n"
    )
    test_file = tmp_path / "test_foo.py"
    test_file.write_text(test_src, encoding="utf-8")
    mock_call.side_effect = [
        _ok({"needs_rewrite": False, "patch_renames": {"pkg.big.A": "pkg.sub_a.A"}}),
        _ok(_VERIFY_OK),
    ]
    list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG))
    # The constant definition on disk should be updated.
    assert '"pkg.sub_a.A"' in helpers.read_text(encoding="utf-8")


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(process_basic._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_cross_file_const_per_file_acc(
    mock_key, mock_client, mock_call, tmp_path
):
    """_acc.files_updated is incremented when a cross-file const in per_file changes."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "pkg.big.A"\n', encoding="utf-8")
    test_src = (
        "from .helpers import TARGET\n\n@patch(TARGET)\ndef test_f(m):\n    pass\n"
    )
    helpers_state = {"source": 'TARGET = "pkg.big.A"\n', "msgs": []}
    per_file = {
        str(tmp_path / "test_foo.py"): {"source": test_src, "msgs": []},
        str(helpers): helpers_state,
    }
    mock_call.side_effect = [
        _ok({"needs_rewrite": False, "patch_renames": {"pkg.big.A": "pkg.sub_a.A"}}),
        _ok(_VERIFY_OK),
    ]
    acc = RewriteAccumulator()
    list(apply_patch_rewrite([_make_fl_ctx()], per_file, None, _CFG, _acc=acc))
    # One file_updated for the test_foo.py source change, one for helpers const.
    assert acc.files_updated >= 1
    assert '"pkg.sub_a.A"' in helpers_state["source"]


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(process_basic._PATCH_MAKE_CLIENT, return_value=MagicMock())
@mock_patch(_PATCH_GET_KEY, return_value="fake_key")
def test_rewrite_cross_file_const_disk_acc(mock_key, mock_client, mock_call, tmp_path):
    """_acc.files_updated is incremented when a cross-file const on disk changes."""
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "pkg.big.A"\n', encoding="utf-8")
    test_src = (
        "from .helpers import TARGET\n\n@patch(TARGET)\ndef test_f(m):\n    pass\n"
    )
    test_file = tmp_path / "test_foo.py"
    test_file.write_text(test_src, encoding="utf-8")
    mock_call.side_effect = [
        _ok({"needs_rewrite": False, "patch_renames": {"pkg.big.A": "pkg.sub_a.A"}}),
        _ok(_VERIFY_OK),
    ]
    acc = RewriteAccumulator()
    list(apply_patch_rewrite([_make_fl_ctx()], {}, str(tmp_path), _CFG, _acc=acc))
    assert acc.files_updated >= 1
