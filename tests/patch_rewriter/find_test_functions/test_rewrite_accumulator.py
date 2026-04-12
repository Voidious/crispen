from __future__ import annotations
from unittest.mock import MagicMock, patch as mock_patch
from crispen.llm_client import LLMCallResult
from crispen.patch_rewriter import (
    RewriteAccumulator,
    _rewrite_candidates_check,
    apply_patch_rewrite,
)
from .. import helpers
from ..helpers import (
    _CFG,
    _PATCH_CALL_TOOL,
    _PATCH_GET_KEY,
    _VERIFY_OK,
    _make_fl_ctx,
    _ok,
)


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


@mock_patch(_PATCH_CALL_TOOL)
@mock_patch(helpers._PATCH_MAKE_CLIENT, return_value=MagicMock())
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
@mock_patch(helpers._PATCH_MAKE_CLIENT, return_value=MagicMock())
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
@mock_patch(helpers._PATCH_MAKE_CLIENT, return_value=MagicMock())
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
@mock_patch(helpers._PATCH_MAKE_CLIENT, return_value=MagicMock())
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
