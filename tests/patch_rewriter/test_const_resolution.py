from __future__ import annotations
from crispen.patch_rewriter import (
    _apply_cross_file_const_updates,
    _build_attr_const_map,
    _build_const_map,
    _build_local_const_map,
    _restore_const_refs,
    _substitute_consts_in_func_text,
)
from .const_resolution import _make_ref


def test_local_const_map_string_assignment():
    src = 'TARGET = "myapp.service.MyClass"\n'
    result = _build_local_const_map(src)
    assert result == {"TARGET": "myapp.service.MyClass"}


def test_local_const_map_non_string_excluded():
    src = "TARGET = 42\n"
    assert _build_local_const_map(src) == {}


def test_local_const_map_multi_target_excluded():
    # a = b = "value" has two targets → not included.
    src = 'a = b = "value"\n'
    assert _build_local_const_map(src) == {}


def test_local_const_map_syntax_error():
    assert _build_local_const_map("def f(:\n") == {}


def test_local_const_map_empty_source():
    assert _build_local_const_map("") == {}


def test_local_const_map_last_wins():
    src = 'X = "first"\nX = "second"\n'
    assert _build_local_const_map(src)["X"] == "second"


def test_local_const_map_annotated_assignment():
    src = 'TARGET: str = "myapp.service.MyClass"\n'
    assert _build_local_const_map(src) == {"TARGET": "myapp.service.MyClass"}


def test_local_const_map_annotated_non_string_excluded():
    src = "TARGET: int = 42\n"
    assert _build_local_const_map(src) == {}


def test_local_const_map_annotated_no_value_excluded():
    # Bare annotation with no value: ``TARGET: str`` — ast.AnnAssign with value=None
    src = "TARGET: str\n"
    assert _build_local_const_map(src) == {}


def test_build_const_map_same_file(tmp_path):
    src = 'TARGET = "myapp.service.MyClass"\n'
    scan = str(tmp_path / "test_foo.py")
    result = _build_const_map(src, scan, None)
    val, def_file = result["TARGET"]
    assert val == "myapp.service.MyClass"
    assert def_file == str((tmp_path / "test_foo.py").resolve())


def test_build_const_map_cross_file(tmp_path):
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "myapp.service.MyClass"\n', encoding="utf-8")
    src = "from .helpers import TARGET\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_const_map(src, scan, None)
    val, def_file = result["TARGET"]
    assert val == "myapp.service.MyClass"
    assert def_file == str(helpers.resolve())


def test_build_const_map_alias(tmp_path):
    helpers = tmp_path / "helpers.py"
    helpers.write_text('X = "myapp.service.MyClass"\n', encoding="utf-8")
    src = "from .helpers import X as MY_TARGET\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_const_map(src, scan, None)
    assert "MY_TARGET" in result
    assert result["MY_TARGET"][0] == "myapp.service.MyClass"


def test_build_const_map_local_priority(tmp_path):
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "imported.value"\n', encoding="utf-8")
    src = 'TARGET = "local.value"\nfrom .helpers import TARGET\n'
    scan = str(tmp_path / "test_foo.py")
    result = _build_const_map(src, scan, None)
    assert result["TARGET"][0] == "local.value"


def test_build_const_map_star_import_skipped(tmp_path):
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "myapp.service.MyClass"\n', encoding="utf-8")
    src = "from .helpers import *\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_const_map(src, scan, None)
    assert result == {}


def test_build_const_map_import_file_not_found(tmp_path):
    src = "from .missing import TARGET\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_const_map(src, scan, None)
    assert result == {}


def test_build_const_map_import_oserror(tmp_path):
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "val"\n', encoding="utf-8")
    helpers.chmod(0o000)
    try:
        src = "from .helpers import TARGET\n"
        scan = str(tmp_path / "test_foo.py")
        result = _build_const_map(src, scan, None)
        assert result == {}
    finally:
        helpers.chmod(0o644)


def test_build_const_map_syntax_error():
    result = _build_const_map("def f(:\n", "/some/file.py", None)
    assert result == {}


def test_build_const_map_no_const_in_import(tmp_path):
    helpers = tmp_path / "helpers.py"
    helpers.write_text("def some_func(): pass\n", encoding="utf-8")
    src = "from .helpers import some_func\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_const_map(src, scan, None)
    assert result == {}


def test_build_attr_const_map_basic(tmp_path):
    """``import constants`` resolves string constants from the module file."""
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('TARGET = "myapp.service.MyClass"\n', encoding="utf-8")
    src = "import constants\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_attr_const_map(src, scan, str(tmp_path))
    assert "constants" in result
    val, def_file = result["constants"]["TARGET"]
    assert val == "myapp.service.MyClass"
    assert def_file == str(constants_file.resolve())


def test_build_attr_const_map_with_alias(tmp_path):
    """``import pkg.constants as C`` maps alias ``C`` to module constants."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    constants_file = pkg / "constants.py"
    constants_file.write_text('TARGET = "myapp.svc.MyClass"\n', encoding="utf-8")
    src = "import pkg.constants as C\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_attr_const_map(src, scan, str(tmp_path))
    assert "C" in result
    assert result["C"]["TARGET"][0] == "myapp.svc.MyClass"


def test_build_attr_const_map_no_file(tmp_path):
    """Import that doesn't resolve to a file → skipped, empty result."""
    src = "import missing_module\n"
    scan = str(tmp_path / "test_foo.py")
    result = _build_attr_const_map(src, scan, str(tmp_path))
    assert result == {}


def test_build_attr_const_map_oserror(tmp_path):
    """Module file exists but is unreadable → skipped."""
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('TARGET = "val"\n', encoding="utf-8")
    constants_file.chmod(0o000)
    try:
        src = "import constants\n"
        scan = str(tmp_path / "test_foo.py")
        result = _build_attr_const_map(src, scan, str(tmp_path))
        assert result == {}
    finally:
        constants_file.chmod(0o644)


def test_build_attr_const_map_syntax_error():
    """SyntaxError in source → empty result."""
    assert _build_attr_const_map("def f(:\n", "/some/file.py", None) == {}


def test_build_attr_const_map_non_import_skipped(tmp_path):
    """Non-``import`` statements (from-imports, assignments) are skipped."""
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('TARGET = "val"\n', encoding="utf-8")
    # Only a from-import and an assignment; no plain ``import`` → empty.
    src = 'from .constants import TARGET\nX = "y"\n'
    scan = str(tmp_path / "test_foo.py")
    result = _build_attr_const_map(src, scan, str(tmp_path))
    assert result == {}


def test_substitute_replaces_const():
    code = "@patch(TARGET)\ndef test_f(mock): pass\n"
    result = _substitute_consts_in_func_text(code, {"TARGET": "myapp.svc.MyClass"})
    assert '@patch("myapp.svc.MyClass")' in result
    assert "TARGET" not in result


def test_substitute_no_subs_unchanged():
    code = "@patch(TARGET)\ndef test_f(mock): pass\n"
    assert _substitute_consts_in_func_text(code, {}) == code


def test_substitute_parse_error_returns_original():
    code = "def f(:\n"
    assert _substitute_consts_in_func_text(code, {"X": "val"}) == code


def test_substitute_non_patch_call_unchanged():
    # other_func(TARGET) inside the body is not a patch call → left as-is.
    code = "@patch(TARGET)\ndef test_f(mock):\n    other_func(TARGET)\n"
    result = _substitute_consts_in_func_text(code, {"TARGET": "myapp.svc.MyClass"})
    assert '@patch("myapp.svc.MyClass")' in result
    assert "other_func(TARGET)" in result


def test_substitute_name_not_in_subs_unchanged():
    # @patch(OTHER) where OTHER is not in substitutions → left as-is (line 311).
    code = "@patch(TARGET)\n@patch(OTHER)\ndef test_f(m1, m2):\n    pass\n"
    result = _substitute_consts_in_func_text(code, {"TARGET": "myapp.svc.MyClass"})
    assert '@patch("myapp.svc.MyClass")' in result
    assert "@patch(OTHER)" in result


def test_substitute_attr_in_subs():
    """@patch(module.CONSTANT) with dotted key in subs → substituted."""
    code = "@patch(constants.TARGET)\ndef test_f(mock):\n    pass\n"
    result = _substitute_consts_in_func_text(
        code, {"constants.TARGET": "myapp.svc.MyClass"}
    )
    assert '@patch("myapp.svc.MyClass")' in result
    assert "constants.TARGET" not in result


def test_substitute_attr_not_in_subs():
    """@patch(constants.OTHER) where dotted key not in subs → unchanged."""
    code = (
        "@patch(constants.TARGET)\n"
        "@patch(constants.OTHER)\n"
        "def test_f(m1, m2):\n    pass\n"
    )
    result = _substitute_consts_in_func_text(
        code, {"constants.TARGET": "myapp.svc.MyClass"}
    )
    assert '@patch("myapp.svc.MyClass")' in result
    assert "@patch(constants.OTHER)" in result


def test_substitute_attr_non_name_base():
    """@patch(a.b.c) where base is Attribute (not Name) → else branch, unchanged."""
    code = "@patch(a.b.c)\ndef test_f(mock):\n    pass\n"
    result = _substitute_consts_in_func_text(code, {"a.b.c": "should.not.replace"})
    assert "@patch(a.b.c)" in result


def test_restore_reverts_unchanged_plain_name():
    """@patch("value") whose value matches a const_ref → reverted to @patch(NAME)."""
    code = '@patch("myapp.svc.MyClass")\ndef test_f(mock): pass\n'
    refs = [_make_ref("TARGET", "myapp.svc.MyClass")]
    result = _restore_const_refs(code, refs)
    assert "@patch(TARGET)" in result
    assert '"myapp.svc.MyClass"' not in result


def test_restore_reverts_unchanged_attr_form():
    """@patch("value") matching module.CONST ref → reverted to @patch(module.CONST)."""
    code = '@patch("myapp.svc.MyClass")\ndef test_f(mock): pass\n'
    refs = [_make_ref("constants.TARGET", "myapp.svc.MyClass")]
    result = _restore_const_refs(code, refs)
    assert "@patch(constants.TARGET)" in result
    assert '"myapp.svc.MyClass"' not in result


def test_restore_leaves_changed_value_as_literal():
    """@patch("new.value") where new.value is not in const_refs → kept as literal."""
    code = '@patch("myapp.new.MyClass")\ndef test_f(mock): pass\n'
    refs = [_make_ref("TARGET", "myapp.old.MyClass")]
    result = _restore_const_refs(code, refs)
    assert '@patch("myapp.new.MyClass")' in result


def test_restore_empty_refs_unchanged():
    """No const_refs → text returned as-is."""
    code = '@patch("myapp.svc.MyClass")\ndef test_f(mock): pass\n'
    assert _restore_const_refs(code, []) == code


def test_restore_parse_error_returns_original():
    """Unparseable text → original returned unchanged."""
    code = "def f(:\n"
    refs = [_make_ref("TARGET", "myapp.svc.X")]
    assert _restore_const_refs(code, refs) == code


def test_restore_empty_args_patch_unchanged():
    """@patch() with no args → left as-is."""
    code = "@patch()\ndef test_f(): pass\n"
    refs = [_make_ref("TARGET", "myapp.svc.MyClass")]
    assert _restore_const_refs(code, refs) == code


def test_restore_non_string_arg_unchanged():
    """@patch(NAME) where arg is a Name node (not SimpleString) → left as-is."""
    code = "@patch(OTHER_NAME)\ndef test_f(mock): pass\n"
    refs = [_make_ref("TARGET", "myapp.svc.MyClass")]
    result = _restore_const_refs(code, refs)
    assert "@patch(OTHER_NAME)" in result


def test_restore_non_patch_call_untouched():
    """other_func("value") is not a patch call → left as-is."""
    code = (
        '@patch("myapp.svc.MyClass")\n'
        "def test_f(mock):\n"
        '    other_func("myapp.svc.OtherClass")\n'
    )
    refs = [
        _make_ref("TARGET", "myapp.svc.MyClass"),
        _make_ref("OTHER", "myapp.svc.OtherClass"),
    ]
    result = _restore_const_refs(code, refs)
    assert "@patch(TARGET)" in result
    assert 'other_func("myapp.svc.OtherClass")' in result


def test_restore_single_quote_string():
    """SimpleString with single quotes → still reverted."""
    code = "@patch('myapp.svc.MyClass')\ndef test_f(mock): pass\n"
    refs = [_make_ref("TARGET", "myapp.svc.MyClass")]
    result = _restore_const_refs(code, refs)
    assert "@patch(TARGET)" in result


def test_restore_partial_revert_mixed():
    """One decorator changed, one unchanged → only unchanged one is reverted."""
    code = (
        '@patch("myapp.svc.MyClass")\n'
        '@patch("myapp.new.Y")\n'
        "def test_f(m1, m2): pass\n"
    )
    # MyClass unchanged (should revert), Y was updated by LLM (keep literal)
    refs = [
        _make_ref("TARGET", "myapp.svc.MyClass"),
        _make_ref("Y_CONST", "myapp.old.Y"),  # old value; new value won't match
    ]
    result = _restore_const_refs(code, refs)
    assert "@patch(TARGET)" in result
    assert '@patch("myapp.new.Y")' in result


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
