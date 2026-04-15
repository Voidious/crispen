from __future__ import annotations
from crispen.patch_rewriter import (
    _ConstRef,
    _build_attr_const_map,
    _build_const_map,
    _build_local_const_map,
    _resolve_import_to_file,
    _restore_const_refs,
    _substitute_consts_in_func_text,
)


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


def test_resolve_relative_level1_py(tmp_path):
    # from .sub import NAME — sub.py exists
    (tmp_path / "sub.py").write_text("X = 1\n", encoding="utf-8")
    scan = str(tmp_path / "test_foo.py")
    result = _resolve_import_to_file("sub", 1, scan, None)
    assert result == str(tmp_path / "sub.py")


def test_resolve_relative_level1_init(tmp_path):
    # from .pkg import NAME — pkg/__init__.py exists
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    scan = str(tmp_path / "test_foo.py")
    result = _resolve_import_to_file("pkg", 1, scan, None)
    assert result == str(pkg / "__init__.py")


def test_resolve_relative_level1_no_module(tmp_path):
    # from . import NAME — finds __init__.py in same dir
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    scan = str(tmp_path / "test_foo.py")
    result = _resolve_import_to_file(None, 1, scan, None)
    assert result == str(tmp_path / "__init__.py")


def test_resolve_relative_level2(tmp_path):
    # from ..sub import NAME — goes up one level
    parent = tmp_path / "parent"
    parent.mkdir()
    child = parent / "child"
    child.mkdir()
    (parent / "sub.py").write_text("X = 1\n", encoding="utf-8")
    scan = str(child / "test_foo.py")
    result = _resolve_import_to_file("sub", 2, scan, None)
    assert result == str(parent / "sub.py")


def test_resolve_relative_not_found(tmp_path):
    scan = str(tmp_path / "test_foo.py")
    assert _resolve_import_to_file("missing", 1, scan, None) is None


def test_resolve_relative_no_module_no_init(tmp_path):
    scan = str(tmp_path / "test_foo.py")
    assert _resolve_import_to_file(None, 1, scan, None) is None


def test_resolve_absolute_found(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "helpers.py").write_text("X = 1\n", encoding="utf-8")
    scan = str(tmp_path / "tests" / "test_foo.py")
    result = _resolve_import_to_file("mypkg.helpers", 0, scan, str(tmp_path))
    assert result == str(pkg / "helpers.py")


def test_resolve_absolute_no_repo_root(tmp_path):
    scan = str(tmp_path / "test_foo.py")
    assert _resolve_import_to_file("mypkg.helpers", 0, scan, None) is None


def test_resolve_absolute_no_module(tmp_path):
    scan = str(tmp_path / "test_foo.py")
    assert _resolve_import_to_file(None, 0, scan, str(tmp_path)) is None


def test_resolve_absolute_not_found(tmp_path):
    scan = str(tmp_path / "test_foo.py")
    assert _resolve_import_to_file("no.such.module", 0, scan, str(tmp_path)) is None


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


def _make_ref(const_name: str, resolved_value: str) -> _ConstRef:
    return _ConstRef(
        const_name=const_name,
        source_file="/proj/tests/helpers.py",
        resolved_value=resolved_value,
        patch_dec_idx=0,
    )


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
