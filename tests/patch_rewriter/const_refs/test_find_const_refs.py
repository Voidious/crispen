from __future__ import annotations
from crispen.patch_rewriter import (
    _build_attr_const_map,
    _build_const_map,
    _build_local_const_map,
    _find_test_functions_to_update,
    _resolve_import_to_file,
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


def test_find_const_ref_same_file(tmp_path):
    """@patch(CONST) where CONST is in the same file → collected, substituted."""
    src = (
        'TARGET = "crispen.before.X"\n\n@patch(TARGET)\ndef test_f(mock_x):\n    pass\n'
    )
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(src, {"crispen.before.X"}, scan_file=scan)
    assert len(result) == 1
    assert result[0].function_name == "test_f"
    # full_text sent to LLM has the value inlined
    assert '"crispen.before.X"' in result[0].full_text
    assert "TARGET" not in result[0].full_text
    # const_ref recorded
    assert len(result[0].const_refs) == 1
    assert result[0].const_refs[0].const_name == "TARGET"
    assert result[0].const_refs[0].resolved_value == "crispen.before.X"
    assert result[0].const_refs[0].patch_dec_idx == 0


def test_find_const_ref_not_in_map_not_collected(tmp_path):
    """@patch(UNRESOLVED) where name not in const_map → not collected."""
    src = "@patch(UNRESOLVED)\ndef test_f(mock): pass\n"
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(src, {"crispen.before.X"}, scan_file=scan)
    assert result == []


def test_find_const_ref_value_no_match(tmp_path):
    """@patch(CONST) where const value doesn't match old_paths → not collected."""
    src = 'TARGET = "other.mod.Y"\n\n@patch(TARGET)\ndef test_f(mock): pass\n'
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(src, {"crispen.before.X"}, scan_file=scan)
    assert result == []


def test_find_mix_literal_and_const(tmp_path):
    """Function with both a literal @patch and a const @patch → both collected."""
    src = (
        'TARGET = "crispen.before.X"\n\n'
        '@patch("crispen.before.X")\n'
        "@patch(TARGET)\n"
        "def test_f(m1, m2):\n    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(src, {"crispen.before.X"}, scan_file=scan)
    assert len(result) == 1
    assert len(result[0].old_patch_paths) == 2
    assert len(result[0].const_refs) == 1
    # patch_dec_idx of the const ref is 1 (second @patch decorator)
    assert result[0].const_refs[0].patch_dec_idx == 1


def test_find_non_matching_decorator_split_into_stable(tmp_path):
    """Non-matching decorators go to stable_patch_paths, not old_patch_paths.

    A test that patches get_api_key (already correct) and call_with_tool
    (forking, needs rewrite) should have only call_with_tool in old_patch_paths
    and get_api_key in stable_patch_paths so the LLM is not asked to evaluate
    the already-correct path.
    """
    src = (
        'KEY = "crispen.mod.get_api_key"\n'
        'CALL = "crispen.mod.call_with_tool"\n\n'
        "@patch(KEY)\n"
        "@patch(CALL)\n"
        "def test_f(mock_call, mock_key):\n    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    # Only CALL's value is in old_paths; KEY's value is already correct.
    result = _find_test_functions_to_update(
        src, {"crispen.mod.call_with_tool"}, scan_file=scan
    )
    assert len(result) == 1
    # Forking path goes to old_patch_paths only.
    assert result[0].old_patch_paths == ["crispen.mod.call_with_tool"]
    # Already-correct path goes to stable_patch_paths.
    assert result[0].stable_patch_paths == ["crispen.mod.get_api_key"]
    # Both const refs must be recorded so their definitions can be updated.
    assert len(result[0].const_refs) == 2


def test_find_patch_no_args_increments_idx(tmp_path):
    """@patch() with no args increments patch_dec_idx before the const @patch."""
    src = (
        'TARGET = "crispen.before.X"\n\n'
        "@patch()\n"
        "@patch(TARGET)\n"
        "def test_f(m):\n    pass\n"
    )
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(src, {"crispen.before.X"}, scan_file=scan)
    assert len(result) == 1
    assert result[0].const_refs[0].patch_dec_idx == 1


def test_find_cross_file_const(tmp_path):
    """@patch(CONST) where CONST comes from a relative import → collected."""
    helpers = tmp_path / "helpers.py"
    helpers.write_text('TARGET = "crispen.before.X"\n', encoding="utf-8")
    src = "from .helpers import TARGET\n\n@patch(TARGET)\ndef test_f(mock):\n    pass\n"
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(
        src, {"crispen.before.X"}, scan_file=scan, repo_root=str(tmp_path)
    )
    assert len(result) == 1
    assert result[0].const_refs[0].source_file == str(helpers.resolve())


def test_find_attr_const_ref_collected(tmp_path):
    """@patch(constants.TARGET) where ``import constants`` resolves → collected."""
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('TARGET = "crispen.before.X"\n', encoding="utf-8")
    src = "import constants\n\n@patch(constants.TARGET)\ndef test_f(mock):\n    pass\n"
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(
        src, {"crispen.before.X"}, scan_file=scan, repo_root=str(tmp_path)
    )
    assert len(result) == 1
    assert result[0].function_name == "test_f"
    assert result[0].const_refs[0].const_name == "constants.TARGET"
    assert result[0].const_refs[0].resolved_value == "crispen.before.X"
    assert result[0].const_refs[0].patch_dec_idx == 0
    assert result[0].const_refs[0].source_file == str(constants_file.resolve())
    # LLM sees inlined value, not the attribute access form.
    assert '"crispen.before.X"' in result[0].full_text
    assert "constants.TARGET" not in result[0].full_text


def test_find_attr_const_module_not_in_map(tmp_path):
    """@patch(unknown.TARGET) where module not in attr_const_map → not collected."""
    src = "@patch(unknown.TARGET)\ndef test_f(mock):\n    pass\n"
    scan = str(tmp_path / "test_foo.py")
    # No ``import unknown`` in source → attr_const_map empty → no match.
    result = _find_test_functions_to_update(
        src, {"crispen.before.X"}, scan_file=scan, repo_root=str(tmp_path)
    )
    assert result == []


def test_find_attr_const_attr_not_in_module(tmp_path):
    """@patch(constants.UNKNOWN) where attr not in module constants → not collected."""
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('TARGET = "crispen.before.X"\n', encoding="utf-8")
    src = "import constants\n\n@patch(constants.UNKNOWN)\ndef test_f(mock):\n    pass\n"
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(
        src, {"crispen.before.X"}, scan_file=scan, repo_root=str(tmp_path)
    )
    assert result == []


def test_find_attr_const_value_no_match(tmp_path):
    """@patch(constants.OTHER) where value doesn't match old_paths → not collected."""
    constants_file = tmp_path / "constants.py"
    constants_file.write_text('OTHER = "unrelated.path.Class"\n', encoding="utf-8")
    src = "import constants\n\n@patch(constants.OTHER)\ndef test_f(mock):\n    pass\n"
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(
        src, {"crispen.before.X"}, scan_file=scan, repo_root=str(tmp_path)
    )
    assert result == []


def test_find_attr_multi_level_not_handled(tmp_path):
    """@patch(a.b.c) multi-level attribute (base not Name) → not collected."""
    src = "@patch(a.b.c)\ndef test_f(mock):\n    pass\n"
    scan = str(tmp_path / "test_foo.py")
    result = _find_test_functions_to_update(
        src, {"a.b.c"}, scan_file=scan, repo_root=str(tmp_path)
    )
    assert result == []
