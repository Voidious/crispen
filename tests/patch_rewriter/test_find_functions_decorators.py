from __future__ import annotations
from crispen.patch_rewriter import (
    _find_test_functions_to_update,
    _patch_strings_in_text,
)


def test_find_empty_old_paths():
    src = '@patch("crispen.before.X")\ndef test_f(): pass\n'
    assert _find_test_functions_to_update(src, set()) == []


def test_find_parse_error():
    assert _find_test_functions_to_update("def f(:\n", {"crispen.before.X"}) == []


def test_find_no_match():
    src = '@patch("other.mod.Y")\ndef test_f(): pass\n'
    assert _find_test_functions_to_update(src, {"crispen.before.X"}) == []


def test_find_match_exact():
    src = '@patch("crispen.before.X")\ndef test_f(): pass\n'
    result = _find_test_functions_to_update(src, {"crispen.before.X"})
    assert len(result) == 1
    assert result[0].function_name == "test_f"
    assert "crispen.before.X" in result[0].old_patch_paths


def test_find_match_prefix():
    src = '@patch("crispen.before.X.method")\ndef test_f(): pass\n'
    result = _find_test_functions_to_update(src, {"crispen.before.X"})
    assert len(result) == 1
    assert "crispen.before.X.method" in result[0].old_patch_paths


def test_find_not_a_call_decorator():
    # @patch used as a bare name (no parentheses), not a Call node.
    src = "@patch\ndef test_f(): pass\n"
    assert _find_test_functions_to_update(src, {"crispen.before.X"}) == []


def test_find_no_args():
    src = "@patch()\ndef test_f(): pass\n"
    assert _find_test_functions_to_update(src, {"crispen.before.X"}) == []


def test_find_arg_not_simple_string():
    # @patch(some_variable) — first arg is a Name, not a SimpleString.
    src = "@patch(some_var)\ndef test_f(): pass\n"
    assert _find_test_functions_to_update(src, {"crispen.before.X"}) == []


def test_find_prefixed_string():
    # b"..." — raw[0] is 'b', not a quote character.
    src = '@patch(b"crispen.before.X")\ndef test_f(): pass\n'
    assert _find_test_functions_to_update(src, {"crispen.before.X"}) == []


def test_find_triple_quoted():
    src = '@patch("""crispen.before.X""")\ndef test_f(): pass\n'
    assert _find_test_functions_to_update(src, {"crispen.before.X"}) == []


def test_find_not_patch_name():
    # @decorate("crispen.before.X") — attribute name is not "patch".
    src = '@decorate("crispen.before.X")\ndef test_f(): pass\n'
    assert _find_test_functions_to_update(src, {"crispen.before.X"}) == []


def test_find_attribute_patch():
    # @mock.patch("crispen.before.X") — Attribute form.
    src = '@mock.patch("crispen.before.X")\ndef test_f(): pass\n'
    result = _find_test_functions_to_update(src, {"crispen.before.X"})
    assert len(result) == 1
    assert result[0].function_name == "test_f"


def test_find_multiple_functions():
    src = (
        '@patch("crispen.before.X")\ndef test_a(): pass\n\n'
        '@patch("crispen.before.Y")\ndef test_b(): pass\n'
    )
    result = _find_test_functions_to_update(
        src, {"crispen.before.X", "crispen.before.Y"}
    )
    assert {f.function_name for f in result} == {"test_a", "test_b"}


def test_find_full_text_includes_decorator():
    src = '@patch("crispen.before.X")\ndef test_f():\n    pass\n'
    result = _find_test_functions_to_update(src, {"crispen.before.X"})
    assert '@patch("crispen.before.X")' in result[0].full_text
    assert "def test_f" in result[0].full_text


def test_find_start_end_lines():
    # line 1: # header, line 2: @patch..., line 3: def test_f, line 4: pass
    src = "# header\n" '@patch("crispen.before.X")\n' "def test_f():\n" "    pass\n"
    result = _find_test_functions_to_update(src, {"crispen.before.X"})
    assert result[0].start_line == 2  # @patch line (first decorator)
    assert result[0].end_line == 4  # last line of body


def test_find_body_with_patch_no_decorator():
    # Function has no @patch decorator but uses ``with patch(...)`` in the body.
    src = (
        "def test_f():\n" '    with patch("crispen.before.X") as m:\n' "        pass\n"
    )
    result = _find_test_functions_to_update(src, {"crispen.before.X"})
    assert len(result) == 1
    assert result[0].function_name == "test_f"
    assert "crispen.before.X" in result[0].old_patch_paths
    # start_line should be the ``def`` line (no decorators).
    assert result[0].start_line == 1


def test_find_body_with_patch_combined_with_decorator():
    # Function has both an @patch decorator and a body-level with patch(...).
    src = (
        '@patch("crispen.before.Y")\n'
        "def test_f(mock_y):\n"
        '    with patch("crispen.before.X") as m:\n'
        "        pass\n"
    )
    result = _find_test_functions_to_update(
        src, {"crispen.before.X", "crispen.before.Y"}
    )
    assert len(result) == 1
    paths = result[0].old_patch_paths
    assert "crispen.before.X" in paths
    assert "crispen.before.Y" in paths


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


def test_patch_strings_in_text_decorator():
    text = '@patch("pkg.mod.A")\ndef test_f(m): pass\n'
    assert _patch_strings_in_text(text) == {"pkg.mod.A"}


def test_patch_strings_in_text_attribute_decorator():
    text = '@mock.patch("pkg.mod.B")\ndef test_f(m): pass\n'
    assert _patch_strings_in_text(text) == {"pkg.mod.B"}


def test_patch_strings_in_text_multiple():
    text = (
        '@patch("pkg.mod.A")\n' '@mock.patch("pkg.mod.B")\n' "def test_f(a, b): pass\n"
    )
    assert _patch_strings_in_text(text) == {"pkg.mod.A", "pkg.mod.B"}


def test_patch_strings_in_text_empty():
    assert _patch_strings_in_text("def test_f(): pass\n") == set()
