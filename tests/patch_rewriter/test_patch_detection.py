from __future__ import annotations
from crispen.patch_rewriter import (
    _compiles,
    _find_test_functions_to_update,
    _find_with_patch_paths_in_body,
    _is_patch_call,
    _matches_any,
    _patch_strings_in_text,
)
import libcst as cst


def test_is_patch_call_name_match():
    call_node = cst.parse_expression('patch("foo")')
    assert _is_patch_call(call_node) is True


def test_is_patch_call_attribute_match():
    call_node = cst.parse_expression('mock.patch("foo")')
    assert _is_patch_call(call_node) is True


def test_is_patch_call_other_name():
    call_node = cst.parse_expression('other("foo")')
    assert _is_patch_call(call_node) is False


def test_matches_any_exact():
    assert _matches_any("a.b.C", {"a.b.C"}) is True


def test_matches_any_prefix():
    assert _matches_any("a.b.C.method", {"a.b.C"}) is True


def test_matches_any_near_miss():
    # "a.b.CExtra" should NOT match "a.b.C"
    assert _matches_any("a.b.CExtra", {"a.b.C"}) is False


def test_matches_any_no_match():
    assert _matches_any("x.y.Z", {"a.b.C"}) is False


def test_compiles_valid():
    assert _compiles("x = 1\n") is True


def test_compiles_invalid():
    assert _compiles("def f(:\n    pass\n") is False


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


def test_body_scan_syntax_error():
    assert _find_with_patch_paths_in_body("def f(:\n", {"old.X"}, {}, {}) == []


def test_body_scan_no_funcdef():
    # Parsed text has no FunctionDef at the top level.
    assert _find_with_patch_paths_in_body("x = 1\n", {"old.X"}, {}, {}) == []


def test_body_scan_simple_match():
    src = 'def test_f():\n    with patch("old.X") as m:\n        pass\n'
    result = _find_with_patch_paths_in_body(src, {"old.X"}, {}, {})
    assert result == ["old.X"]


def test_body_scan_no_match():
    src = 'def test_f():\n    with patch("other.Y") as m:\n        pass\n'
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_attribute_patch():
    # ``with mock.patch(...)`` form.
    src = 'def test_f():\n    with mock.patch("old.X") as m:\n        pass\n'
    result = _find_with_patch_paths_in_body(src, {"old.X"}, {}, {})
    assert result == ["old.X"]


def test_body_scan_not_patch_call():
    src = 'def test_f():\n    with other("old.X") as m:\n        pass\n'
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_no_args():
    src = "def test_f():\n    with patch() as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_non_call_context_manager():
    # Context manager is a plain Name, not a Call.
    src = "def test_f():\n    with ctx as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_non_string_arg():
    # First arg is a Call expression (not string/Name/Attribute).
    src = "def test_f():\n    with patch(get_target()) as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_name_const_match():
    const_map = {"MY_TARGET": ("old.X", "/file.py")}
    src = "def test_f():\n    with patch(MY_TARGET) as m:\n        pass\n"
    result = _find_with_patch_paths_in_body(src, {"old.X"}, const_map, {})
    assert result == ["old.X"]


def test_body_scan_name_const_no_match():
    # Constant value doesn't match old_paths.
    const_map = {"MY_TARGET": ("other.Y", "/file.py")}
    src = "def test_f():\n    with patch(MY_TARGET) as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, const_map, {}) == []


def test_body_scan_name_not_in_const_map():
    src = "def test_f():\n    with patch(unknown_var) as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_attr_const_match():
    attr_const_map = {"consts": {"TARGET": ("old.X", "/consts.py")}}
    src = "def test_f():\n    with patch(consts.TARGET) as m:\n        pass\n"
    result = _find_with_patch_paths_in_body(src, {"old.X"}, {}, attr_const_map)
    assert result == ["old.X"]


def test_body_scan_attr_const_module_not_in_map():
    src = "def test_f():\n    with patch(unknown_mod.X) as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_attr_const_attr_not_in_map():
    attr_const_map = {"consts": {"OTHER": ("old.X", "/consts.py")}}
    src = "def test_f():\n    with patch(consts.MISSING) as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, attr_const_map) == []


def test_body_scan_attr_const_no_match():
    # Attribute constant value doesn't match old_paths.
    attr_const_map = {"consts": {"TARGET": ("other.Y", "/consts.py")}}
    src = "def test_f():\n    with patch(consts.TARGET) as m:\n        pass\n"
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, attr_const_map) == []


def test_body_scan_nested_funcdef_excluded():
    # ``with patch(...)`` inside a nested function should NOT trigger inclusion of
    # the outer function — the nested function is its own unit.
    src = (
        "def test_outer():\n"
        "    def inner():\n"
        '        with patch("old.X") as m:\n'
        "            pass\n"
    )
    assert _find_with_patch_paths_in_body(src, {"old.X"}, {}, {}) == []


def test_body_scan_multiple_with_items():
    # ``with patch("a") as m, patch("b") as n:`` — both items should be found.
    src = (
        "def test_f():\n"
        '    with patch("old.X") as m, patch("old.Y") as n:\n'
        "        pass\n"
    )
    result = _find_with_patch_paths_in_body(src, {"old.X", "old.Y"}, {}, {})
    assert set(result) == {"old.X", "old.Y"}


def test_body_scan_async_with():
    src = 'async def test_f():\n    async with patch("old.X") as m:\n        pass\n'
    result = _find_with_patch_paths_in_body(src, {"old.X"}, {}, {})
    assert result == ["old.X"]


def test_body_scan_nested_in_if():
    # ``with patch(...)`` inside an ``if`` block should still be found.
    src = (
        "def test_f():\n"
        "    if True:\n"
        '        with patch("old.X") as m:\n'
        "            pass\n"
    )
    result = _find_with_patch_paths_in_body(src, {"old.X"}, {}, {})
    assert result == ["old.X"]


def test_patch_strings_in_text_decorator():
    text = '@patch("pkg.mod.A")\ndef test_f(m): pass\n'
    assert _patch_strings_in_text(text) == {"pkg.mod.A"}


def test_patch_strings_in_text_attribute_decorator():
    text = '@mock.patch("pkg.mod.B")\ndef test_f(m): pass\n'
    assert _patch_strings_in_text(text) == {"pkg.mod.B"}


def test_patch_strings_in_text_context_manager():
    text = 'def test_f():\n    with patch("pkg.mod.C") as m: pass\n'
    assert _patch_strings_in_text(text) == {"pkg.mod.C"}


def test_patch_strings_in_text_multiple():
    text = (
        '@patch("pkg.mod.A")\n' '@mock.patch("pkg.mod.B")\n' "def test_f(a, b): pass\n"
    )
    assert _patch_strings_in_text(text) == {"pkg.mod.A", "pkg.mod.B"}


def test_patch_strings_in_text_empty():
    assert _patch_strings_in_text("def test_f(): pass\n") == set()
