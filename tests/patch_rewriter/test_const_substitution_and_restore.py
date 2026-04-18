from __future__ import annotations
from crispen.patch_rewriter import (
    _ConstRef,
    _get_const_votes_from_rewrite,
    _restore_const_refs,
    _substitute_consts_in_func_text,
)


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
