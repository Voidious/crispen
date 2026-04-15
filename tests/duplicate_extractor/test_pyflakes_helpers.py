from unittest.mock import MagicMock, patch
import textwrap
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _pyflakes_new_undefined_names,
    _pyflakes_strip_unused_simple_assigns,
    _replace_unused_in_target,
    _strip_unused_call_assignments,
)
from .test_duplicate_extractor_integration import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)


def test_pyflakes_new_undefined_names_returns_empty_when_no_new_issues():
    # Names undefined in both original and candidate → no NEW issues.
    original = "def foo():\n    return bar()\n"
    candidate = "def _h():\n    pass\n\ndef foo():\n    return bar()\n"
    assert _pyflakes_new_undefined_names(original, candidate) == set()


def test_pyflakes_new_undefined_names_detects_introduced_name():
    # candidate introduces a reference to an unassigned name not in original.
    original = "def foo():\n    x = 1\n    return x\n"
    # candidate removes the assignment, leaving x undefined at the call site
    candidate = "def _h():\n    x = 1\n\ndef foo():\n    _h(x)\n    return x\n"
    assert "x" in _pyflakes_new_undefined_names(original, candidate)


def test_pyflakes_strip_unused_simple_assigns_removes_literal_init():
    # last_import_line = 0 becomes unused after extraction.
    source = textwrap.dedent(
        """\
        def foo(source):
            last_import_line = 0
            lines = source.splitlines()
            return lines
    """
    )
    result = _pyflakes_strip_unused_simple_assigns(source, {"last_import_line"})
    assert "last_import_line" not in result
    assert "lines = source.splitlines()" in result


def test_pyflakes_strip_unused_simple_assigns_keeps_call_rhs():
    # x = func() must NOT be stripped — it has side effects.
    source = textwrap.dedent(
        """\
        def foo():
            x = side_effect()
            return 1
    """
    )
    result = _pyflakes_strip_unused_simple_assigns(source, {"x"})
    assert "x = side_effect()" in result


def test_pyflakes_strip_unused_simple_assigns_no_change_when_used():
    source = textwrap.dedent(
        """\
        def foo(source):
            last_import_line = 0
            for line in source.splitlines():
                last_import_line += 1
            return last_import_line
    """
    )
    result = _pyflakes_strip_unused_simple_assigns(source, {"last_import_line"})
    assert result == source


def test_pyflakes_strip_unused_simple_assigns_fallback_on_empty_block():
    # If stripping would leave a block with no statements (syntax error),
    # the original source is returned unchanged.
    source = textwrap.dedent(
        """\
        def foo():
            x = 0
    """
    )
    # After stripping x = 0 the function body is empty — SyntaxError.
    result = _pyflakes_strip_unused_simple_assigns(source, {"x"})
    assert result == source


def test_pyflakes_strip_unused_simple_assigns_module_level_unchanged():
    # Module-level assignments are not flagged as UnusedVariable by pyflakes.
    source = "x = 0\n"
    result = _pyflakes_strip_unused_simple_assigns(source, {"x"})
    assert result == source


def test_pyflakes_strip_unused_simple_assigns_skips_unrelated_names():
    # A variable unused after extraction but NOT in allowed_names is preserved.
    source = textwrap.dedent(
        """\
        def foo(source):
            unrelated = 0
            lines = source.splitlines()
            return lines
    """
    )
    # "unrelated" is not in the allowed set → must not be removed.
    result = _pyflakes_strip_unused_simple_assigns(source, {"last_import_line"})
    assert "unrelated = 0" in result


def test_pyflakes_strip_unused_simple_assigns_empty_allowed():
    # Empty allowed_names means nothing can be stripped.
    source = textwrap.dedent(
        """\
        def foo(source):
            x = 0
            lines = source.splitlines()
            return lines
    """
    )
    result = _pyflakes_strip_unused_simple_assigns(source, set())
    assert result == source


def test_replace_unused_in_target_name_used():
    import ast

    target = ast.parse("result = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "print(result)\n")
    assert all_r is False and any_r is False
    assert ast.unparse(new_t) == "result"


def test_replace_unused_in_target_name_unused():
    import ast

    target = ast.parse("result = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "return None\n")
    assert all_r is True and any_r is True
    assert ast.unparse(new_t) == "_"


def test_replace_unused_in_target_tuple_all_unused():
    import ast

    target = ast.parse("a, b = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "return None\n")
    assert all_r is True and any_r is True
    assert ast.unparse(new_t) == "(_, _)"


def test_replace_unused_in_target_tuple_some_unused():
    import ast

    target = ast.parse("a, b = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "print(a)\n")
    assert all_r is False and any_r is True
    assert ast.unparse(new_t) == "(a, _)"


def test_replace_unused_in_target_tuple_all_used():
    import ast

    target = ast.parse("a, b = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "print(a, b)\n")
    assert all_r is False and any_r is False


def test_replace_unused_in_target_attribute_treated_as_used():
    import ast

    target = ast.parse("self.x = 1").body[0].targets[0]
    new_t, all_r, any_r = _replace_unused_in_target(target, "return None\n")
    assert all_r is False and any_r is False


def test_strip_unused_call_assignments_removes_unused_single():
    # `result` never appears after the block → assignment stripped.
    replacement = "    result = _helper(x, y)\n"
    following = ["    do_something()\n", "    return z\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    _helper(x, y)\n"


def test_strip_unused_call_assignments_keeps_used_single():
    # `result` is referenced after the block → assignment kept.
    replacement = "    result = _helper(x, y)\n"
    following = ["    print(result)\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_removes_unused_tuple():
    # Both names unused after the block → assignment stripped entirely.
    replacement = "    a, b = _helper(x)\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    _helper(x)\n"


def test_strip_unused_call_assignments_partial_tuple_replaces_with_underscore():
    # One name used, one unused → replace unused with _.
    replacement = "    a, b = _helper(x)\n"
    following = ["    print(a)\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    (a, _) = _helper(x)\n"


def test_strip_unused_call_assignments_attribute_target_unchanged():
    # Target is an attribute (self.x = call()) → treated as used → left unchanged.
    replacement = "    self.result = _helper(x)\n"
    following = []
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_non_call_rhs_unchanged():
    # RHS is not a Call → leave unchanged.
    replacement = "    result = x + y\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_chained_all_unused_stripped():
    # Chained assignment where every name is unused → stripped to just the call.
    replacement = "    a = b = _helper(x)\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    _helper(x)\n"


def test_strip_unused_call_assignments_chained_some_used_unchanged():
    # Chained assignment where one name is used → left unchanged.
    replacement = "    a = b = _helper(x)\n"
    following = ["    print(a)\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_chained_no_names_unchanged():
    # Chained assignment whose targets yield no names (e.g. attributes) → unchanged.
    replacement = "    self.a = self.b = _helper(x)\n"
    following = []
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_no_assignment_unchanged():
    # Replacement is already just a call → returned as-is.
    replacement = "    _helper(x, y)\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_syntax_error_unchanged():
    # Unparseable replacement → returned unchanged.
    replacement = "    def (\n"
    following = []
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_multiline_replacement():
    # Multi-statement replacement: only the unused assignment is stripped.
    replacement = "    result = _helper(x)\n    do_other()\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    _helper(x)\n    do_other()\n"


def test_strip_unused_call_assignments_preserves_indentation():
    # Indentation of stripped replacement matches original block indent.
    replacement = "        result = _helper(x)\n"
    following = []
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "        _helper(x)\n"


def test_strip_unused_call_assignments_leading_blank_line():
    # Replacement with a blank leading line: indent is read from first content line.
    replacement = "\n    result = _helper(x)\n"
    following = []
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "\n    _helper(x)\n"


def test_strip_unused_call_assignments_await_unused_stripped():
    # `result = await _helper(x)` and `result` never used → strip assignment.
    replacement = "    result = await _helper(x)\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    await _helper(x)\n"


def test_strip_unused_call_assignments_await_used_kept():
    # `result = await _helper(x)` and `result` is used → keep assignment.
    replacement = "    result = await _helper(x)\n"
    following = ["    print(result)\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_strip_unused_call_assignments_await_tuple_unused_stripped():
    # `a, b = await _helper(x)` and neither name is used → strip assignment.
    replacement = "    a, b = await _helper(x)\n"
    following = ["    return None\n"]
    out = _strip_unused_call_assignments(replacement, following)
    assert out == "    await _helper(x)\n"


def test_strip_unused_call_assignments_await_non_call_unchanged():
    # `result = await some_awaitable` (not a call) → left unchanged.
    replacement = "    result = await some_awaitable\n"
    following = []
    out = _strip_unused_call_assignments(replacement, following)
    assert out == replacement


def test_restrip_drops_assignment_unused_only_after_all_call_sites_replaced(
    monkeypatch,
):
    # Regression: when two call sites reference the same variable name, the
    # per-call-site strip (which uses original following lines) sees the name
    # in the other call site's original block and keeps the assignment.  After
    # all replacements are assembled the variable is truly unused, so the
    # re-strip pass must drop it.
    #
    # Source:  test_f has two identical 2-line blocks.
    # LLM returns:
    #   - call site 1 replacement: ``data = assert_error(result)``
    #   - call site 2 replacement: ``assert_error(result2)``   (no assignment)
    # After initial per-call-site strip, call site 1 keeps the assignment
    # because "data" appears in the original following source (inside call
    # site 2's original block).  The re-strip must then drop it.
    # Using function parameters avoids the SequenceCollector merging the
    # assignment lines into the duplicate block.
    # Use 3-statement blocks (weight=3 ≥ min_weight) so the SequenceCollector
    # finds the duplicate group.  Mirroring the real lever-mcp pattern:
    # json.loads + two asserts.  Both result and result2 are function
    # parameters so the SequenceCollector cannot absorb the assignment lines
    # into the duplicate block.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        def test_f(result, result2):
            rd = json.loads(result)
            assert rd["value"] is None
            assert "error" in rd
            rd = json.loads(result2)
            assert rd["value"] is None
            assert "error" in rd
        """
    )
    helper = textwrap.dedent(
        """\
        def assert_error_result(result):
            rd = json.loads(result)
            assert rd["value"] is None
            assert "error" in rd
        """
    )
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "identical blocks"),
            _make_extract_response(
                {
                    "function_name": "assert_error_result",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        # LLM assigns the return value at call site 1 …
                        "    rd = assert_error_result(result)\n",
                        # … but not at call site 2 (helper returns None).
                        "    assert_error_result(result2)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor([(2, 4), (5, 7)], source=source)

    assert de._new_source is not None
    # The re-strip must have dropped the unused assignment at call site 1.
    assert "rd = assert_error_result(result)" not in de._new_source
    assert "assert_error_result(result)" in de._new_source
    assert "assert_error_result(result2)" in de._new_source
