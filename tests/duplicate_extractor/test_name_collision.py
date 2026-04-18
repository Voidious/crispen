from unittest.mock import MagicMock, patch
import textwrap
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _collect_ast_store_names,
    _extract_defined_names,
    _names_assigned_in,
    _names_in_edit_texts,
    _scope_end_line,
    _strip_helper_docstring,
)
from .test_extractor_core import _make_extract_response, _make_veto_response


def test_names_in_edit_texts_collects_from_all_edits():
    groups = [
        (
            "_helper",
            [
                (1, 3, "def _helper(last_import_line):\n    return last_import_line\n"),
                (5, 6, "result = _helper(x)\n"),
            ],
            "msg",
        )
    ]
    names = _names_in_edit_texts(groups)
    assert "last_import_line" in names
    assert "_helper" in names
    assert "result" in names
    assert "x" in names


def test_names_in_edit_texts_skips_syntax_errors():
    groups = [("_h", [(1, 2, "def (\n")], "msg")]
    # Should not raise — returns whatever names were parseable.
    names = _names_in_edit_texts(groups)
    assert isinstance(names, set)


def test_names_assigned_in_simple():
    assert _names_assigned_in("x = 1\n") == {"x"}


def test_names_assigned_in_tuple_unpack():
    assert _names_assigned_in("x, y = f()\n") == {"x", "y"}


def test_names_assigned_in_augassign():
    assert _names_assigned_in("x += 1\n") == {"x"}


def test_names_assigned_in_no_assign():
    assert _names_assigned_in("f()\n") == set()


def test_names_assigned_in_syntax_error():
    assert _names_assigned_in("def (\n") == set()


def test_extract_defined_names_basic():
    source = textwrap.dedent(
        """\
        def foo():
            pass

        async def bar():
            pass

        class Baz:
            pass
        """
    )
    assert _extract_defined_names(source) == {"foo", "bar", "Baz"}


def test_extract_defined_names_syntax_error():
    assert _extract_defined_names("def (\n") == set()


# Source that already defines _helper AND has duplicate blocks.
_COLLISION_SOURCE = textwrap.dedent(
    """\
    def _helper(x):
        return x

    def foo():
        if debug:
            pass
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def bar():
        result = None
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_COLLISION_RANGES = [(12, 14)]  # overlaps bar's duplicate block


def test_extraction_name_collision_skipped(monkeypatch, capsys):
    # LLM returns function_name="_helper", which is already defined → skipped.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": "def _helper(x, y):\n    pass\n",
                    "call_site_replacements": [
                        "    _helper(data, x)\n",
                        "    _helper(data, x)\n",
                    ],
                }
            ),
        ]
        de = DuplicateExtractor(
            _COLLISION_RANGES,
            source=_COLLISION_SOURCE,
            verbose=True,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None
    assert de.changes_made == []
    err = capsys.readouterr().err
    assert "name collision" in err
    assert "_helper" in err


def test_extraction_name_collision_silent(monkeypatch, capsys):
    # Same collision, verbose=False → no stderr output.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": "def _helper(x, y):\n    pass\n",
                    "call_site_replacements": [
                        "    _helper(data, x)\n",
                        "    _helper(data, x)\n",
                    ],
                }
            ),
        ]
        de = DuplicateExtractor(
            _COLLISION_RANGES,
            source=_COLLISION_SOURCE,
            verbose=False,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None
    assert de.changes_made == []
    err = capsys.readouterr().err
    assert "name collision" not in err


def test_strip_helper_docstring_with_docstring():
    source = 'def _helper(x):\n    """Strip me."""\n    return x\n'
    result = _strip_helper_docstring(source)
    assert '"""Strip me."""' not in result
    assert "return x" in result


def test_strip_helper_docstring_no_docstring():
    source = "def _helper(x):\n    return x\n"
    result = _strip_helper_docstring(source)
    assert result == source


def test_strip_helper_docstring_parse_error():
    bad = "def f(:\n    pass\n"
    result = _strip_helper_docstring(bad)
    assert result == bad


def test_strip_helper_docstring_non_function():
    source = "x = 1\n"
    result = _strip_helper_docstring(source)
    assert result == source


def test_strip_helper_docstring_docstring_only_body():
    # Function whose body is only a docstring — don't strip (would leave empty body).
    source = 'def _helper():\n    """Only doc."""\n'
    result = _strip_helper_docstring(source)
    assert '"""Only doc."""' in result


def test_collect_ast_store_names_simple_name():
    import ast

    node = ast.parse("x = 1").body[0].targets[0]
    names: list = []
    _collect_ast_store_names(node, names)
    assert names == ["x"]


def test_collect_ast_store_names_tuple():
    import ast

    node = ast.parse("a, b = 1, 2").body[0].targets[0]
    names: list = []
    _collect_ast_store_names(node, names)
    assert set(names) == {"a", "b"}


def test_collect_ast_store_names_nested_tuple():
    import ast

    node = ast.parse("(a, (b, c)) = x").body[0].targets[0]
    names: list = []
    _collect_ast_store_names(node, names)
    assert set(names) == {"a", "b", "c"}


def test_collect_ast_store_names_non_name_non_tuple_noop():
    # ast.Attribute target (e.g. self.x) → nothing collected.
    import ast

    node = ast.parse("self.x = 1").body[0].targets[0]
    names: list = []
    _collect_ast_store_names(node, names)
    assert names == []


def _make_source_lines(src: str):
    return src.splitlines(keepends=True)


def test_scope_end_line_module_returns_full_length():
    lines = _make_source_lines("x = 1\ny = 2\n")
    assert _scope_end_line(lines, "<module>", 1) == len(lines)


def test_scope_end_line_function_scope():
    src = "def foo():\n    x = 1\n    y = 2\n\ndef bar():\n    z = 3\n"
    lines = _make_source_lines(src)
    # Block ends at line 2 (inside foo). foo ends at line 3.
    assert _scope_end_line(lines, "foo", 2) == 3


def test_scope_end_line_does_not_bleed_into_next_function():
    src = "def foo():\n    x = 1\n\ndef bar():\n    x = 2\n"
    lines = _make_source_lines(src)
    # Searching for `x` after line 2 should stop at end of foo (line 2), not
    # reach bar where `x` also appears.
    end = _scope_end_line(lines, "foo", 2)
    assert end == 2  # foo ends at line 2; bar's x is excluded


def test_scope_end_line_picks_innermost_matching_scope():
    # Two functions named "inner" — one nested inside outer, one at module level.
    src = (
        "def outer():\n"
        "    def inner():\n"
        "        a = 1\n"
        "    inner()\n"
        "\n"
        "def inner():\n"
        "    b = 2\n"
    )
    lines = _make_source_lines(src)
    # Block at line 3 is inside the nested inner (lines 2-3). That is the
    # smallest matching span, so end_lineno == 3 is returned.
    assert _scope_end_line(lines, "inner", 3) == 3


def test_scope_end_line_class_scope():
    src = "class Foo:\n    x = 1\n    y = 2\n\nclass Bar:\n    x = 3\n"
    lines = _make_source_lines(src)
    assert _scope_end_line(lines, "Foo", 2) == 3


def test_scope_end_line_no_match_returns_full_length():
    src = "def foo():\n    x = 1\n"
    lines = _make_source_lines(src)
    # Scope name doesn't match any definition.
    assert _scope_end_line(lines, "bar", 1) == len(lines)


def test_scope_end_line_syntax_error_returns_full_length():
    lines = _make_source_lines("def (\n    x = 1\n")
    assert _scope_end_line(lines, "foo", 1) == len(lines)
