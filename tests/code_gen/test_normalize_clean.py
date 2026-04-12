from __future__ import annotations
import textwrap
from crispen.file_limiter.code_gen import (
    _extract_module_docstring,
    _multiline_string_ranges,
    _normalize_blank_lines,
    _source_is_only_docstring,
    _strip_module_docstring,
    _strip_orphaned_indented_comments,
    _strip_orphaned_section_headers,
    _sub_skip_strings,
)


def test_extract_module_docstring_present():
    src = '"""My module."""\n\nimport os\n'
    assert _extract_module_docstring(src) == '"""My module."""'


def test_extract_module_docstring_absent():
    src = "import os\n\ndef foo():\n    pass\n"
    assert _extract_module_docstring(src) is None


def test_extract_module_docstring_syntax_error():
    assert _extract_module_docstring("def (\n") is None


def test_extract_module_docstring_non_string_expr():
    # First statement is an expression but not a string constant.
    src = "1 + 1\n\ndef foo():\n    pass\n"
    assert _extract_module_docstring(src) is None


def test_strip_module_docstring_removes_docstring():
    src = '"""My module."""\n\n_CONST = 1\n'
    result = _strip_module_docstring(src)
    assert '"""My module."""' not in result
    assert "_CONST = 1" in result


def test_strip_module_docstring_no_docstring():
    src = "_CONST = 1\n"
    assert _strip_module_docstring(src) == src


def test_strip_module_docstring_syntax_error():
    src = "def (\n"
    assert _strip_module_docstring(src) == src


def test_source_is_only_docstring_true():
    assert _source_is_only_docstring('"""Just a docstring."""\n') is True


def test_source_is_only_docstring_with_other_content():
    assert _source_is_only_docstring('"""Doc."""\n\nimport os\n') is False


def test_source_is_only_docstring_no_docstring():
    assert _source_is_only_docstring("import os\n") is False


def test_source_is_only_docstring_syntax_error():
    assert _source_is_only_docstring("def (\n") is False


def test_strip_orphaned_3line_header_at_eof():
    """3-line block with no code after it is removed."""
    div = "# ---\n"
    source = "def foo():\n    pass\n\n\n" + div + "# Old Section\n" + div
    result = _strip_orphaned_section_headers(source)
    assert "# Old Section" not in result
    assert "def foo():" in result


def test_strip_orphaned_single_line_header_at_eof():
    """Single-line header with no code after it is removed."""
    source = "def foo():\n    pass\n\n# --- Removed ---\n"
    result = _strip_orphaned_section_headers(source)
    assert "# --- Removed ---" not in result
    assert "def foo():" in result


def test_strip_not_orphaned_3line_header():
    """3-line block followed by substantive code is kept."""
    div = "# ---\n"
    source = div + "# Helpers\n" + div + "\n\ndef helper():\n    pass\n"
    result = _strip_orphaned_section_headers(source)
    assert "# Helpers" in result
    assert "def helper():" in result


def test_strip_not_orphaned_single_line_header():
    """Single-line header followed by substantive code is kept."""
    source = "# --- Tools ---\n\ndef tool():\n    pass\n"
    result = _strip_orphaned_section_headers(source)
    assert "# --- Tools ---" in result


def test_strip_orphaned_header_followed_only_by_another_header():
    """Header followed only by another header (and then nothing) — both orphaned."""
    source = "def foo():\n" "    pass\n" "\n" "# --- First ---\n" "# --- Second ---\n"
    result = _strip_orphaned_section_headers(source)
    assert "# --- First ---" not in result
    assert "# --- Second ---" not in result
    assert "def foo():" in result


def test_strip_partial_orphan():
    """Only the header with no code after it is removed; the other stays."""
    source = (
        "# --- Active ---\n" "\n" "def foo():\n" "    pass\n" "\n" "# --- Empty ---\n"
    )
    result = _strip_orphaned_section_headers(source)
    assert "# --- Active ---" in result
    assert "# --- Empty ---" not in result


def test_strip_no_headers_returns_unchanged():
    """Source with no section headers is returned unchanged."""
    source = "def foo():\n    pass\n"
    assert _strip_orphaned_section_headers(source) == source


def test_strip_all_headers_have_content():
    """When every header has content below it, source is returned unchanged."""
    source = (
        "# --- A ---\n"
        "\n"
        "def a():\n"
        "    pass\n"
        "\n"
        "# --- B ---\n"
        "\n"
        "def b():\n"
        "    pass\n"
    )
    result = _strip_orphaned_section_headers(source)
    assert "# --- A ---" in result
    assert "# --- B ---" in result


def test_strip_equals_single_line_header_orphaned():
    """=== style orphaned header is also removed."""
    source = "def foo():\n    pass\n\n# === OLD SECTION ===\n"
    result = _strip_orphaned_section_headers(source)
    assert "# === OLD SECTION ===" not in result


def test_normalize_blank_lines_strips_leading_blanks():
    """Leading blank lines are removed (prevents E303 at top of file)."""
    source = "\n\n\ndef foo():\n    pass\n"
    result = _normalize_blank_lines(source)
    assert result.startswith("def foo():")


def test_normalize_blank_lines_collapses_excess_top_level():
    """4+ consecutive newlines between top-level defs collapse to 3."""
    source = "def foo():\n    pass\n\n\n\n\ndef bar():\n    pass\n"
    result = _normalize_blank_lines(source)
    assert "\n\n\n\n" not in result
    assert "def foo():" in result
    assert "def bar():" in result


def test_normalize_blank_lines_collapses_body_blanks():
    """2+ blank lines inside an indented body collapse to 1 (prevents E303 in body)."""
    source = "def foo():\n    x = 1\n\n\n    y = 2\n"
    result = _normalize_blank_lines(source)
    assert "\n\n\n    y" not in result
    assert "\n\n    y" in result


def test_normalize_blank_lines_empty_source():
    """Whitespace-only source returns empty string."""
    assert _normalize_blank_lines("\n\n\n") == ""


def test_normalize_blank_lines_trailing_newline():
    """Result always ends with exactly one newline."""
    source = "x = 1\n\n\n"
    result = _normalize_blank_lines(source)
    assert result.endswith("\n")
    assert not result.endswith("\n\n")


def test_normalize_blank_lines_preserves_multiline_string_body_blanks():
    """Blank lines inside a multi-line string literal are never collapsed.

    Regression: _EXCESS_BLANK_BODY_RE matched \\n{3,}(?=[ \\t]) inside
    triple-quoted strings, collapsing 2 blank lines before an indented line
    to 1 (e.g. stored source-code fixtures in tests).
    """
    # The triple-quoted string contains 2 blank lines before an indented `def`.
    # That produces the sequence \\n\\n\\n        def inside the raw source,
    # which _EXCESS_BLANK_BODY_RE would collapse to \\n\\n        def.
    source = textwrap.dedent(
        """\
        import textwrap
        def foo():
            src = textwrap.dedent(
                \"\"\"\\
                @dataclass
                class _SplitTask:
                    pass


                def _find_free_vars():
                    x = 1
                \"\"\"
            )
        """
    )
    result = _normalize_blank_lines(source)
    # Two blank lines before the indented `def` inside the string must survive.
    # After outer textwrap.dedent the string content has 8-space indentation.
    assert "\n\n\n        def _find_free_vars" in result


def test_normalize_blank_lines_still_collapses_excess_outside_strings():
    """Blank-line collapsing still fires for code outside string literals."""
    source = "def foo():\n    x = 1\n\n\n    y = 2\n"
    result = _normalize_blank_lines(source)
    assert "\n\n\n    y" not in result
    assert "\n\n    y" in result


def test_multiline_string_ranges_triple_quoted():
    """Detects a triple-quoted string spanning multiple lines."""
    source = 'x = """\nhello\n"""\n'
    ranges = _multiline_string_ranges(source)
    assert len(ranges) == 1
    start, end = ranges[0]
    assert source[start:end] == '"""\nhello\n"""'


def test_multiline_string_ranges_single_line_string_ignored():
    """Single-line strings (no literal newline) are not returned."""
    source = 'x = "hello\\n"\n'
    ranges = _multiline_string_ranges(source)
    assert ranges == []


def test_multiline_string_ranges_no_strings():
    """Returns empty list when there are no string literals."""
    source = "x = 1 + 2\n"
    ranges = _multiline_string_ranges(source)
    assert ranges == []


def test_multiline_string_ranges_invalid_source():
    """Falls back to empty list on tokenization error."""
    # Unterminated string triggers TokenError.
    source = 'x = """\nhello\n'
    ranges = _multiline_string_ranges(source)
    assert ranges == []


def test_sub_skip_strings_does_not_touch_string_content():
    """Pattern match inside a multi-line string is not substituted."""
    import re

    pattern = re.compile(r"\n{3,}(?=[ \t])")
    source = 'def f():\n    s = """\n    a\n\n\n    b\n    """\n'
    result = _sub_skip_strings(pattern, "\n\n", source)
    # The sequence inside the string must survive unchanged.
    assert "\n\n\n    b" in result


def test_sub_skip_strings_applies_outside_strings():
    """Pattern match outside string literals is substituted normally."""
    import re

    pattern = re.compile(r"\n{3,}(?=[ \t])")
    source = "def f():\n    x = 1\n\n\n    y = 2\n"
    result = _sub_skip_strings(pattern, "\n\n", source)
    assert "\n\n\n    y" not in result
    assert "\n\n    y" in result


def test_sub_skip_strings_no_strings_falls_through():
    """When there are no multi-line strings the plain .sub() path is taken."""
    import re

    pattern = re.compile(r"x")
    source = "x = 1\n"
    result = _sub_skip_strings(pattern, "y", source)
    assert result == "y = 1\n"


def test_strip_orphaned_indented_comments_removes_orphan():
    """Indented comment at module level (outside any AST node) is removed."""
    source = "\n\n    # This comment was left behind after function removal\n"
    result = _strip_orphaned_indented_comments(source)
    assert "# This comment was left behind" not in result


def test_strip_orphaned_indented_comments_keeps_inside_function():
    """Indented comment inside a function body is preserved."""
    source = "def foo():\n    # normal comment\n    pass\n"
    result = _strip_orphaned_indented_comments(source)
    assert "# normal comment" in result


def test_strip_orphaned_indented_comments_keeps_module_level_comment():
    """Non-indented module-level comment is preserved."""
    source = "# module comment\ndef foo():\n    pass\n"
    result = _strip_orphaned_indented_comments(source)
    assert "# module comment" in result


def test_strip_orphaned_indented_comments_syntax_error():
    """SyntaxError in source returns source unchanged."""
    source = "    # orphaned\ndef f(: pass\n"
    result = _strip_orphaned_indented_comments(source)
    assert result == source
