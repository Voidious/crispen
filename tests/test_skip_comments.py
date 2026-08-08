"""Tests for skip_comments — 100% branch coverage."""

from __future__ import annotations

from crispen.skip_comments import extract_comments, has_skip_file_marker, is_skipped

# ---------------------------------------------------------------------------
# extract_comments
# ---------------------------------------------------------------------------


def test_extract_comments_basic():
    source = "x = 1  # a comment\ny = 2\n"
    assert extract_comments(source) == {1: "# a comment"}


def test_extract_comments_no_comments():
    assert extract_comments("x = 1\n") == {}


def test_extract_comments_ignores_hash_in_string():
    source = 'x = "#not a comment"  # real comment\n'
    assert extract_comments(source) == {1: "# real comment"}


def test_extract_comments_syntax_error_returns_empty():
    assert extract_comments("def f(:\n") == {}


# ---------------------------------------------------------------------------
# is_skipped
# ---------------------------------------------------------------------------


def test_is_skipped_trailing_bare_skip():
    source = "if x:\n    pass  # crispen: skip\n"
    lines = source.splitlines()
    comments = extract_comments(source)
    assert is_skipped(2, "if_not_else", lines, comments) is True


def test_is_skipped_no_comment_on_line():
    source = "if x:\n    pass\n"
    lines = source.splitlines()
    comments = extract_comments(source)
    assert is_skipped(2, "if_not_else", lines, comments) is False


def test_is_skipped_trailing_comment_not_a_marker():
    source = "if x:\n    pass  # just a note\n"
    lines = source.splitlines()
    comments = extract_comments(source)
    assert is_skipped(2, "if_not_else", lines, comments) is False


def test_is_skipped_scoped_marker_matches():
    source = "def f():\n    pass  # crispen: skip=duplicate_extractor\n"
    lines = source.splitlines()
    comments = extract_comments(source)
    assert is_skipped(2, "duplicate_extractor", lines, comments) is True


def test_is_skipped_scoped_marker_multiple_names():
    source = "def f():\n    pass  # crispen: skip=duplicate_extractor,tuple_dataclass\n"
    lines = source.splitlines()
    comments = extract_comments(source)
    assert is_skipped(2, "tuple_dataclass", lines, comments) is True
    assert is_skipped(2, "function_splitter", lines, comments) is False


def test_is_skipped_malformed_empty_scope_is_not_a_bare_skip():
    # "skip=" with no names after the "=" (e.g. a typo/paste error) must not
    # silently fall back to a bare, all-refactors skip.
    source = "def f():\n    pass  # crispen: skip=\n"
    lines = source.splitlines()
    comments = extract_comments(source)
    assert is_skipped(2, "duplicate_extractor", lines, comments) is False
    assert is_skipped(2, "function_splitter", lines, comments) is False


def test_is_skipped_leading_comment_directly_above():
    source = "# crispen: skip\ndef f():\n    pass\n"
    lines = source.splitlines()
    comments = extract_comments(source)
    assert is_skipped(2, "function_splitter", lines, comments) is True


def test_is_skipped_leading_comment_through_blank_line():
    source = "# crispen: skip\n\ndef f():\n    pass\n"
    lines = source.splitlines()
    comments = extract_comments(source)
    assert is_skipped(3, "function_splitter", lines, comments) is True


def test_is_skipped_leading_comment_through_decorator():
    source = "# crispen: skip\n@decorator\ndef f():\n    pass\n"
    lines = source.splitlines()
    comments = extract_comments(source)
    assert is_skipped(3, "function_splitter", lines, comments) is True


def test_is_skipped_leading_comment_through_multiple_decorators():
    source = "# crispen: skip\n@dec_a\n@dec_b\ndef f():\n    pass\n"
    lines = source.splitlines()
    comments = extract_comments(source)
    assert is_skipped(4, "function_splitter", lines, comments) is True


def test_is_skipped_stops_at_real_code_line():
    source = "x = 1\ndef f():\n    pass\n"
    lines = source.splitlines()
    comments = extract_comments(source)
    assert is_skipped(2, "function_splitter", lines, comments) is False


def test_is_skipped_leading_comment_not_a_marker():
    source = "# just a docstring-ish comment\ndef f():\n    pass\n"
    lines = source.splitlines()
    comments = extract_comments(source)
    assert is_skipped(2, "function_splitter", lines, comments) is False


def test_is_skipped_walk_reaches_top_of_file():
    source = "def f():\n    pass\n"
    lines = source.splitlines()
    comments = extract_comments(source)
    assert is_skipped(1, "function_splitter", lines, comments) is False


def test_is_skipped_empty_source_lines_no_crash():
    # Regression: querying a line number beyond an empty/short source_lines
    # list (e.g. a Refactor constructed without a `source`) used to raise
    # IndexError instead of returning False.
    assert is_skipped(5, "if_not_else", [], {}) is False


def test_is_skipped_skip_file_marker_is_not_a_bare_skip():
    source = "# crispen: skip-file\ndef f():\n    pass\n"
    lines = source.splitlines()
    comments = extract_comments(source)
    assert is_skipped(2, "function_splitter", lines, comments) is False


# ---------------------------------------------------------------------------
# has_skip_file_marker
# ---------------------------------------------------------------------------


def test_has_skip_file_marker_present():
    assert has_skip_file_marker("# crispen: skip-file\nimport os\n") is True


def test_has_skip_file_marker_absent():
    assert has_skip_file_marker("import os\n") is False


def test_has_skip_file_marker_bare_skip_does_not_count():
    assert has_skip_file_marker("x = 1  # crispen: skip\n") is False
