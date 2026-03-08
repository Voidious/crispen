from crispen.refactors.duplicate_extractor import (
    _seq_ends_with_return,
    _strip_helper_docstring,
)
from ..test_weight_and_normalization import _make_seq_with_source


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


def test_seq_ends_with_return_true():
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    return x\n"))
        is True
    )


def test_seq_ends_with_return_false_no_return():
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    y = 2\n")) is False
    )


def test_seq_ends_with_return_syntax_error():
    assert _seq_ends_with_return(_make_seq_with_source("    (\n")) is False


def test_seq_ends_with_return_empty_body():
    # Pure whitespace → ast.parse produces an empty module body.
    assert _seq_ends_with_return(_make_seq_with_source("   \n")) is False


def test_seq_ends_with_return_bare_return():
    # Bare `return` is equivalent to returning None — not flagged.
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    return\n")) is False
    )


def test_seq_ends_with_return_return_none():
    # Explicit `return None` is also equivalent to implicit None — not flagged.
    assert (
        _seq_ends_with_return(_make_seq_with_source("    x = 1\n    return None\n"))
        is False
    )
