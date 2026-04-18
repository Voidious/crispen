from unittest.mock import MagicMock, patch
import textwrap
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _replacement_contains_return,
    _seq_ends_with_return,
    _seq_source_contains_yield,
)
from .test_edit_operations import _make_seq_with_source
from .test_collectors import _collect_sequences
from .test_extractor_core import _make_extract_response, _make_veto_response


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


def test_seq_source_contains_yield_async_with_yield():
    # The exact pattern that triggered the bug: async with ... as c: yield c
    src = "    async with Client(mcp) as c:\n        yield c\n"
    assert _seq_source_contains_yield(src) is True


def test_seq_source_contains_yield_plain_yield():
    assert _seq_source_contains_yield("    yield x\n") is True


def test_seq_source_contains_yield_from():
    assert _seq_source_contains_yield("    yield from something()\n") is True


def test_seq_source_contains_yield_no_yield():
    assert _seq_source_contains_yield("    x = 1\n    y = 2\n") is False


def test_seq_source_contains_yield_nested_funcdef_not_counted():
    # yield inside a nested def must NOT trigger the guard
    src = "    def inner():\n        yield 1\n"
    assert _seq_source_contains_yield(src) is False


def test_seq_source_contains_yield_syntax_error():
    assert _seq_source_contains_yield("    (\n") is False


def test_collector_skips_yield_sequences():
    # Sequences whose source contains yield should never be collected.
    source = textwrap.dedent(
        """\
        async def make_client():
            x = setup()
            async with Client(x) as c:
                yield c

        async def make_client2():
            x = setup()
            async with Client(x) as c:
                yield c
        """
    )
    seqs = _collect_sequences(source)
    for seq in seqs:
        assert not _seq_source_contains_yield(seq.source)


def test_replacement_contains_return_true():
    assert _replacement_contains_return("    return x\n") is True


def test_replacement_contains_return_false():
    assert _replacement_contains_return("    _helper()\n") is False


def test_replacement_contains_return_syntax_error():
    # Unclosed paren inside the wrapper → SyntaxError → False.
    assert _replacement_contains_return("    (\n") is False


_RETURN_BLOCK_SOURCE = textwrap.dedent(
    """\
    def foo():
        if debug:
            pass
        x = compute(data)
        y = transform(x)
        return y

    def bar():
        result = None
        x = compute(data)
        y = transform(x)
        return y
    """
)
_RETURN_BLOCK_RANGES = [(10, 12)]  # overlaps bar's duplicate block (x/y/return lines)


def _make_return_block_extract_response():
    return _make_extract_response(
        {
            "function_name": "_helper",
            "placement": "module_level",
            "helper_source": (
                "def _helper():\n"
                "    x = compute(data)\n"
                "    y = transform(x)\n"
                "    return y\n"
            ),
            # replacement drops the return — this is the bug being guarded
            "call_site_replacements": [
                "    _helper()\n",
                "    _helper()\n",
            ],
        }
    )


def test_block_ends_with_return_guard_skips(monkeypatch, capsys):
    """Extraction rejected when block ends with return but replacement omits it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_return_block_extract_response(),
        ]
        de = DuplicateExtractor(
            _RETURN_BLOCK_RANGES,
            source=_RETURN_BLOCK_SOURCE,
            extraction_retries=0,
            llm_verify_retries=0,
        )
    assert de._new_source is None
    assert "block ends with return but replacement omits it" in capsys.readouterr().err


def test_block_ends_with_return_guard_skips_silent(monkeypatch):
    """verbose=False: extraction rejected with no stderr output."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_return_block_extract_response(),
        ]
        de = DuplicateExtractor(
            _RETURN_BLOCK_RANGES,
            source=_RETURN_BLOCK_SOURCE,
            verbose=False,
            extraction_retries=0,
            llm_verify_retries=0,
        )
    assert de._new_source is None
