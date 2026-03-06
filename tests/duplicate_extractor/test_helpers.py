import textwrap
from unittest.mock import MagicMock, patch
import libcst as cst
from libcst.metadata import MetadataWrapper
from crispen.refactors.duplicate_extractor import (
    _SequenceCollector,
    _strip_helper_docstring,
    DuplicateExtractor,
)
from .test_extraction import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)

_DUP_SOURCE = textwrap.dedent(
    """\
    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def bar():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_DUP_RANGES = [(7, 9)]  # overlaps bar's body

# Source where foo's duplicate block assigns z, and foo uses z after the block.
# _has_escaping_vars should detect this and skip the extraction.
_ESC_SOURCE = textwrap.dedent(
    """\
    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
        assert z == expected

    def bar():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_ESC_RANGES = [(8, 10)]  # overlaps bar's body

_POST_STEAL_SOURCE = textwrap.dedent(
    """\
    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
        return z

    def bar():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
        logger.info("done")
    """
)
_POST_STEAL_RANGES = [(8, 10)]  # overlaps bar's 3-statement block


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


def test_sequence_collector_min_weight_filters_light_sequences():
    # A single assignment has weight 1. With min_weight=2 it should be excluded.
    source = "def f():\n    a = 1\n    b = 2\n"
    source_lines = source.splitlines(keepends=True)
    tree = cst.parse_module(source)

    collector = _SequenceCollector(source_lines, max_seq_len=2, min_weight=2)
    MetadataWrapper(tree).visit(collector)
    # Single-statement sequences (weight=1) should be filtered out
    single_stmt_seqs = [s for s in collector.sequences if len(s.stmts) == 1]
    assert single_stmt_seqs == []


def test_duplicate_extractor_helper_docstrings_false_strips_docstring(
    monkeypatch, capsys
):
    """When helper_docstrings=False, the LLM-returned docstring is stripped."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_shared",
                    "placement": "module_level",
                    "helper_source": (
                        "def _shared(data):\n"
                        '    """LLM added a docstring."""\n'
                        "    pass\n"
                    ),
                    "call_site_replacements": [
                        "    _shared(data)\n",
                        "    _shared(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor(
            _DUP_RANGES, source=_DUP_SOURCE, verbose=False, helper_docstrings=False
        )

    assert de._new_source is not None
    assert '"""LLM added a docstring."""' not in de._new_source


def test_duplicate_extractor_helper_docstrings_true_keeps_docstring(
    monkeypatch, capsys
):
    """When helper_docstrings=True, the LLM-returned docstring is preserved."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_shared",
                    "placement": "module_level",
                    "helper_source": (
                        "def _shared(data):\n"
                        '    """Keep this docstring."""\n'
                        "    pass\n"
                    ),
                    "call_site_replacements": [
                        "    _shared(data)\n",
                        "    _shared(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor(
            _DUP_RANGES, source=_DUP_SOURCE, verbose=False, helper_docstrings=True
        )

    assert de._new_source is not None
    assert '"""Keep this docstring."""' in de._new_source


def test_duplicate_extractor_custom_model_used(monkeypatch):
    """Custom model string is passed to the Anthropic API."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.return_value = _make_veto_response(False, "no")
        DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE, model="claude-opus-4-6")
    # Verify the custom model was passed
    call_kwargs = mock_client.messages.create.call_args_list[0][1]
    assert call_kwargs["model"] == "claude-opus-4-6"
