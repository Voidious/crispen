from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import FunctionSplitter
from tests.test_function_splitter_helpers import _make_long_func
from tests.test_parsing_helpers import _make_mock_response


def test_function_splitter_parse_error_no_crash():
    # Invalid source should not crash
    splitter = FunctionSplitter([(1, 10)], source="def f(\n  !!invalid", verbose=False)
    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_syntax_error_in_output_is_skipped(mock_anthropic):
    # If the assembled edit is invalid Python, the change is not applied
    # We simulate this by making _generate_call return something invalid
    # Instead, test the path via a function with 1-stmt body (no valid split)
    src = "def foo():\n    x = 1\n"  # only 1 stmt → can't split
    splitter = FunctionSplitter([(1, 10)], source=src, verbose=False, max_lines=0)
    # body lines=1 > 0=max_lines → tries to split but len(body_stmts)=1 < 2 → skip
    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_no_valid_split_skipped(mock_anthropic):
    # max_lines=1 → even a head with 1 stmt (+return call=2) > max_lines=1
    # So no valid splits → no change
    src = _make_long_func(5, "foo")

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter([(1, 1000)], source=src, verbose=False, max_lines=1)

    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_syntax_error_in_generated_output(mock_anthropic):
    """If assembled output fails compile(), the change is not applied."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["helper"])
    )
    src = _make_long_func(80, "foo")

    import builtins as _builtins

    orig_compile = _builtins.compile

    def _selective_compile(source, filename, mode, *args, **kwargs):
        if filename == "<string>":
            raise SyntaxError("mocked error for test")
        return orig_compile(source, filename, mode, *args, **kwargs)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("builtins.compile", side_effect=_selective_compile):
            splitter = FunctionSplitter(
                [(1, 1000)], source=src, verbose=False, max_lines=50
            )

    assert splitter.get_rewritten_source() is None
