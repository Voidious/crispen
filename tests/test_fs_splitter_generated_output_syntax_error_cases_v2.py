from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import FunctionSplitter
from tests.fs_mock_llm_response_factory_v2 import _make_mock_response
from tests.fs_splitter_long_function_fixtures_v3 import _make_long_func


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
