from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import FunctionSplitter
from .mock_responses import _make_mock_response


@patch("crispen.llm_client.anthropic")
def test_function_splitter_method_self_needed_uses_instance_method(mock_anthropic):
    """When every tail needs self, split into a regular instance method helper."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["tail_work"])
    )
    lines = ["class Foo:\n", "    def method(self):\n"]
    for i in range(40):
        lines.append(f"        a{i} = self.val + {i}\n")
    lines.append("        return 0\n")
    src = "".join(lines)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=20
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    assert "@staticmethod" not in result
    assert "return self._tail_work(" in result
    assert "def _tail_work(self" in result
