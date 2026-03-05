from __future__ import annotations
from unittest.mock import patch
from crispen.refactors.function_splitter import FunctionSplitter
from tests.mocks import _make_mock_response
from tests.test_pyflakes_integration import _create_splitter_with_patched_key
from tests.test_splitter_utils import _make_long_func


def test_engine_includes_function_splitter_no_op(tmp_path):
    """FunctionSplitter is in _REFACTORS and runs without error for simple files."""
    from crispen.engine import run_engine
    from crispen.config import CrispenConfig

    py_file = tmp_path / "sample.py"
    py_file.write_text("def foo():\n    return 1\n")
    config = CrispenConfig(max_function_length=75)
    msgs = list(run_engine({str(py_file): [(1, 2)]}, verbose=False, config=config))
    # No split needed — no messages expected (or just no errors)
    assert all("FunctionSplitter" not in m for m in msgs)


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


@patch("crispen.llm_client.anthropic")
def test_function_splitter_skips_name_collision(mock_anthropic):
    """Helper name colliding with an existing function causes the task to be dropped."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["helper"])  # would produce _helper
    )
    # _helper already exists; the LLM would name the extracted helper "helper"
    existing = "def _helper():\n    pass\n\n\n"
    src = existing + _make_long_func(80)

    splitter = _create_splitter_with_patched_key(src)

    # collision detected → task dropped → no rewrite
    assert splitter.get_rewritten_source() is None
