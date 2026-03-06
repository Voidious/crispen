from __future__ import annotations
from unittest.mock import MagicMock, patch
import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider
from crispen.refactors.function_splitter import (
    _head_effective_lines,
    _stmts_source,
    FunctionSplitter,
)


def _parse_func(source: str):
    """Return (body_stmts, positions, source_lines) for the first function.

    Uses a CSTVisitor to capture body_stmts from the wrapper's internal copy,
    ensuring they match the keys in the positions dict.
    """
    tree = cst.parse_module(source)
    wrapper = MetadataWrapper(tree)
    positions = wrapper.resolve(PositionProvider)

    class _Getter(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def __init__(self):
            self.stmts: list = []

        def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
            if not self.stmts:  # first function only
                self.stmts = list(node.body.body)

    getter = _Getter()
    wrapper.visit(getter)
    source_lines = source.splitlines(keepends=True)
    return getter.stmts, positions, source_lines


def _make_mock_response(names_list):
    """Build a mock Anthropic message response for the name_helper_functions tool."""
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "name_helper_functions"
    mock_block.input = {
        "names": [{"id": str(i), "name": n} for i, n in enumerate(names_list)]
    }
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    return mock_response


def _parse_stmt(src: str) -> cst.BaseStatement:
    return cst.parse_module(src).body[0]


def test_stmts_source_basic():
    src = "def foo():\n    x = 1\n    y = 2\n    z = 3\n"
    stmts, positions, lines = _parse_func(src)
    result = _stmts_source(stmts[:2], lines, positions)
    assert "x = 1" in result
    assert "y = 2" in result
    assert "z = 3" not in result


def test_stmts_source_empty():
    src = "def foo():\n    x = 1\n"
    _, positions, lines = _parse_func(src)
    assert _stmts_source([], lines, positions) == ""


def test_stmts_source_dedented():
    src = "def foo():\n    x = 1\n    y = 2\n"
    stmts, positions, lines = _parse_func(src)
    result = _stmts_source(stmts, lines, positions)
    # Should be dedented (no leading 4-space indent)
    assert result.startswith("x = 1") or result.startswith("x = 1\n")


def test_head_effective_lines_no_docstring():
    src = "def foo():\n    x = 1\n    y = 2\n    z = 3\n"
    stmts, positions, lines = _parse_func(src)
    # split_idx=2: head=[x,y], last=y at line 3, first=x at line 2 → 3-2+2=3
    result = _head_effective_lines(stmts, 2, positions, False)
    assert result == 3


def test_head_effective_lines_with_docstring_normal():
    src = 'def foo():\n    """doc"""\n    x = 1\n    y = 2\n    z = 3\n'
    stmts, positions, lines = _parse_func(src)
    # split_idx=3: head=[doc, x, y], first_non_doc=x at line 3, last=y at line 4
    # 4-3+2=3
    result = _head_effective_lines(stmts, 3, positions, True)
    assert result == 3


def test_head_effective_lines_only_docstring_in_head():
    # split_idx=1 with docstring: first_non_doc_idx=1 >= split_idx=1 → returns 1
    src = 'def foo():\n    """doc"""\n    x = 1\n    y = 2\n'
    stmts, positions, lines = _parse_func(src)
    result = _head_effective_lines(stmts, 1, positions, True)
    assert result == 1


def _make_long_func(n_stmts: int, func_name: str = "long_func") -> str:
    """Build a function with n_stmts independent assignments."""
    lines = [f"def {func_name}():\n"]
    for i in range(n_stmts):
        lines.append(f"    a{i} = {i}\n")
    lines.append("    return 0\n")
    return "".join(lines)


def test_function_splitter_under_limits_no_op():
    # A small function should not be split
    src = "def small():\n    x = 1\n    return x\n"
    splitter = FunctionSplitter([(1, 10)], source=src, verbose=False)
    assert splitter.get_rewritten_source() is None


def test_function_splitter_parse_error_no_crash():
    # Invalid source should not crash
    splitter = FunctionSplitter([(1, 10)], source="def f(\n  !!invalid", verbose=False)
    assert splitter.get_rewritten_source() is None


def test_function_splitter_out_of_range_no_op():
    # Function exists but is outside changed ranges
    src = _make_long_func(80)
    splitter = FunctionSplitter([(200, 300)], source=src, verbose=False, max_lines=10)
    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_over_line_limit(mock_anthropic):
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["process_tail"])
    )
    src = _make_long_func(80)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],
            source=src,
            verbose=False,
            max_lines=50,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    assert "_process_tail" in result
    assert "return _process_tail(" in result
    assert len(splitter.changes_made) >= 1


@patch("crispen.llm_client.anthropic")
def test_function_splitter_nested_funcdef_not_split(mock_anthropic):
    # A long function containing a nested funcdef should never be split,
    # even if it far exceeds the line limit.  Splitting across a closure
    # boundary produces cascading re-splits and semantically fragile helpers.
    lines = ["def func_with_closure():\n"]
    for i in range(80):
        lines.append(f"    a{i} = {i}\n")
    lines.append("    def inner():\n")
    lines.append("        return 0\n")
    lines.append("    return inner()\n")
    src = "".join(lines)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=10
        )

    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_async_skipped(mock_anthropic):
    # Async functions should not be split
    src = (
        "async def foo():\n"
        + "".join(f"    a{i} = {i}\n" for i in range(80))
        + "    return 0\n"
    )

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=10
        )

    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_generator_skipped(mock_anthropic):
    # Generator functions should not be split
    src = (
        "def gen():\n"
        + "".join(f"    a{i} = {i}\n" for i in range(80))
        + "    yield 0\n"
    )

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=10
        )

    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_llm_fallback_on_api_error(mock_anthropic):
    # API key not set → get_api_key raises CrispenAPIError → fallback names used
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["tail"])
    )
    src = _make_long_func(60, "my_func")

    # No ANTHROPIC_API_KEY → get_api_key raises → fallback to "my_func_helper"
    with patch.dict("os.environ", {}, clear=True):
        # Remove any existing API key
        import os

        os.environ.pop("ANTHROPIC_API_KEY", None)
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=30
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    # Fallback name used: "my_func_helper"
    assert "_my_func_helper" in result


@patch("crispen.llm_client.anthropic")
def test_function_splitter_recursive_split(mock_anthropic):
    # With small max_lines and broad changed_ranges, triggers multiple iterations
    # First call names helper for first function, second call for helper
    mock_anthropic.Anthropic.return_value.messages.create.side_effect = [
        _make_mock_response(["part1"]),
        _make_mock_response(["part2"]),
        _make_mock_response(["part3"]),
    ]

    # 13 body statements → with max_lines=5, needs multiple splits
    src = _make_long_func(13, "func")

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],  # broad range covers all helpers too
            source=src,
            verbose=False,
            max_lines=5,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    # Multiple splits occurred
    assert len(splitter.changes_made) >= 2


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
def test_function_splitter_with_helper_docstrings(mock_anthropic):
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["process"])
    )
    src = _make_long_func(80)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],
            source=src,
            verbose=False,
            max_lines=50,
            helper_docstrings=True,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    assert '"""' in result


@patch("crispen.llm_client.anthropic")
def test_function_splitter_class_method(mock_anthropic):
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["tail_work"])
    )
    lines = ["class Foo:\n", "    def method(self):\n"]
    for i in range(80):
        lines.append(f"        a{i} = {i}\n")
    lines.append("        return 0\n")
    src = "".join(lines)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)],
            source=src,
            verbose=False,
            max_lines=50,
        )

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    # Class methods use staticmethod and ClassName._ call
    assert "@staticmethod" in result
    assert "Foo._tail_work(" in result


def test_function_splitter_empty_source():
    """FunctionSplitter created with no source does nothing."""
    splitter = FunctionSplitter([(1, 10)])
    assert splitter.get_rewritten_source() is None


@patch("crispen.llm_client.anthropic")
def test_function_splitter_max_iterations_loop_exhausted(mock_anthropic):
    """Loop runs to completion (no break) when max iterations reached."""
    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_mock_response(["helper"])
    )
    src = _make_long_func(80, "foo")

    # Patch _MAX_SPLIT_ITERATIONS to 1 → loop runs exactly once without breaking
    # (break only occurs at START of next iteration when tasks=[])
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("crispen.refactors.function_splitter._MAX_SPLIT_ITERATIONS", 1):
            splitter = FunctionSplitter(
                [(1, 1000)], source=src, verbose=False, max_lines=50
            )

    result = splitter.get_rewritten_source()
    assert result is not None
    assert len(splitter.changes_made) == 1


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

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(
            [(1, 1000)], source=src, verbose=False, max_lines=50
        )

    # collision detected → task dropped → no rewrite
    assert splitter.get_rewritten_source() is None
