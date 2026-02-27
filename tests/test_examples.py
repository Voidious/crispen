"""Full-stack example tests.

Each test loads a realistic code diff from the examples/ directory,
runs the relevant refactor, and verifies the expected outcome.  LLM
calls are mocked so the suite runs fast and offline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import libcst as cst
from libcst.metadata import MetadataWrapper

from crispen.diff_parser import parse_diff
from crispen.refactors.duplicate_extractor import DuplicateExtractor
from crispen.refactors.function_splitter import FunctionSplitter
from crispen.refactors.if_not_else import IfNotElse
from crispen.refactors.tuple_dataclass import TupleDataclass

EXAMPLES = Path(__file__).parent.parent / "examples"


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------


def _load(category: str, name: str) -> tuple[str, str, str]:
    """Return (input_src, diff_text, expected_src) for an example."""
    base = EXAMPLES / category / name
    return (
        (base / "input.py").read_text(),
        (base / "diff.patch").read_text(),
        (base / "expected.py").read_text(),
    )


def _ranges(diff_text: str, filename: str = "input.py") -> list[tuple[int, int]]:
    """Parse diff and return changed ranges for filename."""
    return parse_diff(diff_text)[filename]


# ---------------------------------------------------------------------------
# Deterministic refactor helpers
# ---------------------------------------------------------------------------


def _apply_if_not_else(source: str, ranges: list) -> str:
    tree = cst.parse_module(source)
    wrapper = MetadataWrapper(tree)
    transformer = IfNotElse(ranges)
    return wrapper.visit(transformer).code


def _apply_tuple_dataclass(source: str, ranges: list, min_size: int = 4) -> str:
    tree = cst.parse_module(source)
    wrapper = MetadataWrapper(tree)
    transformer = TupleDataclass(ranges, min_size=min_size, source=source)
    return wrapper.visit(transformer).code


# ---------------------------------------------------------------------------
# LLM mock helpers — DuplicateExtractor
# ---------------------------------------------------------------------------


def _wrap_response(block: MagicMock) -> MagicMock:
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_veto_response(is_valid: bool, reason: str = "test") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "evaluate_duplicate"
    block.input = {"is_valid_duplicate": is_valid, "reason": reason}
    return _wrap_response(block)


def _make_extract_response(data: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "extract_helper"
    block.input = data
    return _wrap_response(block)


def _make_verify_response(is_correct: bool, issues: list) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "verify_extraction"
    block.input = {"is_correct": is_correct, "issues": issues}
    return _wrap_response(block)


# ---------------------------------------------------------------------------
# LLM mock helpers — FunctionSplitter
# ---------------------------------------------------------------------------


def _make_naming_response(names: list[str]) -> MagicMock:
    """Build a mock Anthropic response for the name_helper_functions tool."""
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "name_helper_functions"
    mock_block.input = {
        "names": [{"id": str(i), "name": n} for i, n in enumerate(names)]
    }
    return _wrap_response(mock_block)


# ===========================================================================
# if_not_else examples
# ===========================================================================


def test_if_not_else_basic_flip():
    """Simple `if not x: A else B` is flipped to `if x: B else A`."""
    src, diff, expected = _load("if_not_else", "01_basic_flip")
    result = _apply_if_not_else(src, _ranges(diff))
    assert result == expected


def test_if_not_else_compound_condition():
    """Compound negated condition — the parenthesised expression is preserved."""
    src, diff, expected = _load("if_not_else", "02_compound_condition")
    result = _apply_if_not_else(src, _ranges(diff))
    assert result == expected


def test_if_not_else_multi_body():
    """Both branches contain multiple statements — all are swapped correctly."""
    src, diff, expected = _load("if_not_else", "03_multi_body")
    result = _apply_if_not_else(src, _ranges(diff))
    assert result == expected


def test_if_not_else_partial_range():
    """Only the newly added function is transformed; the pre-existing one is skipped."""
    src, diff, expected = _load("if_not_else", "04_partial_range")
    result = _apply_if_not_else(src, _ranges(diff))
    assert result == expected


def test_if_not_else_elif_skip():
    """`if not ... elif ...` chains are not flipped — source returned unchanged."""
    src, diff, expected = _load("if_not_else", "05_elif_skip")
    result = _apply_if_not_else(src, _ranges(diff))
    assert result == expected


# ===========================================================================
# tuple_dataclass examples
# ===========================================================================


def test_tuple_dataclass_named_fields():
    """4-element return tuple with Name-valued elements → dataclass."""
    src, diff, expected = _load("tuple_dataclass", "01_named_fields")
    result = _apply_tuple_dataclass(src, _ranges(diff))
    assert result == expected


def test_tuple_dataclass_multi_return():
    """Multiple return paths use consistent field names derived from the first path."""
    src, diff, expected = _load("tuple_dataclass", "02_multi_return")
    result = _apply_tuple_dataclass(src, _ranges(diff))
    assert result == expected


def test_tuple_dataclass_non_unpack_caller():
    """A caller that stores the return value without unpacking blocks the transform."""
    src, diff, expected = _load("tuple_dataclass", "03_non_unpack_caller")
    result = _apply_tuple_dataclass(src, _ranges(diff))
    assert result == expected  # unchanged


def test_tuple_dataclass_small_tuple():
    """A 3-element tuple is below the default min_size=4 threshold — skipped."""
    src, diff, expected = _load("tuple_dataclass", "04_small_tuple")
    result = _apply_tuple_dataclass(src, _ranges(diff))
    assert result == expected  # unchanged


# ===========================================================================
# duplicate_extraction examples
# ===========================================================================


def _setup_mock_anthropic(mock_anthropic):
    mock_client = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    mock_anthropic.APIError = Exception
    return mock_client


def test_duplicate_extraction_cross_function(monkeypatch):
    """Same 3-statement setup block duplicated across two functions → extracted."""
    src, diff, _ = _load("duplicate_extraction", "01_cross_function")
    ranges = _ranges(diff)

    helper_src = (
        "def _setup_report_resources():\n"
        "    config = load_config()\n"
        "    db = Database(config.db_url)\n"
        "    formatter = ReportFormatter(config.format)\n"
        "    return config, db, formatter\n"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = _setup_mock_anthropic(mock_anthropic)
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "identical setup block"),
            _make_extract_response(
                {
                    "function_name": "_setup_report_resources",
                    "placement": "module_level",
                    "helper_source": helper_src,
                    "call_site_replacements": [
                        "    config, db, formatter = _setup_report_resources()\n",
                        "    config, db, formatter = _setup_report_resources()\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor(ranges, source=src)

    assert de._new_source is not None
    compile(de._new_source, "<test>", "exec")
    assert "def _setup_report_resources" in de._new_source
    assert de._new_source.count("_setup_report_resources()") >= 2


def test_duplicate_extraction_within_function(monkeypatch):
    """Same 4-statement connection block repeated twice in one function → extracted."""
    src, diff, _ = _load("duplicate_extraction", "02_within_function")
    ranges = _ranges(diff)

    helper_src = (
        "def _open_connections(source_url, dest_url):\n"
        "    src = connect(source_url)\n"
        "    dest = connect(dest_url)\n"
        "    src.ping()\n"
        "    dest.ping()\n"
        "    return src, dest\n"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = _setup_mock_anthropic(mock_anthropic)
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "identical connection setup"),
            _make_extract_response(
                {
                    "function_name": "_open_connections",
                    "placement": "module_level",
                    "helper_source": helper_src,
                    "call_site_replacements": [
                        "    src, dest = _open_connections(source_url, dest_url)\n",
                        "    src, dest = _open_connections(source_url, dest_url)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor(ranges, source=src)

    assert de._new_source is not None
    compile(de._new_source, "<test>", "exec")
    assert "def _open_connections" in de._new_source
    assert de._new_source.count("_open_connections(") >= 2


def test_duplicate_extraction_below_threshold(monkeypatch):
    """Single-statement duplicate (weight 1 < min_duplicate_weight 3) → skipped."""
    src, diff, expected = _load("duplicate_extraction", "03_below_threshold")
    ranges = _ranges(diff)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = _setup_mock_anthropic(mock_anthropic)
        de = DuplicateExtractor(ranges, source=src)

    assert de._new_source is None
    mock_client.messages.create.assert_not_called()


# ===========================================================================
# match_existing_function examples
# ===========================================================================


def test_match_existing_function_no_arg_helper(monkeypatch):
    """Block in new function matches an existing no-arg helper → call replaces it."""
    src, diff, _ = _load("match_existing_function", "01_no_arg_helper")
    ranges = _ranges(diff)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            return_value=(True, "identical logging setup", ""),
        ),
    ):
        de = DuplicateExtractor(ranges, source=src)

    assert de._new_source is not None
    compile(de._new_source, "<test>", "exec")
    assert "_setup_logger()" in de._new_source


# ===========================================================================
# function_splitter examples
# ===========================================================================


@patch("crispen.llm_client.anthropic")
def test_function_splitter_module_level(mock_anthropic):
    """A long module-level function is split into head + private helper."""
    src, diff, _ = _load("function_splitter", "01_module_level")
    ranges = _ranges(diff)

    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_naming_response(["persist_results"])
    )
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(ranges, source=src, verbose=False, max_lines=20)

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    assert "_persist_results" in result
    assert "return _persist_results(" in result
    assert len(splitter.changes_made) >= 1


@patch("crispen.llm_client.anthropic")
def test_function_splitter_method(mock_anthropic):
    """A long class method is split; the helper becomes a @staticmethod."""
    src, diff, _ = _load("function_splitter", "02_method")
    ranges = _ranges(diff)

    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_naming_response(["compute_totals"])
    )
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(ranges, source=src, verbose=False, max_lines=20)

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    assert "_compute_totals" in result
    assert "@staticmethod" in result


def _assert_splitter_skips(diff: str, src: str, mock_anthropic: MagicMock) -> None:
    ranges = _ranges(diff)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(ranges, source=src, verbose=False, max_lines=8)

    assert splitter.get_rewritten_source() is None
    mock_anthropic.Anthropic.return_value.messages.create.assert_not_called()


@patch("crispen.llm_client.anthropic")
def test_function_splitter_skip_async(mock_anthropic):
    """Async functions are never split — no LLM call is made."""
    src, diff, expected = _load("function_splitter", "03_skip_async")
    _assert_splitter_skips(diff, src, mock_anthropic)


@patch("crispen.llm_client.anthropic")
def test_function_splitter_skip_generator(mock_anthropic):
    """Generator functions (containing yield) are never split — no LLM call is made."""
    src, diff, expected = _load("function_splitter", "04_skip_generator")
    _assert_splitter_skips(diff, src, mock_anthropic)


@patch("crispen.llm_client.anthropic")
def test_function_splitter_skip_nested_def(mock_anthropic):
    """Functions containing nested defs (closures) are never split — no LLM call."""
    src, diff, expected = _load("function_splitter", "05_skip_nested_def")
    ranges = _ranges(diff)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(ranges, source=src, verbose=False, max_lines=10)

    assert splitter.get_rewritten_source() is None
    mock_anthropic.Anthropic.return_value.messages.create.assert_not_called()


@patch("crispen.llm_client.anthropic")
def test_function_splitter_instance_method_helper(mock_anthropic):
    """A method whose tail references self → helper is a regular instance method."""
    src, diff, _ = _load("function_splitter", "06_instance_method")
    ranges = _ranges(diff)

    mock_anthropic.Anthropic.return_value.messages.create.return_value = (
        _make_naming_response(["build_summary"])
    )
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(ranges, source=src, verbose=False, max_lines=20)

    result = splitter.get_rewritten_source()
    assert result is not None
    compile(result, "<test>", "exec")
    assert "_build_summary" in result
    assert "@staticmethod" not in result
    assert "return self._build_summary(" in result
    assert len(splitter.changes_made) >= 1
