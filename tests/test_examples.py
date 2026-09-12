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

from crispen.config import CrispenConfig
from crispen.diff_parser import parse_diff
from crispen.file_limiter.runner import run_file_limiter
from crispen.refactors.cross_file_duplicate import run_cross_file_duplicate_extraction
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _build_repo_function_index,
)
from crispen.refactors.function_splitter import FunctionSplitter
from crispen.refactors.if_not_else import IfNotElse
from crispen.refactors.tuple_dataclass import TupleDataclass
from crispen.repo_index import build_repo_index

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


def _make_veto_response(is_valid: bool, reason: str = "test") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "evaluate_duplicate"
    block.input = {"is_valid_duplicate": is_valid, "reason": reason}
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_extract_response(data: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "extract_helper"
    block.input = data
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_verify_response(is_correct: bool, issues: list) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "verify_extraction"
    block.input = {"is_correct": is_correct, "issues": issues}
    resp = MagicMock()
    resp.content = [block]
    return resp


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
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    return mock_response


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
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
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
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
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
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        de = DuplicateExtractor(ranges, source=src)

    assert de._new_source is None
    mock_client.messages.create.assert_not_called()


def test_duplicate_extraction_cross_module(tmp_path, monkeypatch):
    """Same 4-statement function body duplicated across two files → extracted
    into a new shared module at their common ancestor package, both call
    sites rewritten to import and call it. Exercises
    ``run_cross_file_duplicate_extraction`` directly (the pass
    ``cross_file_duplicate.py`` runs, not covered by any other example — the
    existing duplicate_extraction/ examples are all single-file)."""
    base = EXAMPLES / "duplicate_extraction" / "04_cross_module"
    a_src = (base / "a_input.py").read_text()
    a_diff = (base / "a_diff.patch").read_text()
    b_src = (base / "b_input.py").read_text()
    b_diff = (base / "b_diff.patch").read_text()

    pkg = tmp_path / "svc"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    a_file = pkg / "orders.py"
    b_file = pkg / "invoices.py"
    a_file.write_text(a_src, encoding="utf-8")
    b_file.write_text(b_src, encoding="utf-8")

    per_file = {
        str(f): {
            "original": src,
            "source": src,
            "msgs": [],
            "candidates": {},
            "ranges": ranges,
        }
        for f, src, ranges in (
            (a_file, a_src, _ranges(a_diff, filename="a_input.py")),
            (b_file, b_src, _ranges(b_diff, filename="b_input.py")),
        )
    }

    helper_dict = {
        "function_name": "normalize_customer_ref",
        "helper_source": (
            "def normalize_customer_ref(payload):\n"
            '    ref = payload["customer_ref"]\n'
            "    normalized = ref.strip().upper()\n"
            '    tag = normalized.replace("-", "")\n'
            "    return tag\n"
        ),
        "call_site_replacements": [
            "    return normalize_customer_ref(payload)\n",
            "    return normalize_customer_ref(payload)\n",
        ],
    }

    # Cross-file mode's extract tool is named "extract_cross_file_helper",
    # distinct from same-file DuplicateExtractor's "extract_helper" tool
    # used by _make_extract_response above.
    extract_block = MagicMock()
    extract_block.type = "tool_use"
    extract_block.name = "extract_cross_file_helper"
    extract_block.input = helper_dict
    extract_response = MagicMock()
    extract_response.content = [extract_block]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "identical customer-ref normalization"),
            extract_response,
            _make_verify_response(True, []),
        ]
        msgs = list(
            run_cross_file_duplicate_extraction(
                per_file, str(tmp_path), CrispenConfig(), verbose=True
            )
        )

    assert any("extracted 'normalize_customer_ref'" in m for m in msgs)
    assert any("across 2 files" in m for m in msgs)

    helper_file = pkg / "common.py"
    assert helper_file.exists()
    compile(helper_file.read_text(encoding="utf-8"), "<common.py>", "exec")
    assert "def normalize_customer_ref(payload):" in helper_file.read_text(
        encoding="utf-8"
    )

    for f in (a_file, b_file):
        new_src = per_file[str(f)]["source"]
        compile(new_src, f"<{f.name}>", "exec")
        assert "from svc.common import normalize_customer_ref" in new_src
        assert "normalize_customer_ref(payload)" in new_src
        assert 'replace("-", "")' not in new_src  # original block replaced


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


def test_match_existing_function_repo_wide(tmp_path, monkeypatch):
    """Block matches a function defined in a different module, found via a
    real repo-wide scan (``build_repo_index``/``_build_repo_function_index``
    over real files on disk, not a hand-built index like the unit tests in
    test_duplicate_extractor.py use)."""
    base = EXAMPLES / "match_existing_function" / "02_repo_wide"
    src = (base / "input.py").read_text()
    diff = (base / "diff.patch").read_text()
    helper_src = (base / "helpers.py").read_text()
    ranges = _ranges(diff)

    pkg = tmp_path / "billing"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    input_file = pkg / "dashboard.py"
    input_file.write_text(src, encoding="utf-8")
    (pkg / "helpers.py").write_text(helper_src, encoding="utf-8")

    repo_index = build_repo_index(str(tmp_path))
    repo_function_index = _build_repo_function_index(repo_index)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            return_value=(True, "same operation", ""),
        ),
    ):
        de = DuplicateExtractor(
            ranges,
            source=src,
            current_file=str(input_file),
            match_functions_scope="repo",
            repo_function_index=repo_function_index,
            repo_index=repo_index,
        )

    assert de._new_source is not None
    compile(de._new_source, "<test>", "exec")
    assert "from billing.helpers import _summarize_pending_report" in de._new_source
    assert "_summarize_pending_report()" in de._new_source
    assert "fetch_pending_rows" not in de._new_source


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


@patch("crispen.llm_client.anthropic")
def test_function_splitter_skip_async(mock_anthropic):
    """Async functions are never split — no LLM call is made."""
    src, diff, expected = _load("function_splitter", "03_skip_async")
    ranges = _ranges(diff)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(ranges, source=src, verbose=False, max_lines=8)

    assert splitter.get_rewritten_source() is None
    mock_anthropic.Anthropic.return_value.messages.create.assert_not_called()


@patch("crispen.llm_client.anthropic")
def test_function_splitter_skip_generator(mock_anthropic):
    """Generator functions (containing yield) are never split — no LLM call is made."""
    src, diff, expected = _load("function_splitter", "04_skip_generator")
    ranges = _ranges(diff)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        splitter = FunctionSplitter(ranges, source=src, verbose=False, max_lines=8)

    assert splitter.get_rewritten_source() is None
    mock_anthropic.Anthropic.return_value.messages.create.assert_not_called()


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


# ===========================================================================
# file_limiter examples
# ===========================================================================


def _load_fl(name: str) -> tuple[str, str, str]:
    """Return (original_src, input_src, diff_text) for a file_limiter example."""
    base = EXAMPLES / "file_limiter" / name
    original_path = base / "original.py"
    original_src = original_path.read_text() if original_path.exists() else ""
    return (
        original_src,
        (base / "input.py").read_text(),
        (base / "diff.patch").read_text(),
    )


def _make_fl_response(tool_name: str, data: dict) -> MagicMock:
    """Build a mock Anthropic tool-use response for a FileLimiter LLM call."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = data
    resp = MagicMock()
    resp.content = [block]
    return resp


@patch("crispen.llm_client.anthropic")
def test_file_limiter_set2_split(mock_anthropic, tmp_path):
    """New public helpers added by the diff are moved to a sibling utils.py file."""
    original_src, src, diff = _load_fl("01_set2_split")
    ranges = _ranges(diff)

    input_file = tmp_path / "input.py"
    input_file.write_text(src)

    mock_client = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    mock_anthropic.APIError = Exception
    # Two LLM calls: propose output files, then assign_file_placements.
    mock_client.messages.create.side_effect = [
        _make_fl_response(
            "propose_output_files",
            {"files": [{"filename": "utils.py", "description": "utility helpers"}]},
        ),
        _make_fl_response(
            "assign_file_placements",
            {
                "placements": [
                    {"group_id": 0, "target_file": "utils.py"},
                    {"group_id": 1, "target_file": "utils.py"},
                ]
            },
        ),
    ]

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        result = run_file_limiter(
            str(input_file),
            original_source=original_src,
            post_source=src,
            diff_ranges=ranges,
            config=CrispenConfig(),
        )

    assert not result.abort
    assert "utils.py" in result.new_files
    new_src = result.new_files["utils.py"]
    assert "normalize_name" in new_src
    assert "validate_record" in new_src
    compile(new_src, "<utils.py>", "exec")
    compile(result.original_source, "<input.py>", "exec")


def test_file_limiter_skip_single_scc(tmp_path):
    """A file where all entities form one dependency cycle cannot be split."""
    _, src, diff = _load_fl("02_skip_single_scc")
    ranges = _ranges(diff)

    input_file = tmp_path / "input.py"
    input_file.write_text(src)

    result = run_file_limiter(
        str(input_file),
        original_source="",
        post_source=src,
        diff_ranges=ranges,
        config=CrispenConfig(file_limiter_subdir_split=False),
    )

    assert result.abort
    assert not result.new_files
    assert any("cannot be split" in m for m in result.messages)
