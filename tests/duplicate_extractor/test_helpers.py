import textwrap
from unittest.mock import MagicMock, patch
import libcst as cst
from libcst.metadata import MetadataWrapper
from crispen.refactors.duplicate_extractor import (
    _FunctionCollector,
    _FunctionInfo,
    _SeqInfo,
    _SequenceCollector,
    DuplicateExtractor,
)


def _parse_stmt(src: str) -> cst.BaseStatement:
    return cst.parse_module(src).body[0]


def _make_seq(start: int, end: int) -> _SeqInfo:
    return _SeqInfo(
        stmts=[],
        start_line=start,
        end_line=end,
        scope="<module>",
        source="",
        fingerprint="",
    )


def _make_seq_with_source(source: str) -> _SeqInfo:
    return _SeqInfo(
        stmts=[], start_line=1, end_line=1, scope="f", source=source, fingerprint=""
    )


def _make_esc_seq(start: int, end: int) -> _SeqInfo:
    """Create a _SeqInfo for escaping-vars tests."""
    return _SeqInfo(
        stmts=[],
        start_line=start,
        end_line=end,
        scope="foo",
        source="",
        fingerprint="",
    )


def _make_verify_response(is_correct: bool, issues: list) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "verify_extraction"
    block.input = {"is_correct": is_correct, "issues": issues}
    resp = MagicMock()
    resp.content = [block]
    return resp


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


def _make_func_info(name: str, body_source: str = "    pass\n") -> _FunctionInfo:
    return _FunctionInfo(
        name=name,
        source=f"def {name}():\n{body_source}",
        scope="<module>",
        body_source=body_source,
        body_stmt_count=1,
        params=[],
    )


def _collect_sequences(source: str, max_seq_len: int = 8):
    tree = cst.parse_module(source)
    lines = source.splitlines(keepends=True)
    collector = _SequenceCollector(lines, max_seq_len=max_seq_len)
    MetadataWrapper(tree).visit(collector)
    return collector.sequences


def _collect_functions(source: str):
    tree = cst.parse_module(source)
    lines = source.splitlines(keepends=True)
    collector = _FunctionCollector(lines)
    MetadataWrapper(tree).visit(collector)
    return collector.functions


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


def _make_invalid_assembled_extractor(monkeypatch, verbose=True):
    """Helper: DuplicateExtractor where _apply_edits returns invalid Python."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic") as mock_anthropic,
        patch(
            "crispen.refactors.duplicate_extractor._apply_edits",
            return_value="def f(:\n    pass\n",  # invalid Python
        ),
    ):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": "def _helper(x):\n    pass\n",
                    "call_site_replacements": [
                        "    _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
        ]
        return DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=verbose,
            extraction_retries=0,
            llm_verify_retries=0,
        )


def _make_pyflakes_check_extractor(monkeypatch, verbose=True):
    """Helper: extraction that passes compile() but pyflakes finds a new undefined
    name."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic") as mock_anthropic,
        patch(
            "crispen.refactors.duplicate_extractor._pyflakes_new_undefined_names",
            return_value={"mock_client"},
        ),
    ):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": "def _helper(x):\n    pass\n",
                    "call_site_replacements": [
                        "    _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
        ]
        return DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=verbose,
            extraction_retries=0,
            llm_verify_retries=0,
        )


def _make_missing_free_vars_extractor(monkeypatch, verbose=True):
    """Helper: extraction that passes all earlier guards but _missing_free_vars
    detects a free variable absent from the replacement."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic") as mock_anthropic,
        patch(
            "crispen.refactors.duplicate_extractor._missing_free_vars",
            return_value={"new_source"},
        ),
    ):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": "def _helper(x):\n    pass\n",
                    "call_site_replacements": [
                        "    _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
        ]
        return DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=verbose,
            extraction_retries=0,
            llm_verify_retries=0,
        )


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


def _make_new_attr_extractor(monkeypatch, verbose=True):
    """Helper: LLM returns a helper that calls a method not in the original source."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "helper",
                    "placement": "module_level",
                    # helper calls .invented_method() — not present in _DUP_SOURCE
                    "helper_source": (
                        "def helper(data):\n" "    data.invented_method()\n"
                    ),
                    "call_site_replacements": [
                        "helper(data)\n",
                        "helper(data)\n",
                    ],
                }
            ),
        ]
        return DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=verbose,
            extraction_retries=0,
            llm_verify_retries=0,
        )


def _make_no_call_extractor(monkeypatch, verbose=True):
    """Helper: LLM returns call replacements that don't call the helper function."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": "def _helper(data):\n    pass\n",
                    # Call replacements don't reference _helper at all.
                    "call_site_replacements": [
                        "    pass\n",
                        "    pass\n",
                    ],
                }
            ),
        ]
        return DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=verbose,
            extraction_retries=0,
            llm_verify_retries=0,
        )


def _make_seq_info(start: int, end: int, src: str = "") -> _SeqInfo:
    return _SeqInfo(
        stmts=[],
        start_line=start,
        end_line=end,
        scope="foo",
        source=src,
        fingerprint="",
    )


def _make_veto_func_match_response(is_valid: bool, reason: str = "test") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "evaluate_duplicate"
    block.input = {"is_valid_duplicate": is_valid, "reason": reason}
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_call_gen_response(replacement: str) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "generate_call"
    block.input = {"replacement": replacement}
    resp = MagicMock()
    resp.content = [block]
    return resp


# _setup() has no params; called by main() → in func_body_fps.
# foo.body fingerprint == _setup.body fingerprint.
# Diff range (2, 9) covers both _setup.body (2-4) AND foo.body (7-9).
# _setup.body hits the func.name==seq.scope True branch (skipped).
# foo.body hits the False branch and proceeds to veto → replace.
_FUNC_MATCH_SOURCE = textwrap.dedent(
    """\
    def _setup():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def main():
        _setup()
    """
)
_FUNC_MATCH_RANGES = [(2, 9)]  # covers _setup.body AND foo.body

# _process(val) has one param; called by main() → in func_body_fps.
# foo.body fingerprint == _process.body fingerprint (names normalized).
# Diff range covers foo.body only.
_FUNC_MATCH_PARAM_SOURCE = textwrap.dedent(
    """\
    def _process(val):
        y = transform(val)
        z = finalize(y)
        return z

    def foo():
        y = transform(data)
        z = finalize(y)
        return z

    def main():
        _process(data)
    """
)
_FUNC_MATCH_PARAM_RANGES = [(6, 9)]  # overlaps foo.body only

# Source with a function-match AND an independent duplicate group.
# bar/baz use an if-else structure so no sub-window of their bodies matches
# _setup's 3-chained-assignment fingerprint.
_FUNC_MATCH_THEN_DUP_SOURCE = textwrap.dedent(
    """\
    def _setup():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def bar():
        if condition:
            result = process(items)
        else:
            result = fallback(items)
        store(result)

    def baz():
        if condition:
            result = process(items)
        else:
            result = fallback(items)
        store(result)

    def main():
        _setup()
    """
)
_FUNC_MATCH_THEN_DUP_RANGES = [(2, 23)]  # covers foo, bar, baz bodies

# Source that already defines _helper AND has duplicate blocks.
_COLLISION_SOURCE = textwrap.dedent(
    """\
    def _helper(x):
        return x

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
_COLLISION_RANGES = [(9, 11)]  # overlaps bar's body


def _make_steal_seq(end_line: int) -> _SeqInfo:
    return _SeqInfo(
        stmts=[], start_line=1, end_line=end_line, scope="f", source="", fingerprint=""
    )


_RETURN_BLOCK_SOURCE = textwrap.dedent(
    """\
    def foo():
        x = compute(data)
        y = transform(x)
        return y

    def bar():
        x = compute(data)
        y = transform(x)
        return y
    """
)
_RETURN_BLOCK_RANGES = [(7, 9)]  # overlaps bar's body


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


_PARAM_DUP_SOURCE = textwrap.dedent(
    """\
    def test_a(mock_client):
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def test_b(mock_client):
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_PARAM_DUP_RANGES = [(7, 9)]  # overlaps test_b's body


def _make_import_local_extract_response():
    return _make_extract_response(
        {
            "function_name": "_helper",
            "placement": "module_level",
            # helper imports mock_client instead of taking it as a parameter
            "helper_source": (
                "def _helper():\n"
                "    import mock_client\n"
                "    x = compute(data)\n"
                "    y = transform(x)\n"
                "    z = finalize(y)\n"
            ),
            "call_site_replacements": [
                "    _helper()\n",
                "    _helper()\n",
            ],
        }
    )


def _make_veto_response_with_notes(
    is_valid: bool, reason: str, notes: str
) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "evaluate_duplicate"
    block.input = {
        "is_valid_duplicate": is_valid,
        "reason": reason,
        "extraction_notes": notes,
    }
    resp = MagicMock()
    resp.content = [block]
    return resp
