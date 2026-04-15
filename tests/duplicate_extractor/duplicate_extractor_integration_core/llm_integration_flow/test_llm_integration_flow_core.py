from unittest.mock import MagicMock, patch
import textwrap
from crispen.errors import CrispenAPIError
from crispen.refactors.duplicate_extractor import DuplicateExtractor
import pytest


def _make_verify_response(is_correct: bool, issues: list) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "verify_extraction"
    block.input = {"is_correct": is_correct, "issues": issues}
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


def _make_veto_response(is_valid: bool, reason: str = "test") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "evaluate_duplicate"
    block.input = {"is_valid_duplicate": is_valid, "reason": reason}
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_no_source_no_analysis():
    de = DuplicateExtractor([(1, 5)])
    assert de._new_source is None
    assert de.get_rewritten_source() is None


def test_no_duplicates_no_llm_calls(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    source = textwrap.dedent(
        """\
        def foo():
            x = a + b
            y = x * 2

        def bar():
            if condition:
                result = value
            else:
                result = other
        """
    )
    # Structurally different blocks → no duplicate group → no API calls needed
    de = DuplicateExtractor([(6, 9)], source=source)
    assert de._new_source is None


_DUP_SOURCE = textwrap.dedent(
    """\
    def foo():
        if debug:
            pass
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def bar():
        result = None
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_DUP_RANGES = [(10, 12)]  # overlaps bar's duplicate block (x/y/z lines)

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
        result = None
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_ESC_RANGES = [(9, 11)]  # overlaps bar's duplicate block (x/y/z lines)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(CrispenAPIError, match="ANTHROPIC_API_KEY"):
        DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE)


def test_api_error_in_veto_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = Exception("rate limit")

        with pytest.raises(CrispenAPIError):
            DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE)


def test_api_error_in_extract_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        # First call (veto) succeeds, second call (extract) fails
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            Exception("rate limit"),
        ]

        with pytest.raises(CrispenAPIError):
            DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE)


def test_parse_error_in_analyze(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic.Anthropic"):
        # Invalid Python: _analyze should return silently
        de = DuplicateExtractor([(1, 1)], source="def f(: pass")
    assert de._new_source is None


def test_veto_rejects_no_changes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.return_value = _make_veto_response(False)

        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE)

    assert de._new_source is None
    assert de.changes_made == []


def test_wrong_replacement_count_skipped(monkeypatch):
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
                    "helper_source": "def helper():\n    pass\n",
                    "call_site_replacements": ["helper()\n"],  # should be 2
                }
            ),
        ]

        de = DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None


def test_wrong_replacement_count_skipped_verbose_false(monkeypatch):
    # verbose=False covers the False branch of the new if-self.verbose guard.
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
                    "helper_source": "def helper():\n    pass\n",
                    "call_site_replacements": ["helper()\n"],  # should be 2
                }
            ),
        ]

        de = DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=False,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None


def test_escaping_vars_passed_to_extract(monkeypatch):
    # foo's block assigns z; foo uses z after the block.
    # _find_escaping_vars returns {"z"}, which is passed to _llm_extract.
    # The extraction prompt must contain the note instructing the LLM to return z.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    helper_src = (
        "def _helper(data):\n"
        "    x = compute(data)\n"
        "    y = transform(x)\n"
        "    z = finalize(y)\n"
        "    return z\n"
    )
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
                    "helper_source": helper_src,
                    "call_site_replacements": [
                        "    z = _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor(_ESC_RANGES, source=_ESC_SOURCE)

    # The extraction prompt must include the escaping-variable note.
    extract_call = mock_client.messages.create.call_args_list[1]
    extract_prompt = extract_call.kwargs["messages"][0]["content"]
    assert "immediately follows the block" in extract_prompt
    assert de._new_source is not None


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


def test_invalid_assembled_source_skipped(monkeypatch):
    # Individual components pass _verify_extraction but the per-group assembled
    # edit is invalid Python — the group is skipped without poisoning others.
    de = _make_invalid_assembled_extractor(monkeypatch)
    assert de._new_source is None
    assert de.changes_made == []


def test_invalid_assembled_source_skipped_verbose_false(monkeypatch):
    # verbose=False: per-group compile-failure log suppressed (covers False branch).
    de = _make_invalid_assembled_extractor(monkeypatch, verbose=False)
    assert de._new_source is None


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


def test_pyflakes_check_skips_group_verbose(monkeypatch, capsys):
    # Pyflakes finds a new undefined name → group is skipped (verbose path).
    de = _make_pyflakes_check_extractor(monkeypatch, verbose=True)
    assert de._new_source is None
    assert (
        "undefined name(s) introduced by edit: mock_client" in capsys.readouterr().err
    )


def test_pyflakes_check_skips_group_verbose_false(monkeypatch):
    # verbose=False: pyflakes failure is silent.
    de = _make_pyflakes_check_extractor(monkeypatch, verbose=False)
    assert de._new_source is None


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


def test_missing_free_vars_check_skips_group_verbose(monkeypatch, capsys):
    # _missing_free_vars returns a non-empty set → group is rejected (verbose).
    de = _make_missing_free_vars_extractor(monkeypatch, verbose=True)
    assert de._new_source is None
    assert (
        "free variable(s) from original block missing in replacement: new_source"
        in capsys.readouterr().err
    )


def test_missing_free_vars_check_skips_group_verbose_false(monkeypatch):
    # verbose=False: _missing_free_vars failure is silent.
    de = _make_missing_free_vars_extractor(monkeypatch, verbose=False)
    assert de._new_source is None


def test_verify_fails_skipped(monkeypatch):
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
                    "helper_source": "def helper(x:\n    pass\n",  # unclosed paren
                    "call_site_replacements": [
                        "helper(data)\n",
                        "helper(data)\n",
                    ],
                }
            ),
        ]

        de = DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None


def test_verify_fails_skipped_verbose_false(monkeypatch):
    # verbose=False covers the False branch of the new if-self.verbose guard.
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
                    "helper_source": "def helper(x:\n    pass\n",  # unclosed paren
                    "call_site_replacements": [
                        "helper(data)\n",
                        "helper(data)\n",
                    ],
                }
            ),
        ]

        de = DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=False,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None


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


def test_new_attribute_check_skips_group_verbose(monkeypatch, capsys):
    de = _make_new_attr_extractor(monkeypatch, verbose=True)
    assert de._new_source is None
    assert "new attribute access" in capsys.readouterr().err


def test_new_attribute_check_skips_group_verbose_false(monkeypatch):
    de = _make_new_attr_extractor(monkeypatch, verbose=False)
    assert de._new_source is None


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


def test_no_call_check_skips_group_verbose(monkeypatch, capsys):
    de = _make_no_call_extractor(monkeypatch, verbose=True)
    assert de._new_source is None
    assert "not called in candidate output" in capsys.readouterr().err


def test_no_call_check_skips_group_verbose_false(monkeypatch):
    de = _make_no_call_extractor(monkeypatch, verbose=False)
    assert de._new_source is None
