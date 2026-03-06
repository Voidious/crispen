import textwrap
from unittest.mock import MagicMock, patch
import pytest
from crispen.errors import CrispenAPIError
from crispen.refactors.duplicate_extractor import (
    _ApiTimeout,
    _FunctionInfo,
    _generate_no_arg_call,
    _llm_generate_call,
    _llm_veto_func_match,
    _run_with_timeout,
    DuplicateExtractor,
)
from .test_extraction_verification import (
    _make_extract_response,
    _make_seq_info,
    _make_verify_response,
    _make_veto_response,
)


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


def test_replacement_steals_post_block_skipped(monkeypatch):
    """Replacement whose last line matches the post-block line is rejected."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_do_work",
                    "placement": "module_level",
                    "helper_source": (
                        "def _do_work(data):\n"
                        "    x = compute(data)\n"
                        "    y = transform(x)\n"
                        "    z = finalize(y)\n"
                    ),
                    "call_site_replacements": [
                        "    _do_work(data)\n    return z\n",  # steals "return z"
                        "    _do_work(data)\n",
                    ],
                }
            ),
        ]
        de = DuplicateExtractor(
            _POST_STEAL_RANGES,
            source=_POST_STEAL_SOURCE,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None


def test_replacement_steals_post_block_skipped_verbose_false(monkeypatch):
    """verbose=False covers the False branch of the verbose guard."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_do_work",
                    "placement": "module_level",
                    "helper_source": (
                        "def _do_work(data):\n"
                        "    x = compute(data)\n"
                        "    y = transform(x)\n"
                        "    z = finalize(y)\n"
                    ),
                    "call_site_replacements": [
                        "    _do_work(data)\n    return z\n",  # steals "return z"
                        "    _do_work(data)\n",
                    ],
                }
            ),
        ]
        de = DuplicateExtractor(
            _POST_STEAL_RANGES,
            source=_POST_STEAL_SOURCE,
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


def _make_uncalled_in_combined_extractor(monkeypatch, verbose=True):
    """Simulate: per-group call check passes, but combined output lacks the call.

    Achieved by patching _has_call_to: returns True for the per-group check
    (first call) and False for the final combined check (second call).
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic") as mock_anthropic,
        patch(
            "crispen.refactors.duplicate_extractor._has_call_to",
            side_effect=[True, False],
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
                    "helper_source": "def _helper(data):\n    pass\n",
                    "call_site_replacements": [
                        "    _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        return DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE, verbose=verbose)


def test_uncalled_in_combined_drops_group_verbose(monkeypatch, capsys):
    de = _make_uncalled_in_combined_extractor(monkeypatch, verbose=True)
    assert de._new_source is None
    assert "DROPPED" in capsys.readouterr().err


def test_uncalled_in_combined_drops_group_verbose_false(monkeypatch):
    de = _make_uncalled_in_combined_extractor(monkeypatch, verbose=False)
    assert de._new_source is None


def test_llm_veto_skips_non_matching_blocks(monkeypatch):
    from crispen.refactors.duplicate_extractor import _llm_veto

    client = MagicMock()
    non_matching = MagicMock()
    non_matching.type = "text"  # not tool_use → if condition False
    matching = MagicMock()
    matching.type = "tool_use"
    matching.name = "evaluate_duplicate"
    matching.input = {"is_valid_duplicate": True, "reason": "same"}
    response = MagicMock()
    response.content = [non_matching, matching]
    client.messages.create.return_value = response

    group = [_make_seq_info(1, 3), _make_seq_info(5, 7)]
    is_valid, reason, _ = _llm_veto(client, group)
    assert is_valid is True


def test_llm_extract_skips_non_matching_blocks(monkeypatch):
    from crispen.refactors.duplicate_extractor import _llm_extract

    client = MagicMock()
    non_matching = MagicMock()
    non_matching.type = "text"  # not tool_use → if condition False
    matching = MagicMock()
    matching.type = "tool_use"
    matching.name = "extract_helper"
    matching.input = {
        "function_name": "helper",
        "placement": "module_level",
        "helper_source": "def helper(): pass\n",
        "call_site_replacements": ["helper()\n"],
    }
    response = MagicMock()
    response.content = [non_matching, matching]
    client.messages.create.return_value = response

    group = [_make_seq_info(1, 3)]
    result = _llm_extract(client, group, "a = 1\n")
    assert result is not None
    assert result["function_name"] == "helper"


# ---------------------------------------------------------------------------
# engine integration: CrispenAPIError propagates
def test_verbose_false_suppresses_stderr(monkeypatch):
    # verbose=False must take all four if-self.verbose False branches without printing.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        import os

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
    helper = "def _helper(data):\n    pass\n"
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
                    "helper_source": helper,
                    "call_site_replacements": [
                        "    _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]

        de = DuplicateExtractor([(9, 11)], source=source, verbose=False)

    assert de._new_source is not None
    assert "_helper" in de._new_source


def test_engine_propagates_api_error(tmp_path, monkeypatch):
    from crispen.config import CrispenConfig
    from crispen.engine import run_engine

    f = tmp_path / "code.py"
    f.write_text(_DUP_SOURCE, encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("crispen.engine.load_config", lambda: CrispenConfig())

    with pytest.raises(CrispenAPIError):
        list(run_engine({str(f): _DUP_RANGES}))


def test_cli_exits_on_api_error(tmp_path, monkeypatch):
    import io
    from crispen.cli import main
    from crispen.config import CrispenConfig

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("crispen.cli.load_config", lambda: CrispenConfig())
    monkeypatch.setattr("crispen.engine.load_config", lambda: CrispenConfig())

    # Write file so engine can read it
    f = tmp_path / "dup.py"
    f.write_text(_DUP_SOURCE, encoding="utf-8")

    diff = textwrap.dedent(
        f"""\
        --- a/{f}
        +++ b/{f}
        @@ -7,3 +7,3 @@
        -    x = compute(data)
        +    x = compute(data)
             y = transform(x)
             z = finalize(y)
        """
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(diff))

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_run_with_timeout_fires_on_slow_func():
    import threading

    barrier = threading.Event()
    try:
        with pytest.raises(_ApiTimeout):
            _run_with_timeout(barrier.wait, timeout=0.01)
    finally:
        barrier.set()  # allow the daemon thread to exit cleanly


def test_veto_timeout_skips_group(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            side_effect=_ApiTimeout("veto timed out"),
        ),
    ):
        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE)
    assert de._new_source is None
    assert de.changes_made == []


def test_extract_timeout_skips_group(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # First call (veto) returns success; second call (extract) times out.
    side_effects = [(True, "same logic", ""), _ApiTimeout("extract timed out")]

    def _mock_run(func, timeout, *args, **kwargs):
        result = side_effects.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            side_effect=_mock_run,
        ),
    ):
        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE)
    assert de._new_source is None


def _make_veto_func_match_response(is_valid: bool, reason: str = "test") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "evaluate_duplicate"
    block.input = {"is_valid_duplicate": is_valid, "reason": reason}
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_llm_veto_func_match_accepted():
    client = MagicMock()
    client.messages.create.return_value = _make_veto_func_match_response(
        True, "same op"
    )
    seq = _make_seq_info(7, 9, "    x = 1\n")
    func = _FunctionInfo(
        name="fn",
        source="def fn(): pass\n",
        scope="<module>",
        body_source="    pass\n",
        body_stmt_count=1,
        params=[],
    )
    is_valid, reason, _ = _llm_veto_func_match(client, seq, func, "source")
    assert is_valid is True
    assert reason == "same op"


def test_llm_veto_func_match_rejected():
    client = MagicMock()
    client.messages.create.return_value = _make_veto_func_match_response(
        False, "different"
    )
    seq = _make_seq_info(7, 9, "    x = 1\n")
    func = _FunctionInfo(
        name="fn",
        source="def fn(): pass\n",
        scope="<module>",
        body_source="    pass\n",
        body_stmt_count=1,
        params=[],
    )
    is_valid, reason, _ = _llm_veto_func_match(client, seq, func, "source")
    assert is_valid is False


def test_llm_veto_func_match_api_error():
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_anthropic.APIError = Exception
        client = MagicMock()
        client.messages.create.side_effect = Exception("api error")
        seq = _make_seq_info(7, 9, "    x = 1\n")
        func = _FunctionInfo(
            name="fn",
            source="def fn(): pass\n",
            scope="<module>",
            body_source="    pass\n",
            body_stmt_count=1,
            params=[],
        )
        with pytest.raises(CrispenAPIError):
            _llm_veto_func_match(client, seq, func, "source")


def test_llm_veto_func_match_skips_non_matching_blocks():
    """Non-matching content block is skipped; matching block still found."""
    client = MagicMock()
    non_matching = MagicMock()
    non_matching.type = "text"  # not tool_use → False branch of the if
    matching = MagicMock()
    matching.type = "tool_use"
    matching.name = "evaluate_duplicate"
    matching.input = {"is_valid_duplicate": True, "reason": "same"}
    response = MagicMock()
    response.content = [non_matching, matching]
    client.messages.create.return_value = response
    seq = _make_seq_info(7, 9, "    x = 1\n")
    func = _FunctionInfo(
        name="fn",
        source="def fn(): pass\n",
        scope="<module>",
        body_source="    pass\n",
        body_stmt_count=1,
        params=[],
    )
    is_valid, reason, _ = _llm_veto_func_match(client, seq, func, "source")
    assert is_valid is True


def test_generate_no_arg_call_indented():
    seq = _make_seq_info(7, 9, "    x = 1\n    y = 2\n")
    func = _FunctionInfo(
        name="setup",
        source="def setup(): pass\n",
        scope="<module>",
        body_source="    pass\n",
        body_stmt_count=1,
        params=[],
    )
    result = _generate_no_arg_call(seq, func)
    assert result == "    setup()\n"


def test_generate_no_arg_call_no_indent():
    seq = _make_seq_info(1, 2, "x = 1\ny = 2\n")
    func = _FunctionInfo(
        name="setup",
        source="def setup(): pass\n",
        scope="<module>",
        body_source="    pass\n",
        body_stmt_count=1,
        params=[],
    )
    result = _generate_no_arg_call(seq, func)
    assert result == "setup()\n"


def _make_call_gen_response(replacement: str) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "generate_call"
    block.input = {"replacement": replacement}
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_llm_generate_call_success():
    client = MagicMock()
    client.messages.create.return_value = _make_call_gen_response(
        "    _process(data)\n"
    )
    seq = _make_seq_info(7, 9, "    y = 1\n")
    func = _FunctionInfo(
        name="_process",
        source="def _process(val):\n    pass\n",
        scope="<module>",
        body_source="    pass\n",
        body_stmt_count=1,
        params=["val"],
    )
    result = _llm_generate_call(client, seq, func, "source")
    assert result == "    _process(data)\n"


def test_llm_generate_call_api_error():
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_anthropic.APIError = Exception
        client = MagicMock()
        client.messages.create.side_effect = Exception("api error")
        seq = _make_seq_info(7, 9, "    y = 1\n")
        func = _FunctionInfo(
            name="_process",
            source="def _process(val):\n    pass\n",
            scope="<module>",
            body_source="    pass\n",
            body_stmt_count=1,
            params=["val"],
        )
        with pytest.raises(CrispenAPIError):
            _llm_generate_call(client, seq, func, "source")


def test_llm_generate_call_skips_non_matching_blocks():
    """Non-matching content block is skipped; matching block still found."""
    client = MagicMock()
    non_matching = MagicMock()
    non_matching.type = "text"  # not tool_use → False branch of the if
    matching = MagicMock()
    matching.type = "tool_use"
    matching.name = "generate_call"
    matching.input = {"replacement": "    _process(data)\n"}
    response = MagicMock()
    response.content = [non_matching, matching]
    client.messages.create.return_value = response
    seq = _make_seq_info(7, 9, "    y = 1\n")
    func = _FunctionInfo(
        name="_process",
        source="def _process(val):\n    pass\n",
        scope="<module>",
        body_source="    pass\n",
        body_stmt_count=1,
        params=["val"],
    )
    result = _llm_generate_call(client, seq, func, "source")
    assert result == "    _process(data)\n"
