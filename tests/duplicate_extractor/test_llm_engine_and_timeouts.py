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
from .test_editing_and_insertion import _make_seq_info


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


# Source with two structurally distinct duplicate pairs so _find_duplicate_groups
# returns two separate groups.  The groups differ in argument count so that
# _ASTNormalizer produces different fingerprints for each group:
#   group 1 (foo/bar): 3-stmt bodies using 2-argument calls → fingerprint A
#   group 2 (baz/qux): 3-stmt bodies using 3-argument calls → fingerprint B
_TWO_PAIR_SOURCE = textwrap.dedent(
    """\
    import os

    def foo():
        x = compute(data, config)
        y = transform(x, scale)
        z = finalize(y, mode)

    def bar():
        x = compute(data, config)
        y = transform(x, scale)
        z = finalize(y, mode)

    def baz():
        a = process(item, key, idx)
        b = convert(a, fmt, enc)
        c = export(b, path, opts)

    def qux():
        a = process(item, key, idx)
        b = convert(a, fmt, enc)
        c = export(b, path, opts)
    """
)
_TWO_PAIR_RANGES = [(4, 21)]  # overlaps all duplicate sequences


def _make_two_group_drop_extractor(monkeypatch, verbose=True):
    """Two extraction groups; the combined check drops one, exercising line 1533.

    _has_call_to is patched with side_effect=[True, True, True, False]:
    - calls 1-2: per-group checks for each group → both pass
    - call 3: combined check for first group → kept
    - call 4: combined check for second group → dropped
    After the drop, extraction_groups still has one entry, so the inner
    ``for _, g_edits, _ in extraction_groups`` loop runs once (line 1533).
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic") as mock_anthropic,
        patch(
            "crispen.refactors.duplicate_extractor._has_call_to",
            side_effect=[True, True, True, False],
        ),
    ):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        # Six LLM calls: veto+extract+verify for each of the two groups.
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper1",
                    "placement": "module_level",
                    "helper_source": "def _helper1():\n    pass\n",
                    "call_site_replacements": [
                        "    _helper1()\n",
                        "    _helper1()\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper2",
                    "placement": "module_level",
                    "helper_source": "def _helper2():\n    pass\n",
                    "call_site_replacements": [
                        "    _helper2()\n",
                        "    _helper2()\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        return DuplicateExtractor(
            _TWO_PAIR_RANGES, source=_TWO_PAIR_SOURCE, verbose=verbose
        )


def test_two_groups_one_dropped_combined_check(monkeypatch, capsys):
    """One of two groups is dropped by the combined call check; the other is kept."""
    de = _make_two_group_drop_extractor(monkeypatch, verbose=True)
    assert de._new_source is not None
    assert "DROPPED" in capsys.readouterr().err


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


def test_func_match_no_arg_replaces_body(monkeypatch):
    """No-param module-level function: algorithmic replacement, no call-gen LLM."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            return_value=(True, "same operation", ""),
        ),
    ):
        de = DuplicateExtractor(
            _FUNC_MATCH_RANGES, source=_FUNC_MATCH_SOURCE, verbose=True
        )
    assert de._new_source is not None
    assert "_setup" in de.changes_made[0]


def test_func_match_verbose_false(monkeypatch):
    """verbose=False covers all False branches of new if-self.verbose guards."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            return_value=(True, "same operation", ""),
        ),
    ):
        de = DuplicateExtractor(
            _FUNC_MATCH_RANGES, source=_FUNC_MATCH_SOURCE, verbose=False
        )
    assert de._new_source is not None


def test_func_match_veto_rejects(monkeypatch):
    """Veto rejects func match → no replacement."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            return_value=(False, "different", ""),
        ),
    ):
        de = DuplicateExtractor(
            _FUNC_MATCH_RANGES, source=_FUNC_MATCH_SOURCE, verbose=True
        )
    assert de._new_source is None


def test_func_match_veto_timeout(monkeypatch):
    """Veto times out → seq skipped; subsequent dup group also times out."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            side_effect=_ApiTimeout("timed out"),
        ),
    ):
        de = DuplicateExtractor(
            _FUNC_MATCH_RANGES, source=_FUNC_MATCH_SOURCE, verbose=True
        )
    assert de._new_source is None


def test_func_match_verify_fails(monkeypatch):
    """_verify_extraction returns False → func match skipped; dup group veto rejects."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Call 1: func match veto → (True, "ok")
    # Call 2: dup group veto → (False, "different") so extract is never called
    side_effects = [(True, "ok", ""), (False, "different", "")]

    def _mock_run(func, timeout, *args, **kwargs):
        return side_effects.pop(0)

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            side_effect=_mock_run,
        ),
        patch(
            "crispen.refactors.duplicate_extractor._verify_extraction",
            return_value=False,
        ),
    ):
        de = DuplicateExtractor(_FUNC_MATCH_RANGES, source=_FUNC_MATCH_SOURCE)
    assert de._new_source is None


def test_func_match_param_call_gen_success(monkeypatch):
    """Parametrised function: LLM generates call expression successfully."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Call 1: func match veto → (True, "ok")
    # Call 2: _llm_generate_call → replacement string
    side_effects: list = [(True, "ok", ""), "    _process(data)\n"]

    def _mock_run(func, timeout, *args, **kwargs):
        return side_effects.pop(0)

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            side_effect=_mock_run,
        ),
    ):
        de = DuplicateExtractor(
            _FUNC_MATCH_PARAM_RANGES,
            source=_FUNC_MATCH_PARAM_SOURCE,
            verbose=True,
        )
    assert de._new_source is not None


def test_func_match_param_call_gen_timeout(monkeypatch):
    """Call generation times out → seq skipped; dup group veto rejects."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Call 1: func match veto → (True, "ok")
    # Call 2: _llm_generate_call → timeout
    # Call 3: dup group veto → (False, "reject") so no extract called
    side_effects: list = [
        (True, "ok", ""),
        _ApiTimeout("timed out"),
        (False, "reject", ""),
    ]

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
        de = DuplicateExtractor(
            _FUNC_MATCH_PARAM_RANGES,
            source=_FUNC_MATCH_PARAM_SOURCE,
            verbose=True,
        )
    assert de._new_source is None


def test_func_match_then_dup_extract(monkeypatch):
    """Func match succeeds; remaining dup group triggers standard veto/extract."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    extraction_dict = {
        "function_name": "_helper",
        "placement": "module_level",
        "helper_source": "def _helper():\n    pass\n",
        "call_site_replacements": ["    _helper()\n", "    _helper()\n"],
    }
    # Call 1: func match veto → (True, "ok", "")
    # Call 2: dup group veto → (True, "ok", "")
    # Call 3: dup group extract → extraction dict
    # Call 4: LLM verify → (True, [])
    side_effects: list = [
        (True, "ok", ""),
        (True, "ok", ""),
        extraction_dict,
        (True, []),
    ]

    def _mock_run(func, timeout, *args, **kwargs):
        return side_effects.pop(0)

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            side_effect=_mock_run,
        ),
    ):
        de = DuplicateExtractor(
            _FUNC_MATCH_THEN_DUP_RANGES,
            source=_FUNC_MATCH_THEN_DUP_SOURCE,
        )
    assert de._new_source is not None
    # One func-match change + one dup-extract change
    assert len(de.changes_made) == 2


def test_match_functions_false_skips_func_match_pass(monkeypatch):
    """match_functions=False: func-match veto never called even when match exists."""

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    veto_func_match_called: list = []

    def _mock_run_with_timeout(fn, timeout, *args, **kwargs):
        if fn is _llm_veto_func_match:
            veto_func_match_called.append(True)
        # Reject any extraction-pass LLM call so no new source is produced.
        return (False, "rejected", "")

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            side_effect=_mock_run_with_timeout,
        ),
    ):
        de = DuplicateExtractor(
            _FUNC_MATCH_RANGES,
            source=_FUNC_MATCH_SOURCE,
            verbose=False,
            match_functions=False,
        )
    assert veto_func_match_called == []
    assert de._new_source is None


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


def test_extraction_name_collision_skipped(monkeypatch, capsys):
    # LLM returns function_name="_helper", which is already defined → skipped.
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
                    "helper_source": "def _helper(x, y):\n    pass\n",
                    "call_site_replacements": [
                        "    _helper(data, x)\n",
                        "    _helper(data, x)\n",
                    ],
                }
            ),
        ]
        de = DuplicateExtractor(
            _COLLISION_RANGES,
            source=_COLLISION_SOURCE,
            verbose=True,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None
    assert de.changes_made == []
    err = capsys.readouterr().err
    assert "name collision" in err
    assert "_helper" in err


def test_extraction_name_collision_silent(monkeypatch, capsys):
    # Same collision, verbose=False → no stderr output.
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
                    "helper_source": "def _helper(x, y):\n    pass\n",
                    "call_site_replacements": [
                        "    _helper(data, x)\n",
                        "    _helper(data, x)\n",
                    ],
                }
            ),
        ]
        de = DuplicateExtractor(
            _COLLISION_RANGES,
            source=_COLLISION_SOURCE,
            verbose=False,
            extraction_retries=0,
            llm_verify_retries=0,
        )

    assert de._new_source is None
    assert de.changes_made == []
    err = capsys.readouterr().err
    assert "name collision" not in err


def test_duplicate_extractor_custom_model_used(monkeypatch):
    """Custom model string is passed to the Anthropic API."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.return_value = _make_veto_response(False, "no")
        DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE, model="claude-opus-4-6")
    # Verify the custom model was passed
    call_kwargs = mock_client.messages.create.call_args_list[0][1]
    assert call_kwargs["model"] == "claude-opus-4-6"


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


def test_veto_notes_passed_to_extract(monkeypatch):
    """extraction_notes from veto are forwarded to the extract prompt."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response_with_notes(True, "same logic", "watch out for x"),
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
        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE)

    assert de._new_source is not None
    extract_call = mock_client.messages.create.call_args_list[1]
    extract_prompt = extract_call.kwargs["messages"][0]["content"]
    assert "watch out for x" in extract_prompt


def test_extraction_retry_on_alg_failure_verbose(monkeypatch, capsys):
    """First extract has wrong call count -> retry -> second succeeds. verbose=True."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
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
                    "call_site_replacements": ["    _helper(data)\n"],  # wrong count
                }
            ),
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
        de = DuplicateExtractor(
            _DUP_RANGES, source=_DUP_SOURCE, verbose=True, extraction_retries=1
        )

    assert de._new_source is not None
    err = capsys.readouterr().err
    assert "retrying" in err


def test_extraction_retry_on_alg_failure_silent(monkeypatch):
    """First extract has wrong call count -> retry -> second succeeds. verbose=False."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
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
                    "call_site_replacements": ["    _helper(data)\n"],  # wrong count
                }
            ),
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
        de = DuplicateExtractor(
            _DUP_RANGES, source=_DUP_SOURCE, verbose=False, extraction_retries=1
        )

    assert de._new_source is not None


def test_llm_verify_timeout_verbose(monkeypatch, capsys):
    """Verify times out (verbose=True) -> extraction is accepted and logged."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from crispen.refactors.duplicate_extractor import _llm_verify_extraction

    extraction_dict = {
        "function_name": "_helper",
        "placement": "module_level",
        "helper_source": "def _helper(data):\n    pass\n",
        "call_site_replacements": ["    _helper(data)\n", "    _helper(data)\n"],
    }
    side_effects: list = [(True, "same logic", ""), extraction_dict]

    def _mock_run(func, timeout, *args, **kwargs):
        if func is _llm_verify_extraction:
            raise _ApiTimeout("verify timed out")
        return side_effects.pop(0)

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            side_effect=_mock_run,
        ),
    ):
        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE, verbose=True)

    assert de._new_source is not None
    err = capsys.readouterr().err
    assert "verify timed out" in err


def test_llm_verify_rejects_then_retries_verbose(monkeypatch, capsys):
    """Verify rejects first attempt; retry extract passes. verbose=True."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
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
            _make_verify_response(False, ["wrong variable name"]),
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
        de = DuplicateExtractor(
            _DUP_RANGES, source=_DUP_SOURCE, verbose=True, llm_verify_retries=1
        )

    assert de._new_source is not None
    err = capsys.readouterr().err
    assert "REJECTED" in err
    assert "wrong variable name" in err
    assert "retrying" in err


def test_llm_verify_rejects_then_retries_silent(monkeypatch):
    """Verify rejects first attempt; retry extract passes. verbose=False."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
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
            _make_verify_response(False, ["wrong variable name"]),
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
        de = DuplicateExtractor(
            _DUP_RANGES, source=_DUP_SOURCE, verbose=False, llm_verify_retries=1
        )

    assert de._new_source is not None


def test_llm_verify_exhausted_skips_group(monkeypatch):
    """All verify attempts fail -> group skipped."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
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
            _make_verify_response(False, ["issue"]),
        ]
        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE, llm_verify_retries=0)

    assert de._new_source is None


def test_llm_verify_timeout_silent(monkeypatch):
    """Verify times out (verbose=False) -> extraction is accepted silently."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from crispen.refactors.duplicate_extractor import _llm_verify_extraction

    extraction_dict = {
        "function_name": "_helper",
        "placement": "module_level",
        "helper_source": "def _helper(data):\n    pass\n",
        "call_site_replacements": ["    _helper(data)\n", "    _helper(data)\n"],
    }
    side_effects: list = [(True, "same logic", ""), extraction_dict]

    def _mock_run(func, timeout, *args, **kwargs):
        if func is _llm_verify_extraction:
            raise _ApiTimeout("verify timed out")
        return side_effects.pop(0)

    with (
        patch("crispen.llm_client.anthropic.Anthropic"),
        patch(
            "crispen.refactors.duplicate_extractor._run_with_timeout",
            side_effect=_mock_run,
        ),
    ):
        de = DuplicateExtractor(_DUP_RANGES, source=_DUP_SOURCE, verbose=False)

    assert de._new_source is not None
