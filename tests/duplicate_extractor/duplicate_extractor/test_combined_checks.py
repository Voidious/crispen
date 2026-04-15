from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _would_create_proxy_wrappers,
)
from .utils import (
    _DUP_RANGES,
    _DUP_SOURCE,
    _PROXY_RANGES,
    _PROXY_SOURCE,
    _TWO_PAIR_RANGES,
    _TWO_PAIR_SOURCE,
    _make_extract_response,
    _make_proxy_func,
    _make_proxy_seq,
    _make_verify_response,
    _make_veto_response,
)


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
            "crispen.refactors.duplicate_extractor.extractor._has_call_to",
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


def _make_undefined_in_combined_extractor(monkeypatch, verbose=True):
    """Simulate: per-group checks all pass but helper definition is absent from
    the combined output (insertion edit blocked by overlap detector).

    Achieved by patching _has_funcdef: returns True for the per-group pyflakes
    check (not called there directly, but we patch the combined check) and
    False for the final combined check.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic") as mock_anthropic,
        patch(
            "crispen.refactors.duplicate_extractor.extractor._has_funcdef",
            side_effect=[False],
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


def test_undefined_helper_in_combined_drops_group_verbose(monkeypatch, capsys):
    de = _make_undefined_in_combined_extractor(monkeypatch, verbose=True)
    assert de._new_source is None
    assert "not defined in combined output" in capsys.readouterr().err


def test_undefined_helper_in_combined_drops_group_verbose_false(monkeypatch):
    de = _make_undefined_in_combined_extractor(monkeypatch, verbose=False)
    assert de._new_source is None


def test_undefined_helper_in_combined_two_groups_one_dropped(monkeypatch):
    """Two groups: first group's helper missing from combined, second kept.

    _has_funcdef returns [False, True]: first group dropped, second kept.
    This exercises the all_edits.extend(g_edits) loop after the drop (line 2304).
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with (
        patch("crispen.llm_client.anthropic") as mock_anthropic,
        patch(
            "crispen.refactors.duplicate_extractor.extractor._has_funcdef",
            side_effect=[False, True],
        ),
    ):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
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
        de = DuplicateExtractor(
            _TWO_PAIR_RANGES, source=_TWO_PAIR_SOURCE, verbose=False
        )
    # First group dropped (undefined), second group kept → new source written
    assert de._new_source is not None


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
            "crispen.refactors.duplicate_extractor.extractor._has_call_to",
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


def test_would_create_proxy_wrappers_false_single_full_body():
    """Single-member group where the seq covers the entire function body.

    All members are proxies, so extraction is still worthwhile → False.
    """
    seq = _make_proxy_seq(3, scope="foo")
    func = _make_proxy_func("foo", body_stmt_count=3, scope="<module>")
    assert _would_create_proxy_wrappers([seq], [func]) is False


def test_would_create_proxy_wrappers_false_all_full_bodies():
    """All group members cover entire function bodies → False.

    When every member becomes a proxy the group is all-or-nothing: extracting
    a shared helper is still worthwhile, so the guard should not block it.
    """
    seq1 = _make_proxy_seq(3, scope="process", class_scope="ClassA")
    seq2 = _make_proxy_seq(3, scope="process", class_scope="ClassB")
    func1 = _make_proxy_func("process", body_stmt_count=3, scope="ClassA")
    func2 = _make_proxy_func("process", body_stmt_count=3, scope="ClassB")
    assert _would_create_proxy_wrappers([seq1, seq2], [func1, func2]) is False


def test_would_create_proxy_wrappers_false_partial_body():
    """A seq that covers only part of a function body → False."""
    seq = _make_proxy_seq(2, scope="foo")
    func = _make_proxy_func("foo", body_stmt_count=4, scope="<module>")
    assert _would_create_proxy_wrappers([seq], [func]) is False


def test_would_create_proxy_wrappers_false_module_scope():
    """A seq at module scope (not inside a function) is never a proxy → False."""
    seq = _make_proxy_seq(3, scope="<module>")
    func = _make_proxy_func("foo", body_stmt_count=3, scope="<module>")
    assert _would_create_proxy_wrappers([seq], [func]) is False


def test_would_create_proxy_wrappers_false_no_matching_func():
    """No function with matching name → False."""
    seq = _make_proxy_seq(3, scope="foo")
    func = _make_proxy_func("bar", body_stmt_count=3, scope="<module>")
    assert _would_create_proxy_wrappers([seq], [func]) is False


def test_would_create_proxy_wrappers_false_scope_mismatch():
    """Seq in class method but func is module-level with same name → False."""
    seq = _make_proxy_seq(3, scope="foo", class_scope="MyClass")
    func = _make_proxy_func("foo", body_stmt_count=3, scope="<module>")
    assert _would_create_proxy_wrappers([seq], [func]) is False


def test_would_create_proxy_wrappers_group_with_one_proxy():
    """A group with multiple seqs, one of which covers an entire body → True."""
    seq_partial = _make_proxy_seq(2, scope="foo")
    seq_full = _make_proxy_seq(3, scope="bar")
    func_foo = _make_proxy_func("foo", body_stmt_count=5, scope="<module>")
    func_bar = _make_proxy_func("bar", body_stmt_count=3, scope="<module>")
    assert (
        _would_create_proxy_wrappers([seq_partial, seq_full], [func_foo, func_bar])
        is True
    )


def test_proxy_wrapper_guard_skips_group_verbose(monkeypatch, capsys):
    """Groups that would leave a function as a trivial proxy are skipped (verbose)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic.Anthropic"):
        de = DuplicateExtractor(_PROXY_RANGES, source=_PROXY_SOURCE, verbose=True)

    assert de._new_source is None
    captured = capsys.readouterr()
    assert "trivial proxy wrapper" in captured.err


def test_proxy_wrapper_guard_skips_group_silent(monkeypatch):
    """Groups that would leave a trivial proxy are skipped with verbose=False."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic.Anthropic"):
        de = DuplicateExtractor(_PROXY_RANGES, source=_PROXY_SOURCE, verbose=False)

    assert de._new_source is None
