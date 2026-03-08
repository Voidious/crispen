from unittest.mock import MagicMock, patch
from libcst.metadata import MetadataWrapper
from crispen.errors import CrispenAPIError
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _ApiTimeout,
    _SequenceCollector,
    _llm_veto_func_match,
)
from .integration import (
    _COLLISION_RANGES,
    _COLLISION_SOURCE,
    _DUP_RANGES,
    _DUP_SOURCE,
    _ESC_RANGES,
    _ESC_SOURCE,
    _FUNC_MATCH_PARAM_RANGES,
    _FUNC_MATCH_PARAM_SOURCE,
    _FUNC_MATCH_RANGES,
    _FUNC_MATCH_SOURCE,
    _FUNC_MATCH_THEN_DUP_RANGES,
    _FUNC_MATCH_THEN_DUP_SOURCE,
    _PARAM_DUP_RANGES,
    _PARAM_DUP_SOURCE,
    _POST_STEAL_RANGES,
    _POST_STEAL_SOURCE,
    _RETURN_BLOCK_RANGES,
    _RETURN_BLOCK_SOURCE,
    _make_extract_response,
    _make_import_local_extract_response,
    _make_invalid_assembled_extractor,
    _make_missing_free_vars_extractor,
    _make_new_attr_extractor,
    _make_no_call_extractor,
    _make_pyflakes_check_extractor,
    _make_return_block_extract_response,
    _make_two_group_drop_extractor,
    _make_uncalled_in_combined_extractor,
    _make_verify_response,
    _make_veto_response,
)
import textwrap
import libcst as cst
import pytest


def test_successful_extraction_has_two_blank_lines(monkeypatch):
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
        de = DuplicateExtractor([(9, 11)], source=source)

    assert de._new_source is not None
    # Exactly 2 blank lines before and after the inserted helper.
    assert "\n\n\ndef _helper" in de._new_source
    assert "\n\n\n\ndef _helper" not in de._new_source
    assert "def _helper(data):\n    pass\n\n\ndef foo" in de._new_source


def test_helper_placed_before_class_not_inside(monkeypatch):
    """Helper extracted from class methods must be placed BEFORE the class.

    When duplicate blocks live inside class methods, inserting a module-level
    helper before the method (inside the class body) ends the class definition
    prematurely and turns the remaining methods into nested functions.  The fix
    in _find_insertion_point walks backwards to the enclosing class.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        import os

        class MyClass:
            def method_a(self, x):
                a = compute(x)
                b = transform(a)
                c = finalize(b)
                return c

            def method_b(self, x):
                a = compute(x)
                b = transform(a)
                c = finalize(b)
                return c
        """
    )
    helper = "def _do_work(x):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_do_work",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        return _do_work(x)\n",
                        "        return _do_work(x)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor([(1, 100)], source=source)

    assert de._new_source is not None
    compile(de._new_source, "<test>", "exec")
    # Helper must appear BEFORE the class definition, not inside it.
    helper_pos = de._new_source.find("def _do_work")
    class_pos = de._new_source.find("class MyClass")
    assert (
        helper_pos < class_pos
    ), "helper was placed after/inside class instead of before it"
    # The class structure must be intact: MyClass still has both methods.
    import ast as _ast

    tree = _ast.parse(de._new_source)
    classes = [n for n in _ast.walk(tree) if isinstance(n, _ast.ClassDef)]
    assert len(classes) == 1
    assert classes[0].name == "MyClass"
    methods = [n.name for n in classes[0].body if isinstance(n, _ast.FunctionDef)]
    assert "method_a" in methods
    assert "method_b" in methods


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


def test_new_attribute_check_skips_group_verbose(monkeypatch, capsys):
    de = _make_new_attr_extractor(monkeypatch, verbose=True)
    assert de._new_source is None
    assert "new attribute access" in capsys.readouterr().err


def test_new_attribute_check_skips_group_verbose_false(monkeypatch):
    de = _make_new_attr_extractor(monkeypatch, verbose=False)
    assert de._new_source is None


def test_no_call_check_skips_group_verbose(monkeypatch, capsys):
    de = _make_no_call_extractor(monkeypatch, verbose=True)
    assert de._new_source is None
    assert "not called in candidate output" in capsys.readouterr().err


def test_no_call_check_skips_group_verbose_false(monkeypatch):
    de = _make_no_call_extractor(monkeypatch, verbose=False)
    assert de._new_source is None


def test_uncalled_in_combined_drops_group_verbose(monkeypatch, capsys):
    de = _make_uncalled_in_combined_extractor(monkeypatch, verbose=True)
    assert de._new_source is None
    assert "DROPPED" in capsys.readouterr().err


def test_uncalled_in_combined_drops_group_verbose_false(monkeypatch):
    de = _make_uncalled_in_combined_extractor(monkeypatch, verbose=False)
    assert de._new_source is None


def test_two_groups_one_dropped_combined_check(monkeypatch, capsys):
    """One of two groups is dropped by the combined call check; the other is kept."""
    de = _make_two_group_drop_extractor(monkeypatch, verbose=True)
    assert de._new_source is not None
    assert "DROPPED" in capsys.readouterr().err


def test_successful_extraction_module_level(monkeypatch, tmp_path):
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

        de = DuplicateExtractor([(9, 11)], source=source)

    assert de._new_source is not None
    assert "_helper" in de._new_source
    assert len(de.changes_made) == 1
    assert "'_helper'" in de.changes_made[0]
    assert de.get_rewritten_source() == de._new_source


def test_staticmethod_placement(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class MyClass:
            def foo(self):
                x = compute(data)
                y = transform(x)
                z = finalize(y)

            def bar(self):
                x = compute(data)
                y = transform(x)
                z = finalize(y)
        """
    )
    helper = "    @staticmethod\n    def _helper(data):\n        pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:MyClass",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]

        de = DuplicateExtractor([(8, 10)], source=source)

    assert de._new_source is not None


def test_cross_class_duplicates_use_module_level_placement(monkeypatch):
    """Duplicates in different classes must be extracted as module-level functions."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class ClassA:
            def foo(self):
                x = compute(data)
                y = transform(x)
                z = finalize(y)

        class ClassB:
            def bar(self):
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
        responses = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        _helper(data)\n",
                        "        _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        mock_client.messages.create.side_effect = responses
        de = DuplicateExtractor([(3, 5)], source=source)

    assert de._new_source is not None
    # The extraction call prompt should tell the LLM to use module_level
    extract_prompt = mock_client.messages.create.call_args_list[1][1]["messages"][0][
        "content"
    ]
    assert "module_level" in extract_prompt
    assert "staticmethod" not in extract_prompt


def test_cross_class_staticmethod_placement_rejected(monkeypatch):
    """LLM returning staticmethod placement for a cross-class group is rejected."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class ClassA:
            def foo(self):
                x = compute(data)
                y = transform(x)
                z = finalize(y)

        class ClassB:
            def bar(self):
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
        # First extraction attempt: LLM ignores prompt and returns staticmethod
        # placement for a cross-class group → rejected; second attempt: correct.
        responses = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:ClassA",
                    "helper_source": (
                        "    @staticmethod\n    def _helper(data):\n        pass\n"
                    ),
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        _helper(data)\n",
                        "        _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        mock_client.messages.create.side_effect = responses
        de = DuplicateExtractor([(3, 5)], source=source)

    assert de._new_source is not None
    # Three LLM calls: veto + two extraction attempts
    assert mock_client.messages.create.call_count == 4


def test_cross_class_staticmethod_placement_rejected_non_verbose(monkeypatch):
    """Defensive cross-class check works when verbose=False (no print side-effect)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class ClassA:
            def foo(self):
                x = compute(data)
                y = transform(x)
                z = finalize(y)

        class ClassB:
            def bar(self):
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
        responses = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:ClassA",
                    "helper_source": (
                        "    @staticmethod\n    def _helper(data):\n        pass\n"
                    ),
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        _helper(data)\n",
                        "        _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        mock_client.messages.create.side_effect = responses
        de = DuplicateExtractor([(3, 5)], source=source, verbose=False)

    assert de._new_source is not None


def test_sequence_collector_class_scope():
    """_SequenceCollector sets class_scope for sequences inside class methods."""

    source = textwrap.dedent(
        """\
        x = 1
        y = 2
        z = 3

        class MyClass:
            def method(self):
                a = 1
                b = 2
                c = 3
        """
    )
    lines = source.splitlines(keepends=True)
    tree = cst.parse_module(source)
    collector = _SequenceCollector(lines, max_seq_len=8)
    MetadataWrapper(tree).visit(collector)

    module_seqs = [s for s in collector.sequences if s.class_scope is None]
    class_seqs = [s for s in collector.sequences if s.class_scope == "MyClass"]
    assert module_seqs, "expected module-level sequences with class_scope=None"
    assert class_seqs, "expected class-method sequences with class_scope='MyClass'"


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


def test_duplicate_extractor_helper_docstrings_false_strips_docstring(
    monkeypatch, capsys
):
    """When helper_docstrings=False, the LLM-returned docstring is stripped."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_shared",
                    "placement": "module_level",
                    "helper_source": (
                        "def _shared(data):\n"
                        '    """LLM added a docstring."""\n'
                        "    pass\n"
                    ),
                    "call_site_replacements": [
                        "    _shared(data)\n",
                        "    _shared(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor(
            _DUP_RANGES, source=_DUP_SOURCE, verbose=False, helper_docstrings=False
        )

    assert de._new_source is not None
    assert '"""LLM added a docstring."""' not in de._new_source


def test_duplicate_extractor_helper_docstrings_true_keeps_docstring(
    monkeypatch, capsys
):
    """When helper_docstrings=True, the LLM-returned docstring is preserved."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_shared",
                    "placement": "module_level",
                    "helper_source": (
                        "def _shared(data):\n"
                        '    """Keep this docstring."""\n'
                        "    pass\n"
                    ),
                    "call_site_replacements": [
                        "    _shared(data)\n",
                        "    _shared(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor(
            _DUP_RANGES, source=_DUP_SOURCE, verbose=False, helper_docstrings=True
        )

    assert de._new_source is not None
    assert '"""Keep this docstring."""' in de._new_source


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


def test_block_ends_with_return_guard_skips(monkeypatch, capsys):
    """Extraction rejected when block ends with return but replacement omits it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_return_block_extract_response(),
        ]
        de = DuplicateExtractor(
            _RETURN_BLOCK_RANGES,
            source=_RETURN_BLOCK_SOURCE,
            extraction_retries=0,
            llm_verify_retries=0,
        )
    assert de._new_source is None
    assert "block ends with return but replacement omits it" in capsys.readouterr().err


def test_block_ends_with_return_guard_skips_silent(monkeypatch):
    """verbose=False: extraction rejected with no stderr output."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_return_block_extract_response(),
        ]
        de = DuplicateExtractor(
            _RETURN_BLOCK_RANGES,
            source=_RETURN_BLOCK_SOURCE,
            verbose=False,
            extraction_retries=0,
            llm_verify_retries=0,
        )
    assert de._new_source is None


def test_helper_imports_local_guard_skips(monkeypatch, capsys):
    """Extraction rejected when helper imports a name that is a param in original."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_import_local_extract_response(),
        ]
        de = DuplicateExtractor(
            _PARAM_DUP_RANGES,
            source=_PARAM_DUP_SOURCE,
            extraction_retries=0,
            llm_verify_retries=0,
        )
    assert de._new_source is None
    assert "helper imports a name that is a parameter/local" in capsys.readouterr().err


def test_helper_imports_local_guard_skips_silent(monkeypatch):
    """verbose=False: extraction rejected with no stderr output."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_import_local_extract_response(),
        ]
        de = DuplicateExtractor(
            _PARAM_DUP_RANGES,
            source=_PARAM_DUP_SOURCE,
            verbose=False,
            extraction_retries=0,
            llm_verify_retries=0,
        )
    assert de._new_source is None


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
