from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import DuplicateExtractor
from ..test_verification_and_extraction import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)
from ...test_duplicate_extractor import _DUP_RANGES, _DUP_SOURCE
from ..test_llm_helpers_and_matching import _make_seq_info
import textwrap


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
