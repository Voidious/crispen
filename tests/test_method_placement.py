import textwrap
from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import DuplicateExtractor
from tests.test_response_helpers import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)


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
