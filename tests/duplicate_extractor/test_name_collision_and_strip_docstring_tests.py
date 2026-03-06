from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import (
    _strip_helper_docstring,
    DuplicateExtractor,
)
from .name_collision_and_strip_docstring_tests import (
    _COLLISION_RANGES,
    _COLLISION_SOURCE,
)
from .orchestrator_fixtures import _make_extract_response, _make_veto_response


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


def test_strip_helper_docstring_with_docstring():
    source = 'def _helper(x):\n    """Strip me."""\n    return x\n'
    result = _strip_helper_docstring(source)
    assert '"""Strip me."""' not in result
    assert "return x" in result


def test_strip_helper_docstring_no_docstring():
    source = "def _helper(x):\n    return x\n"
    result = _strip_helper_docstring(source)
    assert result == source


def test_strip_helper_docstring_parse_error():
    bad = "def f(:\n    pass\n"
    result = _strip_helper_docstring(bad)
    assert result == bad


def test_strip_helper_docstring_non_function():
    source = "x = 1\n"
    result = _strip_helper_docstring(source)
    assert result == source


def test_strip_helper_docstring_docstring_only_body():
    # Function whose body is only a docstring — don't strip (would leave empty body).
    source = 'def _helper():\n    """Only doc."""\n'
    result = _strip_helper_docstring(source)
    assert '"""Only doc."""' in result
