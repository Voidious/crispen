from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _replacement_contains_return,
)
from ..test_extraction_validation import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)
from ...test_duplicate_extractor import _DUP_RANGES, _DUP_SOURCE
from .utils import (
    _COLLISION_RANGES,
    _COLLISION_SOURCE,
    _PARAM_DUP_RANGES,
    _PARAM_DUP_SOURCE,
    _RETURN_BLOCK_RANGES,
    _RETURN_BLOCK_SOURCE,
    _make_import_local_extract_response,
    _make_invalid_assembled_extractor,
    _make_missing_free_vars_extractor,
    _make_pyflakes_check_extractor,
    _make_return_block_extract_response,
)
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


def test_replacement_contains_return_true():
    assert _replacement_contains_return("    return x\n") is True


def test_replacement_contains_return_false():
    assert _replacement_contains_return("    _helper()\n") is False


def test_replacement_contains_return_syntax_error():
    # Unclosed paren inside the wrapper → SyntaxError → False.
    assert _replacement_contains_return("    (\n") is False


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
