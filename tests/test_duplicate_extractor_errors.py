from unittest.mock import MagicMock, patch
import pytest
from crispen.errors import CrispenAPIError
from crispen.refactors.duplicate_extractor import DuplicateExtractor
from tests.test_duplicate_extractor_core_extractor import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)
from tests.test_duplicate_extractor_end_to_end import (
    _DUP_RANGES,
    _DUP_SOURCE,
    _ESC_RANGES,
    _ESC_SOURCE,
)


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


def _make_extractor_with_veto_and_extract_side_effects(mock_anthropic, verbose):
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
        return _make_extractor_with_veto_and_extract_side_effects(
            mock_anthropic, verbose
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
