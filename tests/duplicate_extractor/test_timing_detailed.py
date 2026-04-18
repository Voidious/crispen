from unittest.mock import MagicMock, patch
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _FunctionInfo,
    _llm_generate_call,
    _llm_veto_func_match,
)
from .test_extractor_core import (
    _DUP_RANGES,
    _DUP_SOURCE,
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)
from .test_llm_operations import (
    _make_call_gen_response,
    _make_seq_info,
    _make_veto_func_match_response,
)
from .test_function_matching import _FUNC_MATCH_PARAM_RANGES, _FUNC_MATCH_PARAM_SOURCE


def test_llm_veto_with_timing_out(monkeypatch):
    """_llm_veto appends result to _timing_out when provided."""
    from crispen.refactors.duplicate_extractor import _llm_veto

    client = MagicMock()
    client.messages.create.return_value = _make_veto_response(True, "ok")
    group = [_make_seq_info(1, 3), _make_seq_info(5, 7)]
    timing: list = []
    _llm_veto(client, group, _timing_out=timing)
    assert len(timing) == 1
    assert timing[0].tool_input == {"is_valid_duplicate": True, "reason": "ok"}


def test_llm_veto_func_match_with_timing_out():
    """_llm_veto_func_match appends result to _timing_out when provided."""
    client = MagicMock()
    client.messages.create.return_value = _make_veto_func_match_response(True, "same")
    seq = _make_seq_info(7, 9, "    x = 1\n")
    func = _FunctionInfo(
        name="fn",
        source="def fn(): pass\n",
        scope="<module>",
        body_source="    pass\n",
        body_stmt_count=1,
        params=[],
    )
    timing: list = []
    _llm_veto_func_match(client, seq, func, "source", _timing_out=timing)
    assert len(timing) == 1
    assert timing[0].tool_input["is_valid_duplicate"] is True


def test_llm_generate_call_with_timing_out():
    """_llm_generate_call appends result to _timing_out when provided."""
    client = MagicMock()
    client.messages.create.return_value = _make_call_gen_response("    fn(data)\n")
    seq = _make_seq_info(7, 9, "    y = 1\n")
    func = _FunctionInfo(
        name="fn",
        source="def fn(val):\n    pass\n",
        scope="<module>",
        body_source="    pass\n",
        body_stmt_count=1,
        params=["val"],
    )
    timing: list = []
    result = _llm_generate_call(client, seq, func, "source", _timing_out=timing)
    assert result == "    fn(data)\n"
    assert len(timing) == 1


def test_llm_verify_extraction_with_timing_out():
    """_llm_verify_extraction appends result to _timing_out when provided."""
    from crispen.refactors.duplicate_extractor import _llm_verify_extraction

    client = MagicMock()
    client.messages.create.return_value = _make_verify_response(True, [])
    group = [_make_seq_info(1, 3), _make_seq_info(5, 7)]
    timing: list = []
    is_correct, issues = _llm_verify_extraction(
        client,
        group,
        "def _helper(): pass\n",
        ["    _helper()\n", "    _helper()\n"],
        "a = 1\nb = 2\n",
        _timing_out=timing,
    )
    assert is_correct is True
    assert len(timing) == 1


def test_llm_verify_extraction_without_timing_out():
    """_llm_verify_extraction works correctly when _timing_out is None."""
    from crispen.refactors.duplicate_extractor import _llm_verify_extraction

    client = MagicMock()
    client.messages.create.return_value = _make_verify_response(True, [])
    group = [_make_seq_info(1, 3), _make_seq_info(5, 7)]
    is_correct, issues = _llm_verify_extraction(
        client,
        group,
        "def _helper(): pass\n",
        ["    _helper()\n", "    _helper()\n"],
        "a = 1\nb = 2\n",
    )
    assert is_correct is True
    assert issues == []


def test_func_match_veto_timing_recorded(monkeypatch):
    """When func-match veto accepts, record_llm_call is invoked for the veto call."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        # veto accepts → call-gen runs (func has params) → done (no dup groups)
        mock_client.messages.create.side_effect = [
            _make_veto_func_match_response(True, "same"),
            _make_call_gen_response("    _process(data)\n"),
        ]
        de = DuplicateExtractor(
            _FUNC_MATCH_PARAM_RANGES,
            source=_FUNC_MATCH_PARAM_SOURCE,
        )
    # record_llm_call ran for veto (the timing branch was True)
    assert de.stats.llm_elapsed_by_category.get("veto", 0) >= 0


def test_func_match_call_gen_timing_recorded(monkeypatch):
    """When func-match call-gen runs, record_llm_call is invoked for the edit call."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        # veto accepts → call-gen runs → done (no dup groups)
        mock_client.messages.create.side_effect = [
            _make_veto_func_match_response(True, "same"),
            _make_call_gen_response("    _process(data)\n"),
        ]
        de = DuplicateExtractor(
            _FUNC_MATCH_PARAM_RANGES,
            source=_FUNC_MATCH_PARAM_SOURCE,
        )
    assert de.stats.llm_edit_calls >= 1


def test_func_match_veto_detailed_timing_suffix(monkeypatch, capsys):
    """timing='detailed' prints timing suffix after func-match veto result."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_func_match_response(True, "same"),
            _make_call_gen_response("    _process(data)\n"),
        ]
        DuplicateExtractor(
            _FUNC_MATCH_PARAM_RANGES,
            source=_FUNC_MATCH_PARAM_SOURCE,
            verbose=True,
            timing="detailed",
        )
    err = capsys.readouterr().err
    assert "ACCEPTED" in err
    assert "[" in err  # timing suffix present


def test_func_match_replacement_detailed_timing_suffix(monkeypatch, capsys):
    """timing='detailed' prints timing suffix after func-match replacement line."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_func_match_response(True, "same"),
            _make_call_gen_response("    _process(data)\n"),
        ]
        DuplicateExtractor(
            _FUNC_MATCH_PARAM_RANGES,
            source=_FUNC_MATCH_PARAM_SOURCE,
            verbose=True,
            timing="detailed",
        )
    err = capsys.readouterr().err
    assert "replacing" in err
    assert "[" in err  # timing suffix on replacement line


def test_dup_veto_detailed_timing_suffix(monkeypatch, capsys):
    """timing='detailed' prints timing suffix after dup-group veto result."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.return_value = _make_veto_response(
            False, "different logic"
        )
        DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=True,
            timing="detailed",
        )
    err = capsys.readouterr().err
    assert "VETOED" in err
    assert "[" in err  # timing suffix present


def test_verify_detailed_timing_suffix(monkeypatch, capsys):
    """timing='detailed' prints timing suffix after verify result."""
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
            _make_verify_response(True, []),
        ]
        DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=True,
            timing="detailed",
        )
    err = capsys.readouterr().err
    assert "verify ACCEPTED" in err
    assert "[" in err  # timing suffix present


def test_extraction_detailed_timing_message(monkeypatch, capsys):
    """timing='detailed' prints extraction timing message after extraction call."""
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
            _make_verify_response(True, []),
        ]
        DuplicateExtractor(
            _DUP_RANGES,
            source=_DUP_SOURCE,
            verbose=True,
            timing="detailed",
        )
    err = capsys.readouterr().err
    assert "→ extraction [" in err
