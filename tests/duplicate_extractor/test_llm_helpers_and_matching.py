from unittest.mock import MagicMock, patch
from crispen.errors import CrispenAPIError
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _ApiTimeout,
    _FunctionInfo,
    _SeqInfo,
    _generate_no_arg_call,
    _llm_generate_call,
    _llm_veto_func_match,
)
from ..test_duplicate_extractor import (
    _FUNC_MATCH_PARAM_RANGES,
    _FUNC_MATCH_PARAM_SOURCE,
    _FUNC_MATCH_RANGES,
    _FUNC_MATCH_SOURCE,
    _FUNC_MATCH_THEN_DUP_RANGES,
    _FUNC_MATCH_THEN_DUP_SOURCE,
)
import pytest


def _make_call_gen_response(replacement: str) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "generate_call"
    block.input = {"replacement": replacement}
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_veto_func_match_response(is_valid: bool, reason: str = "test") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "evaluate_duplicate"
    block.input = {"is_valid_duplicate": is_valid, "reason": reason}
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_seq_info(start: int, end: int, src: str = "") -> _SeqInfo:
    return _SeqInfo(
        stmts=[],
        start_line=start,
        end_line=end,
        scope="foo",
        source=src,
        fingerprint="",
    )


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
