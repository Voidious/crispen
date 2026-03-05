import textwrap
from unittest.mock import MagicMock, patch
import pytest
from crispen.errors import CrispenAPIError
from crispen.refactors.duplicate_extractor import (
    _ApiTimeout,
    _FunctionInfo,
    _llm_veto_func_match,
    _run_with_timeout,
    DuplicateExtractor,
)
from tests.test_duplicate_extractor_core_extractor import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_func_match_response,
    _make_veto_response,
)
from tests.test_duplicate_extractor_end_to_end import _DUP_RANGES, _DUP_SOURCE
from tests.test_duplicate_extractor_sequences import _make_seq_info


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
