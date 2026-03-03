from unittest.mock import patch
from crispen.refactors.duplicate_extractor import _ApiTimeout, DuplicateExtractor
from tests.test_block_data import _FUNC_MATCH_PARAM_RANGES, _FUNC_MATCH_PARAM_SOURCE


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
