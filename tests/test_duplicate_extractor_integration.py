from tests.dup_extractor_anthropic_result_fixture import (
    _setup_anthropic_extract_and_build_de,
)


def test_successful_extraction_has_two_blank_lines(monkeypatch):
    _result = _setup_anthropic_extract_and_build_de(monkeypatch)
    de = _result.de
    _mock_client = _result.mock_client
    _source = _result.source
    _helper_src = _result.helper
    # Exactly 2 blank lines before and after the inserted helper.
    assert "\n\n\ndef _helper" in de._new_source
    assert "\n\n\n\ndef _helper" not in de._new_source
    assert "def _helper(data):\n    pass\n\n\ndef foo" in de._new_source
