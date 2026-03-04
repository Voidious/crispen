from tests.dup_extractor_anthropic_result_fixture import (
    _setup_anthropic_extract_and_build_de,
)


def test_successful_extraction_module_level(monkeypatch, tmp_path):
    _result = _setup_anthropic_extract_and_build_de(monkeypatch)
    de = _result.de
    _mock_client = _result.mock_client
    _source = _result.source
    _helper_src = _result.helper
    assert "_helper" in de._new_source
    assert len(de.changes_made) == 1
    assert "'_helper'" in de.changes_made[0]
    assert de.get_rewritten_source() == de._new_source
