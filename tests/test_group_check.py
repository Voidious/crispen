from tests.test_no_call import _make_no_call_extractor


def test_no_call_check_skips_group_verbose_false(monkeypatch):
    de = _make_no_call_extractor(monkeypatch, verbose=False)
    assert de._new_source is None
