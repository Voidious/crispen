import pytest
from crispen.errors import CrispenAPIError
from tests.test_no_source_analysis import _DUP_RANGES, _DUP_SOURCE


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
