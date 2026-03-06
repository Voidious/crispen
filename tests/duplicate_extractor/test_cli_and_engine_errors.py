import textwrap
import pytest
from crispen.errors import CrispenAPIError
from .test_high_level_duplicate_extractor_flow import _DUP_RANGES, _DUP_SOURCE


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
