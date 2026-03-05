import textwrap
from crispen.config import CrispenConfig
from crispen.engine import _categorize_into_stats, run_engine
from crispen.stats import RunStats


def _write_if_not_else_source(tmp_path):
    source = textwrap.dedent(
        """\
        if not x:
            a()
        else:
            b()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    return source, f


def test_categorize_if_not_else():
    s = RunStats()
    _categorize_into_stats(s, "IfNotElse: flipped if/else at line 3")
    assert s.if_not_else == 1
    assert s.total_edits == 1


def test_engine_disabled_refactors_skips_if_not_else(tmp_path):
    """With if_not_else disabled the pattern is left unchanged."""
    source, f = _write_if_not_else_source(tmp_path)
    msgs = list(
        run_engine(
            {str(f): [(1, 4)]},
            config=CrispenConfig(disabled_refactors=["if_not_else"]),
        )
    )
    assert not any("IfNotElse" in m for m in msgs)
    assert f.read_text(encoding="utf-8") == source
