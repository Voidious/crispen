import textwrap
from crispen.config import CrispenConfig
from crispen.engine import run_engine


def test_update_diff_file_callers_false_blocks_private_with_outside_caller(tmp_path):
    """Private function with a caller outside diff ranges is NOT transformed."""
    source = textwrap.dedent(
        """\
        def _make_result():
            return (a, b, c)

        def use_in_diff():
            x, y, z = _make_result()

        def use_outside_diff():
            p, q, r = _make_result()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    config = CrispenConfig(min_tuple_size=3, update_diff_file_callers=False)
    # Diff only covers the function definition and use_in_diff
    msgs = list(run_engine({str(f): [(1, 5)]}, config=config))
    # Should NOT have been transformed (outside callers exist)
    assert not any("TupleDataclass" in m for m in msgs)
    assert "return (a, b, c)" in f.read_text(encoding="utf-8")
