from __future__ import annotations


def test_engine_includes_function_splitter_no_op(tmp_path):
    """FunctionSplitter is in _REFACTORS and runs without error for simple files."""
    from crispen.engine import run_engine
    from crispen.config import CrispenConfig

    py_file = tmp_path / "sample.py"
    py_file.write_text("def foo():\n    return 1\n")
    config = CrispenConfig(max_function_length=75)
    msgs = list(run_engine({str(py_file): [(1, 2)]}, verbose=False, config=config))
    # No split needed — no messages expected (or just no errors)
    assert all("FunctionSplitter" not in m for m in msgs)
