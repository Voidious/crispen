from crispen.engine import run_engine


def test_no_public_candidates_with_repo_root(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    msgs = list(run_engine({str(f): [(1, 1)]}, _repo_root=str(tmp_path)))
    assert msgs == []
