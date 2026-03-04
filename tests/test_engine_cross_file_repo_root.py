from crispen.config import CrispenConfig
from crispen.engine import run_engine


def test_cross_file_file_not_under_repo_root(tmp_path):
    # repo_root is a separate directory; changed file is not under it.
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    f = tmp_path / "code.py"
    f.write_text("def public_func():\n    return (1, 2, 3)\n", encoding="utf-8")
    # _compute_qname raises ValueError → all_candidates stays empty → 317->406 branch.
    msgs = list(
        run_engine(
            {str(f): [(1, 2)]},
            _repo_root=str(repo_root),
            config=CrispenConfig(min_tuple_size=3),
        )
    )
    assert not any("callers" in m for m in msgs)
