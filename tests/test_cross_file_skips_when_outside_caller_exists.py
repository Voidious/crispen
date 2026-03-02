from crispen.config import CrispenConfig
from crispen.engine import run_engine
from ._make_pkg import _make_pkg


def test_cross_file_skips_when_outside_caller_exists(tmp_path):
    pkg = _make_pkg(tmp_path, "mypkg")

    service = pkg / "service.py"
    service.write_text(
        "def get_user():\n    return (name, age, score)\n", encoding="utf-8"
    )

    # This file is NOT in the diff but calls get_user.
    outside = pkg / "outside.py"
    outside.write_text(
        "from mypkg.service import get_user\na, b, c = get_user()\n",
        encoding="utf-8",
    )

    changed = {str(service): [(1, 2)]}
    msgs = list(
        run_engine(
            changed,
            _repo_root=str(tmp_path),
            config=CrispenConfig(min_tuple_size=3),
        )
    )

    assert any("callers exist outside the diff" in m for m in msgs)
    assert "return (name, age, score)" in service.read_text(encoding="utf-8")
