from crispen.config import CrispenConfig
from crispen.engine import run_engine
from .test_update_diff_file_callers_false import _make_pkg


def test_cross_file_transforms_public_func_and_caller(tmp_path):
    pkg = _make_pkg(tmp_path, "mypkg")

    service = pkg / "service.py"
    service.write_text(
        "def get_user():\n    return (name, age, score)\n", encoding="utf-8"
    )

    api = pkg / "api.py"
    api.write_text(
        "from mypkg.service import get_user\n"
        "def main():\n"
        "    a, b, c = get_user()\n",
        encoding="utf-8",
    )

    changed = {str(service): [(1, 2)], str(api): [(1, 4)]}
    msgs = list(
        run_engine(
            changed,
            _repo_root=str(tmp_path),
            config=CrispenConfig(min_tuple_size=3),
        )
    )

    assert any("TupleDataclass" in m for m in msgs)
    assert any("CallerUpdater" in m for m in msgs)

    service_text = service.read_text(encoding="utf-8")
    assert "GetUserResult(" in service_text
    assert "@dataclass" in service_text

    api_text = api.read_text(encoding="utf-8")
    assert "_ = get_user()" in api_text
    assert "_.name" in api_text


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


def test_cross_file_one_approved_one_blocked(tmp_path):
    pkg = _make_pkg(tmp_path, "mypkg")

    a = pkg / "a.py"
    a.write_text("def approved_func():\n    return (1, 2, 3)\n", encoding="utf-8")

    b = pkg / "b.py"
    b.write_text("def blocked_func():\n    return (1, 2, 3)\n", encoding="utf-8")

    # outside.py calls blocked_func and is NOT in the diff.
    outside = pkg / "outside.py"
    outside.write_text(
        "from mypkg.b import blocked_func\nblocked_func()\n", encoding="utf-8"
    )

    changed = {str(a): [(1, 2)], str(b): [(1, 2)]}
    msgs = list(
        run_engine(
            changed, _repo_root=str(tmp_path), config=CrispenConfig(min_tuple_size=3)
        )
    )

    # blocked_func is skipped; its identity entry in alias_map hits the 349->348 branch.
    assert any(
        "blocked_func" in m and "callers exist outside the diff" in m for m in msgs
    )
    # approved_func is transformed.
    assert any("TupleDataclass" in m for m in msgs)


def test_cross_file_init_alias_detected_as_outside_caller(tmp_path):
    pkg = _make_pkg(tmp_path, "mypkg")

    # Re-export get_user through __init__.py
    (pkg / "__init__.py").write_text(
        "from mypkg.service import get_user\n", encoding="utf-8"
    )

    service = pkg / "service.py"
    service.write_text(
        "def get_user():\n    return (name, age, score)\n", encoding="utf-8"
    )

    # Outside file imports via the alias (pkg.get_user)
    outside = tmp_path / "outside.py"
    outside.write_text(
        "from mypkg import get_user\na, b, c = get_user()\n", encoding="utf-8"
    )

    changed = {str(service): [(1, 2)]}
    msgs = list(
        run_engine(
            changed, _repo_root=str(tmp_path), config=CrispenConfig(min_tuple_size=3)
        )
    )

    assert any("callers exist outside the diff" in m for m in msgs)
