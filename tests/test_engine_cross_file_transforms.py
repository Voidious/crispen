from crispen.config import CrispenConfig
from crispen.engine import run_engine
from tests.engine_pkg_service_api_fixtures import _setup_pkg_service_api_get_user_main
from tests.engine_run_helpers import _run_engine_expect_outside_callers_msg
from tests.test_engine_cross_file_fixtures import _make_pkg


def test_cross_file_transforms_public_func_and_caller(tmp_path):
    pkg, service, api = _setup_pkg_service_api_get_user_main(tmp_path, "mypkg")

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

    msgs = _run_engine_expect_outside_callers_msg(service, tmp_path)
    assert "return (name, age, score)" in service.read_text(encoding="utf-8")
