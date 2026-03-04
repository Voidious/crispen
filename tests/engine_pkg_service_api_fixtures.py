from tests.test_engine_cross_file_fixtures import _make_pkg


def _setup_pkg_service_api_get_user_main(tmp_path, pkg_name):
    pkg = _make_pkg(tmp_path, pkg_name)

    service = pkg / "service.py"
    service.write_text(
        "def get_user():\n    return (name, age, score)\n", encoding="utf-8"
    )

    api = pkg / "api.py"
    api.write_text(
        f"from {pkg_name}.service import get_user\n"
        "def main():\n"
        "    a, b, c = get_user()\n",
        encoding="utf-8",
    )

    return pkg, service, api
