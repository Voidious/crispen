from crispen.config import CrispenConfig
from crispen.engine import run_engine


def _make_pkg(root, name):
    pkg = root / name
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    return pkg


def _create_cross_file_test_pkg(
    tmp_path,
    pkg_name="mypkg",
    func_name="get_user",
    return_vals="name, age, score",
    caller_func="main",
):
    pkg = _make_pkg(tmp_path, pkg_name)

    service = pkg / "service.py"
    service.write_text(
        f"def {func_name}():\n    return ({return_vals})\n", encoding="utf-8"
    )

    api = pkg / "api.py"
    api.write_text(
        f"from {pkg_name}.service import {func_name}\n"
        f"def {caller_func}():\n"
        f"    a, b, c = {func_name}()\n",
        encoding="utf-8",
    )

    return pkg, service, api


def _run_cross_file_engine(service, tmp_path):
    changed = {str(service): [(1, 2)]}
    msgs = list(
        run_engine(
            changed,
            _repo_root=str(tmp_path),
            config=CrispenConfig(min_tuple_size=3),
        )
    )

    assert any("callers exist outside the diff" in m for m in msgs)
