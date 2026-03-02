from unittest.mock import patch
from crispen.config import CrispenConfig
from crispen.engine import run_engine
from ._make_pkg import _make_pkg


def test_cross_file_caller_updater_raises(tmp_path):
    pkg = _make_pkg(tmp_path, "mypkg")

    service = pkg / "service.py"
    service.write_text("def approved():\n    return (1, 2, 3)\n", encoding="utf-8")

    changed = {str(service): [(1, 2)]}

    with patch("crispen.engine.CallerUpdater", side_effect=RuntimeError("fail")):
        # Should not crash; the exception is caught.
        list(
            run_engine(
                changed,
                _repo_root=str(tmp_path),
                config=CrispenConfig(min_tuple_size=3),
            )
        )
