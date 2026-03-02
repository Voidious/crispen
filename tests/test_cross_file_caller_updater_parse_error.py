from unittest.mock import patch
import libcst as cst
from crispen.config import CrispenConfig
from crispen.engine import run_engine
from ._make_pkg import _make_pkg


def test_cross_file_caller_updater_parse_error(tmp_path):
    pkg = _make_pkg(tmp_path, "mypkg")

    service = pkg / "service.py"
    service.write_text("def approved():\n    return (1, 2, 3)\n", encoding="utf-8")

    changed = {str(service): [(1, 2)]}

    original_parse = cst.parse_module

    def patched_parse(source):
        # After Phase 2 transforms the source, it will contain "@dataclass".
        # Fail on that call to exercise the 374-375 parse-error branch.
        if "@dataclass" in source:
            raise cst.ParserSyntaxError(
                "fake error", lines=("@dataclass",), raw_line=0, raw_column=0
            )
        return original_parse(source)

    with patch("crispen.engine.cst.parse_module", patched_parse):
        # Should not crash; CallerUpdater pass silently continues.
        list(
            run_engine(
                changed,
                _repo_root=str(tmp_path),
                config=CrispenConfig(min_tuple_size=3),
            )
        )
