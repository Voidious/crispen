from crispen.config import CrispenConfig
from crispen.engine import run_engine


def _run(changed):
    return list(run_engine(changed, config=CrispenConfig(min_tuple_size=3)))


def _make_pkg(root, name):
    pkg = root / name
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    return pkg


def _make_phase1_pkg(root):
    """Helper: return a tmp_path containing a package for Phase 1 tests."""
    pkg = root / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    return pkg


_FL_PATCH = "crispen.engine.run_file_limiter"
