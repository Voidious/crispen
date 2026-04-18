from crispen.config import CrispenConfig
from crispen.engine import run_engine
from crispen.errors import CrispenAPIError
from crispen.file_limiter.runner import FileLimiterResult
from crispen.refactors.base import Refactor


def _run(changed):
    return list(run_engine(changed, config=CrispenConfig(min_tuple_size=3)))


class _RaisingTransformer(Refactor):
    """A Refactor subclass that always raises during tree traversal."""

    @classmethod
    def name(cls):
        return "RaisingRefactor"

    def leave_Module(self, original_node, updated_node):
        raise RuntimeError("intentional transform error")


class _CrispenApiErrorRefactor(Refactor):
    @classmethod
    def name(cls):
        return "ApiErrorRefactor"

    def leave_Module(self, original_node, updated_node):
        raise CrispenAPIError("test api error")


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


def _make_fl_result_with_entities(source="# reduced\n"):
    """Build a FileLimiterResult that moved MyClass → utils.py."""
    return FileLimiterResult(
        original_source=source,
        new_files={"utils.py": "class MyClass: pass\n"},
        messages=["big.py: FileLimiter: moved MyClass → utils.py"],
        abort=False,
        entity_to_target={"MyClass": "utils.py"},
    )
