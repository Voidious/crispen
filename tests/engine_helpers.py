import textwrap
from crispen.config import CrispenConfig
from crispen.engine import run_engine
from crispen.errors import CrispenAPIError
from crispen.refactors.base import Refactor


def _run(changed):
    return list(run_engine(changed, config=CrispenConfig(min_tuple_size=3)))


def _prepare_code_file_with_if_else(tmp_path):
    source = textwrap.dedent(
        """\
        if not x:
            a()
        else:
            b()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    return f, source


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


def _setup_mypkg_files(tmp_path):
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

    return api, service


def _run_engine_with_changed(service, tmp_path):
    changed = {str(service): [(1, 2)]}
    msgs = list(
        run_engine(
            changed,
            _repo_root=str(tmp_path),
            config=CrispenConfig(min_tuple_size=3),
        )
    )

    assert any("callers exist outside the diff" in m for m in msgs)

    return msgs


def _make_phase1_pkg(root):
    """Helper: return a tmp_path containing a package for Phase 1 tests."""
    pkg = root / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    return pkg


def _prepare_code_file(tmp_path):
    source = textwrap.dedent(
        """\
        def _make_result():
            return (1, 2, 3)

        def use_it():
            a, b, c = _make_result()
        """
    )
    f = tmp_path / "code.py"
    f.write_text(source, encoding="utf-8")
    return f


_FL_PATCH = "crispen.engine.run_file_limiter"


def _prepare_duplicate_extractor_test_file(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")

    constructed_with: dict = {}

    original_init = __import__(
        "crispen.refactors.duplicate_extractor", fromlist=["DuplicateExtractor"]
    ).DuplicateExtractor.__init__
    return f, constructed_with, original_init
