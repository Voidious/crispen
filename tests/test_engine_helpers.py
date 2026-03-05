from crispen.config import CrispenConfig
from crispen.engine import run_engine
from crispen.errors import CrispenAPIError
from crispen.refactors.base import Refactor


def _run(changed):
    return list(run_engine(changed, config=CrispenConfig(min_tuple_size=3)))


def _write_code_file(tmp_path, source):
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
