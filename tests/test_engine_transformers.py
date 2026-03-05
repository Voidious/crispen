from crispen.errors import CrispenAPIError
from crispen.refactors.base import Refactor


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
