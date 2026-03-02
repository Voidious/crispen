from crispen.refactors.base import Refactor


class _RaisingTransformer(Refactor):
    """A Refactor subclass that always raises during tree traversal."""

    @classmethod
    def name(cls):
        return "RaisingRefactor"

    def leave_Module(self, original_node, updated_node):
        raise RuntimeError("intentional transform error")
