from crispen.errors import CrispenAPIError
from crispen.refactors.base import Refactor


class _CrispenApiErrorRefactor(Refactor):
    @classmethod
    def name(cls):
        return "ApiErrorRefactor"

    def leave_Module(self, original_node, updated_node):
        raise CrispenAPIError("test api error")
