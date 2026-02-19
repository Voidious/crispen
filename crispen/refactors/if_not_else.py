"""Refactor: if not X: A else B  →  if X: B else A"""

from typing import Union

import libcst as cst
from libcst.metadata import PositionProvider

from .base import Refactor


class IfNotElse(Refactor):
    """Flip `if not condition:` branches when an else clause is present.

    Transforms:
        if not condition:
            body_a()
        else:
            body_b()

    Into:
        if condition:
            body_b()
        else:
            body_a()

    Skipped if there is no else clause, or if the orelse is an elif chain.
    """

    def leave_If(
        self, original_node: cst.If, updated_node: cst.If
    ) -> Union[cst.If, cst.BaseStatement]:
        # Must be in a changed range
        if not self._in_changed_range(original_node):
            return updated_node

        # Test must be a UnaryOperation with Not operator
        test = updated_node.test
        if not isinstance(test, cst.UnaryOperation):
            return updated_node
        if not isinstance(test.operator, cst.Not):
            return updated_node

        # Must have an else clause
        orelse = updated_node.orelse
        if orelse is None:
            return updated_node

        # Skip elif chains (orelse is another If node)
        if isinstance(orelse, cst.If):
            return updated_node

        # orelse must be an Else node
        if not isinstance(orelse, cst.Else):
            return updated_node

        # Unwrap the Not: `not condition` → `condition`
        new_test = test.expression

        # Swap bodies
        original_body = updated_node.body
        original_else_body = orelse.body

        new_node = updated_node.with_changes(
            test=new_test,
            body=original_else_body,
            orelse=orelse.with_changes(body=original_body),
        )

        try:
            pos = self.get_metadata(PositionProvider, original_node)
            line = pos.start.line
        except KeyError:
            line = "?"
        self.changes_made.append(f"IfNotElse: flipped if/else at line {line}")
        return new_node
