"""Update @patch string literals to reflect FileLimiter entity moves."""

from __future__ import annotations

from typing import Dict, List, Tuple

import libcst as cst


class _PatchStringUpdater(cst.CSTTransformer):
    """Replace string literals whose values match keys in *patch_map*."""

    def __init__(self, sorted_items: List[Tuple[str, str]]) -> None:
        # (old, new) sorted longest-first to avoid partial matches
        self._sorted = sorted_items

    def leave_SimpleString(
        self,
        original_node: cst.SimpleString,
        updated_node: cst.SimpleString,
    ) -> cst.SimpleString:
        raw = updated_node.value  # includes quotes, e.g. '"myapp.module.MyClass"'
        if not raw or raw[0] not in ('"', "'"):
            return updated_node  # skip prefixed strings (b"", r"", f"", etc.)
        if raw.startswith(('"""', "'''")):
            return updated_node  # skip triple-quoted strings
        quote = raw[0]
        inner = raw[1:-1]  # strip opening and closing quote
        for old, new in self._sorted:
            if inner == old:
                return updated_node.with_changes(value=quote + new + quote)
            if inner.startswith(old + "."):
                return updated_node.with_changes(
                    value=quote + new + inner[len(old) :] + quote
                )
        return updated_node


def apply_patch_strings(source: str, patch_map: Dict[str, str]) -> str:
    """Return *source* with string literals updated per *patch_map*.

    Each key in *patch_map* is an old dotted path (e.g.
    ``"myapp.module.MyClass"``); its value is the new path.  A string
    literal whose value equals *old* or starts with *old + "."* is
    updated: the *old* prefix is replaced with *new*, preserving any
    suffix (e.g. ``"myapp.module.MyClass.method"`` →
    ``"myapp.module.helpers.MyClass.method"``).

    Returns *source* unchanged when *patch_map* is empty or the file
    cannot be parsed as Python.
    """
    if not patch_map:
        return source
    # Sort longest-first so "a.b.C" matches before "a.b" when both present.
    sorted_items = sorted(patch_map.items(), key=lambda x: len(x[0]), reverse=True)
    try:
        tree = cst.parse_module(source)
    except cst.ParserSyntaxError:
        return source
    updater = _PatchStringUpdater(sorted_items)
    new_tree = tree.visit(updater)
    return new_tree.code
