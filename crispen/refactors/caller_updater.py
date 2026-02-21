"""Refactor: update call sites of tuple→dataclass transformed functions."""

from __future__ import annotations

import re
from typing import Dict, Optional, Union

import libcst as cst
from libcst.metadata import PositionProvider

from .base import Refactor
from .tuple_dataclass import TransformInfo


def _pascal_to_snake(name: str) -> str:
    """Convert PascalCase to snake_case (e.g. GetUserResult → get_user_result)."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()


def _tmp_name(dataclass_name: str, source: str) -> str:
    """Return a temporary variable name for the unpacking expansion.

    Tries candidates in priority order and returns the first whose identifier
    does not already appear as a standalone word in *source*:

    1. ``_``               — short, conventional throwaway
    2. ``_result``         — clear meaning, still brief
    3. snake_case of *dataclass_name*  (e.g. ``get_user_result``)
    4. ``_crispen_result`` — project-namespaced last resort
    """
    candidates = [
        "_",
        "_result",
        _pascal_to_snake(dataclass_name),
        "_crispen_result",
    ]
    for candidate in candidates:
        if not re.search(r"\b" + re.escape(candidate) + r"\b", source):
            return candidate
    return candidates[-1]  # pragma: no cover


def _module_to_str(node: Optional[Union[cst.Attribute, cst.Name]]) -> str:
    """Convert a libcst module node to a dotted string."""
    if node is None:
        return ""
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return f"{_module_to_str(node.value)}.{node.attr.value}"
    return ""  # pragma: no cover


def _resolve_relative_module(file_module: str, num_dots: int, suffix: str) -> str:
    """Resolve a relative import to an absolute module path.

    file_module: dotted module path of the importing file (e.g. "pkg.api")
    num_dots:    1 for ".", 2 for "..", etc.
    suffix:      "service" for "from .service import x", or "" for "from . import x"
    """
    parts = file_module.split(".")
    # "." means relative to the current package, so drop `num_dots` trailing parts.
    package_parts = parts[:-num_dots] if num_dots <= len(parts) else []
    if suffix:
        return ".".join(package_parts + [suffix])
    return ".".join(package_parts)


class CallerUpdater(Refactor):
    """Expand `a, b, c = func()` into per-field attribute assignments.

    Only fires when `func` was transformed from returning a tuple to returning a
    dataclass (recorded in `transforms`), and only for tuple-unpacking assignments
    in the changed ranges.
    """

    def __init__(
        self,
        changed_ranges,
        transforms: Dict[str, TransformInfo],
        file_module: str = "",
        source: str = "",
        verbose: bool = True,
        local_transforms: Optional[Dict[str, TransformInfo]] = None,
    ) -> None:
        super().__init__(changed_ranges, source=source, verbose=verbose)
        self.source = source
        # transforms: all qualified names (canonical + __init__ aliases) → TransformInfo
        self.transforms = transforms
        self.file_module = file_module
        # Maps local name (as used in this file) → TransformInfo
        self._local_transforms: Dict[str, TransformInfo] = {}
        if local_transforms:
            self._local_transforms.update(local_transforms)

    @classmethod
    def name(cls) -> str:
        return "CallerUpdater"

    # ------------------------------------------------------------------
    # Build the local-name → transform map from import statements
    # ------------------------------------------------------------------

    def visit_ImportFrom(self, node: cst.ImportFrom) -> Optional[bool]:
        if isinstance(node.names, cst.ImportStar):
            return None
        if not isinstance(node.names, (list, tuple)):
            return None  # pragma: no cover

        num_dots = len(node.relative)
        module_str = _module_to_str(node.module)

        if num_dots > 0:
            abs_module = _resolve_relative_module(
                self.file_module, num_dots, module_str
            )
        else:
            abs_module = module_str

        for alias in node.names:
            if not isinstance(alias, cst.ImportAlias):
                continue  # pragma: no cover
            if not isinstance(alias.name, cst.Name):
                continue  # pragma: no cover
            imported_name = alias.name.value

            if (
                alias.asname is not None
                and isinstance(alias.asname, cst.AsName)
                and isinstance(alias.asname.name, cst.Name)
            ):
                local_name = alias.asname.name.value
            else:
                local_name = imported_name

            full_qname = (
                f"{abs_module}.{imported_name}" if abs_module else imported_name
            )
            if full_qname in self.transforms:
                self._local_transforms[local_name] = self.transforms[full_qname]

        return None

    # ------------------------------------------------------------------
    # Replace tuple-unpacking assignments
    # ------------------------------------------------------------------

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> Union[cst.BaseStatement, cst.FlattenSentinel]:
        if not self._local_transforms:
            return updated_node
        if len(updated_node.body) != 1:
            return updated_node

        stmt = updated_node.body[0]
        if not isinstance(stmt, cst.Assign):
            return updated_node
        if len(stmt.targets) != 1:
            return updated_node

        target = stmt.targets[0].target
        if not isinstance(target, cst.Tuple):
            return updated_node

        call = stmt.value
        if not isinstance(call, cst.Call):
            return updated_node
        if not isinstance(call.func, cst.Name):
            return updated_node

        func_name = call.func.value
        transform = self._local_transforms.get(func_name)
        if transform is None:
            return updated_node

        # Collect unpacking variable names
        names = []
        for el in target.elements:
            if isinstance(el, cst.Element) and isinstance(el.value, cst.Name):
                names.append(el.value.value)
            else:
                return updated_node  # complex target (subscript, attribute), skip

        if len(names) != len(transform.field_names):
            return updated_node

        try:
            pos = self.get_metadata(PositionProvider, original_node)
            lineno = pos.start.line
        except KeyError:  # pragma: no cover
            lineno = 0

        tmp = _tmp_name(transform.dataclass_name, self.source)

        first = cst.SimpleStatementLine(
            body=[
                cst.Assign(
                    targets=[cst.AssignTarget(target=cst.Name(tmp))],
                    value=call,
                )
            ],
            leading_lines=original_node.leading_lines,
        )
        rest = [
            cst.SimpleStatementLine(
                body=[
                    cst.Assign(
                        targets=[cst.AssignTarget(target=cst.Name(var))],
                        value=cst.Attribute(
                            value=cst.Name(tmp),
                            attr=cst.Name(field),
                        ),
                    )
                ]
            )
            for var, field in zip(names, transform.field_names)
        ]

        self.changes_made.append(
            f"CallerUpdater: expanded {transform.dataclass_name} unpacking"
            f" at line {lineno}"
        )
        return cst.FlattenSentinel([first] + rest)
