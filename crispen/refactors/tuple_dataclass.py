"""Refactor: large tuple literals → @dataclass."""

from __future__ import annotations

import ast
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import libcst as cst
from libcst.metadata import PositionProvider

from .base import Refactor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATACLASS_IMPORT = cst.parse_statement("from dataclasses import dataclass\n")
_ANY_IMPORT = cst.parse_statement("from typing import Any\n")


def _name_str(node: cst.BaseExpression) -> Optional[str]:
    """Return the string value of a Name node, or None."""
    if isinstance(node, cst.Name):
        return node.value
    return None


def _is_variable_index(node: cst.BaseExpression) -> bool:
    """Return True if node is not an integer literal."""
    return not isinstance(node, cst.Integer)


def _int_val(node: cst.BaseExpression) -> Optional[int]:
    if isinstance(node, cst.Integer):
        return int(node.value)
    return None


def _snake_to_pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_") if part)


def _all_callers_unpack(source: str, func_name: str) -> bool:
    """Return True if every call to *func_name* in *source* is a tuple-unpacking.

    A "tuple-unpacking" call is one that is the direct RHS of a single-target
    assignment whose target is an ``ast.Tuple``:  ``a, b = func_name(...)``.

    Returns True when there are no calls to *func_name* at all (vacuously true),
    and False if *source* cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    # Collect ids of Call nodes that ARE used in tuple-unpacking assignments.
    unpacking_call_ids: set = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == func_name
        ):
            unpacking_call_ids.add(id(node.value))

    # Any call to func_name NOT in the unpacking set is a non-unpacking usage.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == func_name
            and id(node) not in unpacking_call_ids
        ):
            return False

    return True


class TransformInfo(NamedTuple):
    """Record of a tuple→dataclass transformation applied to a function."""

    func_name: str
    dataclass_name: str
    field_names: List[str]


# ---------------------------------------------------------------------------
# Pass 1: collect unpacking assignments
# ---------------------------------------------------------------------------


class _UnpackingCollector(cst.CSTVisitor):
    """Collect `a, b, c = <expr>` unpacking assignments."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self) -> None:
        self.unpackings: Dict[int, List[str]] = {}  # tuple_start_line → field names

    def visit_Assign(self, node: cst.Assign) -> None:
        # We only care about a single target that is a Tuple/StarredElement pattern
        if len(node.targets) != 1:
            return
        target = node.targets[0].target
        if not isinstance(target, cst.Tuple):
            return
        names = []
        for el in target.elements:
            if isinstance(el, cst.Element) and isinstance(el.value, cst.Name):
                names.append(el.value.value)
            else:
                return  # give up if anything complex

        # Record under the line of the RHS expression
        try:
            pos = self.get_metadata(PositionProvider, node.value)
            self.unpackings[pos.start.line] = names
        except KeyError:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Main transformer
# ---------------------------------------------------------------------------


class TupleDataclass(Refactor):
    """Replace large tuple literals with @dataclass instances.

    Only operates on tuples with at least `min_size` elements (default 3)
    that fall within a changed line range.
    """

    def __init__(
        self,
        changed_ranges: List[Tuple[int, int]],
        min_size: int = 4,
        source: str = "",
        verbose: bool = True,
        approved_public_funcs: Optional[Set[str]] = None,
        blocked_scopes: Optional[Set[str]] = None,
    ) -> None:
        super().__init__(changed_ranges, source=source, verbose=verbose)
        self.min_size = min_size
        self._source = source
        self.approved_public_funcs: Set[str] = set(approved_public_funcs or [])
        self._blocked_scopes: Set[str] = set(blocked_scopes or [])

        # State populated during the first visit pass
        self._unpackings: Dict[int, List[str]] = {}

        # Public function candidates discovered (whether or not transformed)
        self._candidate_public_transforms: Dict[str, TransformInfo] = {}

        # Private function transforms actually applied
        self._private_transforms: Dict[str, TransformInfo] = {}

        # Dataclasses to inject: list of (class_name, field_names, values)
        # Keyed by class name to avoid duplicates
        self._pending_classes: Dict[str, Tuple[List[str], List[cst.BaseExpression]]] = (
            {}
        )

        # Replacements: original Tuple node → Name("ClassName")
        self._tuple_replacements: Dict[int, Tuple[str, List[str]]] = (
            {}
        )  # id(node) → (class_name, fields)

        # Whether we need to inject imports
        self._need_dataclass_import = False
        self._need_any_import = False
        self._has_dataclass_import = False
        self._has_any_import = False

        # Track enclosing function/class names for naming
        self._scope_stack: List[Optional[str]] = []

        # Counter to detect tuples inside function call arguments
        self._in_call_arg: int = 0

        # Counter to detect tuples inside return statements
        self._in_return: int = 0

    # ------------------------------------------------------------------
    # Collect context before transforms
    # ------------------------------------------------------------------

    def visit_Module(self, node: cst.Module) -> Optional[bool]:
        self._scope_stack = [None]
        return None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        self._scope_stack.append(node.name.value)
        return None

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.BaseStatement:
        self._scope_stack.pop()
        return updated_node

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self._scope_stack.append(node.name.value)
        return None

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.BaseStatement:
        self._scope_stack.pop()
        return updated_node

    def visit_ImportFrom(self, node: cst.ImportFrom) -> Optional[bool]:
        if isinstance(node.module, cst.Attribute):
            return None
        if isinstance(node.module, cst.Name) and node.module.value == "dataclasses":
            self._has_dataclass_import = True
        if isinstance(node.module, cst.Name) and node.module.value == "typing":
            if isinstance(node.names, (list, tuple)):
                for alias in node.names:
                    if isinstance(
                        alias, cst.ImportAlias
                    ) and isinstance(  # pragma: no branch
                        alias.name, cst.Name
                    ):
                        if alias.name.value == "Any":
                            self._has_any_import = True
        return None

    def visit_Arg(self, node: cst.Arg) -> Optional[bool]:
        self._in_call_arg += 1
        return None

    def leave_Arg(self, original_node: cst.Arg, updated_node: cst.Arg) -> cst.Arg:
        self._in_call_arg -= 1
        return updated_node

    def visit_Return(self, node: cst.Return) -> Optional[bool]:
        self._in_return += 1
        return None

    def leave_Return(
        self, original_node: cst.Return, updated_node: cst.Return
    ) -> cst.BaseStatement:
        self._in_return -= 1
        return updated_node

    # ------------------------------------------------------------------
    # Core: detect and schedule tuple replacements
    # ------------------------------------------------------------------

    def _current_scope_name(self) -> Optional[str]:
        for name in reversed(self._scope_stack):
            if name is not None:
                return name
        return None

    def _class_name_for(self, assign_target_name: Optional[str]) -> str:
        scope = self._current_scope_name()
        if scope:
            return f"{_snake_to_pascal(scope)}Result"
        if assign_target_name:
            return f"{_snake_to_pascal(assign_target_name)}Tuple"
        return "DataTuple"

    def _field_names_for(self, tuple_node: cst.Tuple, lineno: int) -> List[str]:
        if lineno in self._unpackings:
            names = self._unpackings[lineno]
            if len(names) == len(tuple_node.elements):
                return names
        # Infer names from variable names in the tuple itself
        names = []
        for el in tuple_node.elements:
            if isinstance(el, cst.Element) and isinstance(el.value, cst.Name):
                names.append(el.value.value)
            else:
                names.append(None)
        if all(n is not None for n in names):
            return names
        return [f"field_{i}" for i in range(len(tuple_node.elements))]

    def _is_safe_tuple(self, node: cst.Tuple) -> bool:
        """Return False if the tuple contains starred elements."""
        for el in node.elements:
            if isinstance(el, cst.StarredElement):
                return False
        return True

    def _element_values(self, node: cst.Tuple) -> List[cst.BaseExpression]:
        return [el.value for el in node.elements if isinstance(el, cst.Element)]

    def leave_Tuple(
        self, original_node: cst.Tuple, updated_node: cst.Tuple
    ) -> cst.BaseExpression:
        if len(updated_node.elements) < self.min_size:
            return updated_node
        if self._in_return == 0:
            return updated_node
        if self._in_call_arg > 0:
            return updated_node
        if not self._in_changed_range(original_node):
            return updated_node
        if not self._is_safe_tuple(updated_node):
            return updated_node

        scope = self._current_scope_name()
        if scope is None:
            return updated_node
        is_public = not scope.startswith("_")

        if not is_public and scope in self._blocked_scopes:
            return updated_node

        # Only transform when every in-file caller uses tuple unpacking on the
        # return value.  This prevents broken code when a caller stores or passes
        # the result without unpacking (e.g. lst.append(func()) or x = func()).
        if not _all_callers_unpack(self._source, scope):
            return updated_node

        try:
            pos = self.get_metadata(PositionProvider, original_node)
            lineno = pos.start.line
        except KeyError:  # pragma: no cover
            return updated_node

        class_name = self._class_name_for(None)
        field_names = self._field_names_for(updated_node, lineno)
        values = self._element_values(updated_node)

        if is_public:
            self._candidate_public_transforms[scope] = TransformInfo(
                func_name=scope,
                dataclass_name=class_name,
                field_names=field_names,
            )
            if scope not in self.approved_public_funcs:
                return updated_node

        # Reuse field names from a prior return path of the same function so
        # all constructor calls use a consistent set of keyword arguments.
        if class_name in self._pending_classes:
            field_names = self._pending_classes[class_name][0]
        else:
            self._pending_classes[class_name] = (field_names, values)

        if not is_public:
            self._private_transforms[scope] = TransformInfo(
                func_name=scope,
                dataclass_name=class_name,
                field_names=field_names,
            )

        if len(values) != len(field_names):  # pragma: no cover
            return updated_node
        self._tuple_replacements[id(original_node)] = (class_name, field_names)

        # Build replacement call: ClassName(field=val, ...)
        args = [
            cst.Arg(keyword=cst.Name(fname), value=val)
            for fname, val in zip(field_names, values)
        ]
        # Add commas between args
        return self._construct_dataclass_call(args, class_name, field_names, lineno)

    def _construct_dataclass_call(self, args, class_name, field_names, lineno):
        args_with_comma = []
        for i, arg in enumerate(args):
            if i < len(args) - 1:
                args_with_comma.append(
                    arg.with_changes(comma=cst.MaybeSentinel.DEFAULT)
                )
            else:
                args_with_comma.append(arg)

        self._need_dataclass_import = True
        self._need_any_import = True
        n = len(field_names)
        self.changes_made.append(
            f"TupleDataclass: replaced {n}-tuple with {class_name} at line {lineno}"
        )
        return cst.Call(func=cst.Name(class_name), args=args_with_comma)

    def get_private_transforms(self) -> Dict[str, "TransformInfo"]:
        """Return private functions whose tuples were actually transformed."""
        return dict(self._private_transforms)

    def get_candidate_public_transforms(self) -> Dict[str, "TransformInfo"]:
        """Return public functions whose tuples were candidates for transformation.

        These are public functions in the changed range with large-enough tuples.
        They may or may not have been transformed (depends on approved_public_funcs).
        """
        return dict(self._candidate_public_transforms)

    # ------------------------------------------------------------------
    # Inject dataclass definitions and imports at module level
    # ------------------------------------------------------------------

    def _build_dataclass(self, class_name: str, field_names: List[str]) -> cst.ClassDef:
        fields = []
        for fname in field_names:
            ann = cst.Annotation(annotation=cst.Name("Any"))
            stmt = cst.SimpleStatementLine(
                body=[cst.AnnAssign(target=cst.Name(fname), annotation=ann, value=None)]
            )
            fields.append(stmt)

        return cst.ClassDef(
            decorators=[cst.Decorator(decorator=cst.Name("dataclass"))],
            name=cst.Name(class_name),
            body=cst.IndentedBlock(body=fields),
            leading_lines=(cst.EmptyLine(), cst.EmptyLine()),
        )

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        if not self._pending_classes:
            return updated_node

        new_body = list(updated_node.body)

        # Inject dataclass definitions before the first function/class def
        # that isn't an import statement, but after all imports
        insert_pos = 0
        for i, stmt in enumerate(new_body):  # pragma: no branch
            if isinstance(stmt, (cst.SimpleStatementLine,)):
                # imports and simple assignments
                insert_pos = i + 1
            else:
                break

        existing_class_names = {
            stmt.name.value
            for stmt in new_body
            if isinstance(stmt, cst.ClassDef) and isinstance(stmt.name, cst.Name)
        }
        class_stmts = []
        for class_name, (field_names, _values) in self._pending_classes.items():
            if class_name not in existing_class_names:
                class_stmts.append(self._build_dataclass(class_name, field_names))

        for i, cls_stmt in enumerate(reversed(class_stmts)):
            new_body.insert(insert_pos, cls_stmt)

        # Ensure 2 blank lines between the last inserted class and the next stmt.
        following_idx = insert_pos + len(class_stmts)
        if class_stmts and following_idx < len(new_body):
            following = new_body[following_idx]
            new_body[following_idx] = following.with_changes(
                leading_lines=(cst.EmptyLine(), cst.EmptyLine())
            )

        # Inject imports at the very top (after any __future__ imports)
        import_insert_pos = 0
        for i, stmt in enumerate(new_body):
            if isinstance(stmt, cst.SimpleStatementLine):
                for s in stmt.body:
                    if isinstance(s, cst.ImportFrom):
                        mod = s.module
                        if isinstance(
                            mod, cst.Attribute
                        ) and isinstance(  # pragma: no cover
                            mod.value, cst.Name
                        ):
                            if mod.value.value == "__future__":
                                import_insert_pos = i + 1
                        elif isinstance(mod, cst.Name) and mod.value == "__future__":
                            import_insert_pos = i + 1

        imports_to_add = []
        if self._need_any_import and not self._has_any_import:
            imports_to_add.append(_ANY_IMPORT)
        if self._need_dataclass_import and not self._has_dataclass_import:
            imports_to_add.append(_DATACLASS_IMPORT)

        for imp in reversed(imports_to_add):
            new_body.insert(import_insert_pos, imp)

        return updated_node.with_changes(body=new_body)
