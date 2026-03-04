from __future__ import annotations
import ast
from typing import Dict, Optional, Set
from .entity_parser import Entity, EntityKind
from .ast_utils import _collect_name_loads
from .imports import _import_line_numbers


def _update_deps_for_scc_selection(
    deps: Dict[str, Set[str]],
    selected_file: str,
    wanting_files: Set[str],
    helper_names: Set[str],
    entity_source_map: Dict[str, str],
    name_to_target_file: Dict[str, str],
    file_entity_names: Dict[str, Set[str]],
) -> None:
    for wanting_file in wanting_files:
        if wanting_file != selected_file:
            deps[wanting_file].add(selected_file)

    for helper_name in helper_names:
        src = entity_source_map.get(helper_name, "")
        for ref_name in _collect_name_loads(src):
            dep_file = name_to_target_file.get(ref_name)
            if dep_file and dep_file != selected_file and dep_file in file_entity_names:
                deps[selected_file].add(dep_file)


def _mark_lines_for_removal_when_pruned(
    *,
    kept_len: int,
    original_len: int,
    node: ast.AST,
    line_map: Dict[int, Optional[str]],
) -> bool:
    if kept_len == original_len:
        return False

    lineno = node.lineno
    end_lineno = node.end_lineno

    for ln in range(lineno, end_lineno + 1):
        line_map[ln] = None

    return True


def _remove_entity_lines(
    source: str,
    migrated_names: Set[str],
    entity_map: Dict[str, Entity],
    entity_source_map: Dict[str, str],
) -> str:
    """Return *source* with lines belonging to migrated entities removed.

    For TOP_LEVEL entities, import statement lines are preserved in the
    original file even when the entity is migrated: the remaining code may
    still reference those imported names, and stdlib/third-party names
    cannot be safely re-exported from a new module.
    """
    remove: Set[int] = set()
    preserve: Set[int] = set()
    for name in migrated_names:
        entity = entity_map.get(name)
        if entity is None:
            continue
        for ln in range(entity.start_line, entity.end_line + 1):
            remove.add(ln)
        if entity.kind == EntityKind.TOP_LEVEL:
            preserve |= _import_line_numbers(entity, entity_source_map.get(name, ""))

    lines = source.splitlines(keepends=True)
    return "".join(
        line for i, line in enumerate(lines, 1) if i not in remove or i in preserve
    )
