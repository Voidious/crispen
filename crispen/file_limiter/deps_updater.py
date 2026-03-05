from __future__ import annotations
from typing import Dict, Set
from .imports import _collect_name_loads


def _update_deps_for_target(
    target_file: str,
    deps: Dict[str, Set[str]],
    scc: Set[str],
    scc_wanting: Set[str],
    entity_source_map: Dict[str, str],
    name_to_target_file: Dict[str, str],
    file_entity_names: Set[str],
) -> None:
    for wanting_file in scc_wanting:
        if wanting_file != target_file:
            deps[wanting_file].add(target_file)
    for helper_name in scc:
        src = entity_source_map.get(helper_name, "")
        for ref_name in _collect_name_loads(src):
            dep_file = name_to_target_file.get(ref_name)
            if dep_file and dep_file != target_file and dep_file in file_entity_names:
                deps[target_file].add(dep_file)
