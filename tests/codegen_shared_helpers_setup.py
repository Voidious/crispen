from __future__ import annotations
from crispen.file_limiter.code_gen import _extract_shared_helpers


def _setup_shared_helpers_extraction(
    entity_source_map,
    entity_map,
    classified,
    migrated_names,
    original_file: str,
):
    file_entity_names = {"f1.py": ["fn_1"], "f2.py": ["fn_2"]}
    name_to_target_file = {
        "helper_a": original_file,
        "helper_b": original_file,
        "fn_1": "f1.py",
        "fn_2": "f2.py",
    }

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        original_file,
    )

    return file_entity_names, name_to_target_file, synthetic
