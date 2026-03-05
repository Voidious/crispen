from __future__ import annotations
from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import _extract_shared_helpers, generate_file_splits
from crispen.file_limiter.entity_parser import Entity, EntityKind


def _make_entity(name: str, start: int, end: int, defines=None) -> Entity:
    return Entity(EntityKind.FUNCTION, name, start, end, defines or [name])


def _classified(
    *, entities=None, set_2_groups=None, set_3_groups=None
) -> ClassifiedEntities:
    return ClassifiedEntities(
        entities=entities or [],
        entity_class={},
        graph={},
        set_1=[],
        set_2_groups=set_2_groups or [],
        set_3_groups=set_3_groups or [],
        abort=False,
    )


def _plan(placements=None) -> FileLimiterPlan:
    return FileLimiterPlan(set3_migrate=[], placements=placements or [], abort=False)


def _abort_plan() -> FileLimiterPlan:
    return FileLimiterPlan(set3_migrate=[], placements=[], abort=True)


def _make_generated_new_src(
    source: str, target_file: str = "utils.py", original_file: str = "big.py"
):
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file=target_file)])

    result = generate_file_splits(c, plan, source, original_file)

    return result.new_files[target_file]


def _make_classified(entities, migrated_names=None):
    migrated = set(migrated_names or [])
    return (
        ClassifiedEntities(
            entities=entities,
            entity_class={},
            graph={},
            set_1=[],
            set_2_groups=[],
            set_3_groups=[],
            abort=False,
        ),
        migrated,
    )


def _make_shared_helper_extraction_inputs(
    entity_source_map, entity_map, classified, migrated_names
):
    file_entity_names = {"f1.py": ["fn_1"], "f2.py": ["fn_2"]}
    name_to_target_file = {
        "helper_a": "original.py",
        "helper_b": "original.py",
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
        "original.py",
    )
    return file_entity_names, name_to_target_file, synthetic
