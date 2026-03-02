from __future__ import annotations
from crispen.file_limiter.advisor import FileLimiterPlan
from crispen.file_limiter.classifier import ClassifiedEntities
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
