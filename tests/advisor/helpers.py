from __future__ import annotations
from crispen.config import CrispenConfig
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.entity_parser import Entity, EntityKind


def _make_entity(
    name: str,
    start: int,
    end: int,
    *,
    docstring=None,
    params=None,
) -> Entity:
    return Entity(
        EntityKind.FUNCTION,
        name,
        start,
        end,
        [name],
        docstring=docstring,
        params=params or [],
    )


def _classified(
    *,
    entities=None,
    entity_class=None,
    graph=None,
    set_1=None,
    set_2_groups=None,
    set_3_groups=None,
    abort=False,
) -> ClassifiedEntities:
    return ClassifiedEntities(
        entities=entities or [],
        entity_class=entity_class or {},
        graph=graph if graph is not None else {},
        set_1=set_1 or [],
        set_2_groups=set_2_groups or [],
        set_3_groups=set_3_groups or [],
        abort=abort,
    )


_CONFIG = CrispenConfig()
_PATCH_KEY = "crispen.file_limiter.advisor.get_api_key"
_PATCH_CLIENT = "crispen.file_limiter.advisor.make_client"
_PATCH_CALL = "crispen.file_limiter.advisor.call_with_tool"

_CONFLICTING_PLACEMENTS = [
    GroupPlacement(group=["foo"], target_file="utils.py"),  # plan-vs-plan conflict
    GroupPlacement(group=["bar"], target_file="utils/io.py"),  # plan-vs-plan conflict
    GroupPlacement(group=["baz"], target_file="helpers.py"),  # not conflicting
]

_CLEAN_PLACEMENTS = [
    GroupPlacement(group=["foo"], target_file="utils.py"),
    GroupPlacement(group=["bar"], target_file="helpers.py"),
]
