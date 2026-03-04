from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import generate_file_splits
from tests.helpers import _classified, _make_entity, _plan


def _setup_generate_test(
    source: str, entity_name: str = "foo", target_file: str = "utils.py"
):
    entity = _make_entity(entity_name, 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=[entity_name], target_file=target_file)])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files[target_file]
    return new_src


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
