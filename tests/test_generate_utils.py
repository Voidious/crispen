from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import generate_file_splits
from tests.test_helpers import _classified, _make_entity, _plan


def _generate_and_get_new_src(
    source, entity_name="foo", start=3, end=4, target_file="utils.py"
):
    entity = _make_entity(entity_name, start, end)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=[entity_name], target_file=target_file)])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files[target_file]
    return result, new_src
