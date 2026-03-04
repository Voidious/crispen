from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import generate_file_splits
from tests.codegen_test_fixtures import _classified, _make_entity, _plan


def _gen_utils_new_src_for_single_entity(
    *, source: str, original_filename: str
) -> tuple[object, str]:
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, original_filename)

    new_src = result.new_files["utils.py"]
    return result, new_src
