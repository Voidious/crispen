from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import generate_file_splits
from tests.codegen_test_fixtures import _classified, _make_entity, _plan


def _setup_generate_splits_for_foo_migration(source: str):
    e_foo = _make_entity("foo", 3, 4)
    e_bar = _make_entity("bar", 6, 7)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])
    result = generate_file_splits(c, plan, source, "big.py")
    return result
