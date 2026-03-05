from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import generate_file_splits
from tests.test_helpers import _classified, _make_entity, _plan
from tests.test_models import GenerateFileSplitsSetupResult


def _generate_file_splits_setup(source, target_file="utils.py", source_file="big.py"):
    e_foo = _make_entity("foo", 3, 4)
    e_bar = _make_entity("bar", 6, 7)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["foo"], target_file=target_file)])
    result = generate_file_splits(c, plan, source, source_file)
    return GenerateFileSplitsSetupResult(
        e_foo=e_foo, e_bar=e_bar, c=c, plan=plan, result=result
    )
