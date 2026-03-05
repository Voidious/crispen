from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import generate_file_splits
from tests.test_code_gen_planning_helpers import _classified, _make_entity, _plan


def _generate_file_splits_with_two_entities(
    source: str,
    big_filename: str = "big.py",
    foo_name: str = "foo",
    bar_name: str = "bar",
    foo_start: int = 3,
    foo_end: int = 4,
    bar_start: int = 6,
    bar_end: int = 7,
    target_group: list[str] | None = None,
    target_file: str = "utils.py",
):
    e_foo = _make_entity(foo_name, foo_start, foo_end)
    e_bar = _make_entity(bar_name, bar_start, bar_end)
    c = _classified(entities=[e_foo, e_bar])
    if target_group is None:
        target_group = [foo_name]
    plan = _plan([GroupPlacement(group=target_group, target_file=target_file)])

    result = generate_file_splits(c, plan, source, big_filename)
    return result


def test_generate_file_splits_removes_inline_redundant_imports():
    # When a split new file has both a top-level import and an inline re-import
    # of the same name, the inline one should be removed.
    source = textwrap.dedent(
        """\
        from mymod import Helper

        def test_uses_helper():
            from mymod import Helper
            assert Helper()
        """
    )
    entity = _make_entity("test_uses_helper", 3, 5)
    c = _classified(entities=[entity])
    plan = _plan(
        [GroupPlacement(group=["test_uses_helper"], target_file="test_split.py")]
    )
    result = generate_file_splits(c, plan, source, "big.py")
    new_src = result.new_files["test_split.py"]
    # The inline re-import should be removed; the module-level one covers it.
    assert new_src.count("from mymod import Helper") == 1
