from __future__ import annotations
from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.file_limiter.code_gen import _add_re_exports
from crispen.file_limiter.entity_parser import Entity, EntityKind
from tests.test_setup_results import SetupAddReExportsTestResult


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


def _setup_pkg_with_utils(
    tmp_path, pkg_name: str = "mypkg", mod_name: str = "utils.py"
):
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / pkg_name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    mod = pkg / mod_name
    mod.write_text("def _helper():\n    pass\n")
    return pkg, mod


def _setup_add_re_exports_test():
    source = "import os\n"
    entity = _make_entity("_helper", 1, 2)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(
        source, [placement], {"_helper": entity}, {}, external_loads={"_helper"}
    )
    return SetupAddReExportsTestResult(
        source=source, entity=entity, placement=placement, result=result
    )
