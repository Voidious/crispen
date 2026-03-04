from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import _add_re_exports
from tests.codegen_test_fixtures import _make_entity


def _setup_add_re_exports_private_external_loads():
    source = "import os\n"
    entity = _make_entity("_helper", 1, 2)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(
        source, [placement], {"_helper": entity}, {}, external_loads={"_helper"}
    )
    return result
