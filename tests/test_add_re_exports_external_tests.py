from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import _add_re_exports
from .entity_utils import _make_entity


def test_add_re_exports_private_in_external_loads():
    # Private name not referenced in remaining source but present in external_loads
    # → re-export proxy IS added so the external caller continues to work.
    source = "import os\n"
    entity = _make_entity("_helper", 1, 2)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(
        source, [placement], {"_helper": entity}, {}, external_loads={"_helper"}
    )
    assert "from .utils import _helper" in result


def test_add_re_exports_test_function_in_external_loads_not_re_exported():
    # test_ functions must never get a proxy even when listed in external_loads,
    # because pytest would discover and run them twice.
    source = "import os\n"
    entity = _make_entity("test_something", 1, 2)
    placement = GroupPlacement(group=["test_something"], target_file="helpers.py")
    result = _add_re_exports(
        source,
        [placement],
        {"test_something": entity},
        {},
        external_loads={"test_something"},
    )
    assert result == source
