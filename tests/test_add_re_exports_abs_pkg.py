from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import _add_re_exports
from tests.codegen_test_fixtures import _make_entity


def test_add_re_exports_abs_pkg_package_prefix():
    # abs_pkg="tests" → absolute import: "from tests.utils import foo"
    source = "import os\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {}, abs_pkg="tests")
    assert "from tests.utils import foo" in result
    assert "from .utils import foo" not in result


def test_add_re_exports_abs_pkg_root_level():
    # abs_pkg="" → root-level absolute import: "from utils import foo"
    source = "import os\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {}, abs_pkg="")
    assert "from utils import foo" in result
    assert "from .utils import foo" not in result
