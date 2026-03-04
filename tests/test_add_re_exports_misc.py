from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import _add_re_exports
from tests.codegen_test_fixtures import _make_entity


def test_add_re_exports_entity_not_in_map_falls_back_to_entity_name():
    # Entity name in group is missing from entity_map → falls back to entity name.
    source = "import os\n\nghost()\n"  # 'ghost' is still referenced
    placement = GroupPlacement(group=["ghost"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {}, {})
    assert "from .utils import ghost" in result


def test_add_re_exports_syntax_error_returns_source_unchanged():
    # If the source has a SyntaxError, _add_re_exports cannot determine where
    # to insert re-exports and must return the source unchanged.
    source = "import os\ndef (invalid\n"
    entity = _make_entity("baz", 1, 1)
    placement = GroupPlacement(group=["baz"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"baz": entity}, {})
    assert result == source
