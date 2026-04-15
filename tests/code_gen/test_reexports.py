from __future__ import annotations
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    _add_re_exports,
    _class_has_test_methods,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .test_shared_helpers_extraction import _classified, _make_entity, _plan


def test_add_re_exports_top_level_import_derived_names_not_re_exported():
    # A TOP_LEVEL entity that includes import statements: the names introduced
    # by those imports must NOT appear in re-exports because they are preserved
    # in the original file by _remove_entity_lines, not moved to the new file.
    source = "import os\n\nMY_CONST\n"  # MY_CONST still loaded
    entity_src = "from typing import Dict\n\nMY_CONST = 42\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["Dict", "MY_CONST"])
    placement = GroupPlacement(group=["_block_1"], target_file="constants.py")
    result = _add_re_exports(
        source, [placement], {"_block_1": entity}, {"_block_1": entity_src}
    )
    assert "MY_CONST" in result  # assignment-defined name re-exported
    assert "Dict" not in result  # import-derived name suppressed


def test_add_re_exports_all_private_no_change():
    # Private name not called anywhere in remaining source → no import added.
    source = "import os\n\ndef _helper():\n    pass\n"
    entity = _make_entity("_helper", 3, 4)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"_helper": entity}, {})
    assert result == source


def test_add_re_exports_private_referenced_in_source():
    # Private name still called in remaining source → import is added.
    source = "import os\n\n_helper()\n"
    entity = _make_entity("_helper", 3, 3)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"_helper": entity}, {})
    assert "from .utils import _helper" in result


def test_add_re_exports_public_inserted_after_imports():
    source = "import os\n\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 3, 4)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {})
    assert "from .utils import foo" in result
    # Re-export line should come after "import os"
    lines = result.splitlines()
    import_idx = next(i for i, l in enumerate(lines) if "import os" in l)
    reexport_idx = next(i for i, l in enumerate(lines) if "from .utils import foo" in l)
    assert reexport_idx > import_idx


def test_add_re_exports_no_import_in_source():
    # No imports and no docstring → re-export inserted at beginning.
    source = "\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 2, 3)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {})
    assert "from .utils import foo" in result


def test_add_re_exports_no_import_with_module_docstring():
    # No imports but module docstring present → re-export inserted after docstring,
    # not before it, so the docstring remains the first statement.
    source = '"""Module docstring."""\n\n\ndef foo():\n    pass\n'
    entity = _make_entity("foo", 4, 5)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {})
    lines = result.splitlines()
    docstring_idx = next(
        i for i, l in enumerate(lines) if '"""Module docstring."""' in l
    )
    reexport_idx = next(i for i, l in enumerate(lines) if "from .utils import foo" in l)
    assert docstring_idx == 0
    assert reexport_idx > docstring_idx


def test_add_re_exports_from_import_line():
    # "from pathlib import Path" should be detected as an import line.
    source = "from pathlib import Path\n\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 3, 4)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {})
    lines = result.splitlines()
    from_import_idx = next(
        i for i, l in enumerate(lines) if "from pathlib import Path" in l
    )
    reexport_idx = next(i for i, l in enumerate(lines) if "from .utils import foo" in l)
    assert reexport_idx > from_import_idx


def test_add_re_exports_multiple_targets_sorted():
    source = "import os\n"
    e1 = _make_entity("foo", 1, 2)
    e2 = _make_entity("bar", 3, 4)
    placements = [
        GroupPlacement(group=["foo"], target_file="b_module.py"),
        GroupPlacement(group=["bar"], target_file="a_module.py"),
    ]
    result = _add_re_exports(source, placements, {"foo": e1, "bar": e2}, {})
    # a_module comes before b_module (sorted)
    a_idx = result.index("a_module")
    b_idx = result.index("b_module")
    assert a_idx < b_idx


def test_add_re_exports_mixed_public_private():
    source = "import os\n"
    entity_map = {
        "pub": _make_entity("pub", 1, 2),
        "_priv": _make_entity("_priv", 3, 4),
    }
    placement = GroupPlacement(group=["pub", "_priv"], target_file="utils.py")
    result = _add_re_exports(source, [placement], entity_map, {})
    # Only "pub" in re-export, not "_priv"
    assert "pub" in result
    assert "_priv" not in result


def test_add_re_exports_test_function_not_re_exported():
    # test_ functions must never get a proxy import — pytest would discover and
    # run them twice (once from the original file, once from the new file).
    source = "import os\n"
    entity = _make_entity("test_something", 1, 3)
    placement = GroupPlacement(group=["test_something"], target_file="tests/helpers.py")
    result = _add_re_exports(source, [placement], {"test_something": entity}, {})
    assert result == source


def test_add_re_exports_test_function_never_re_exported_even_when_referenced():
    # test_* names are never re-exported at module level even when the
    # remaining source references them — _inject_inline_test_imports_original
    # handles those cases inline to prevent pytest double-discovery.
    source = "import os\n\ntest_something()\n"
    entity = _make_entity("test_something", 1, 3)
    placement = GroupPlacement(group=["test_something"], target_file="tests/helpers.py")
    result = _add_re_exports(source, [placement], {"test_something": entity}, {})
    assert "from .tests.helpers import test_something" not in result


def test_class_has_test_methods_true():
    src = "class TestFoo:\n    def test_bar(self): pass\n"
    assert _class_has_test_methods(src) is True


def test_class_has_test_methods_false():
    src = "class Helper:\n    def run(self): pass\n"
    assert _class_has_test_methods(src) is False


def test_class_has_test_methods_syntax_error():
    assert _class_has_test_methods("def (") is False


def test_add_re_exports_test_class_not_re_exported():
    # A class that contains test_ methods must not be re-exported — pytest
    # would discover it via the original file and the new file, running every
    # test twice.
    source = "import os\n"
    entity = Entity(EntityKind.CLASS, "TestFoo", 1, 5, ["TestFoo"])
    entity_src = "class TestFoo:\n    def test_bar(self): pass\n"
    placement = GroupPlacement(group=["TestFoo"], target_file="tests/helpers.py")
    result = _add_re_exports(
        source, [placement], {"TestFoo": entity}, {"TestFoo": entity_src}
    )
    assert result == source


def test_add_re_exports_test_class_never_re_exported_even_when_referenced():
    # Test-named symbols are never re-exported at module level even when
    # referenced in remaining source — _inject_inline_test_imports_original
    # handles them inline to prevent pytest double-discovery.
    source = "import os\n\nTestFoo()\n"
    entity = Entity(EntityKind.CLASS, "TestFoo", 1, 5, ["TestFoo"])
    entity_src = "class TestFoo:\n    def test_bar(self): pass\n"
    placement = GroupPlacement(group=["TestFoo"], target_file="tests/helpers.py")
    result = _add_re_exports(
        source, [placement], {"TestFoo": entity}, {"TestFoo": entity_src}
    )
    assert "from .tests.helpers import TestFoo" not in result


def test_add_re_exports_top_level_block_private_names_referenced():
    # TOP_LEVEL block entity name (_block_1) differs from its defined names.
    # Both defined names are still loaded in remaining source → re-imported.
    source = "import os\n\n_DUP_SOURCE\n_DUP_RANGES\n"
    entity = Entity(
        EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_DUP_SOURCE", "_DUP_RANGES"]
    )
    placement = GroupPlacement(group=["_block_1"], target_file="test_helpers.py")
    result = _add_re_exports(source, [placement], {"_block_1": entity}, {})
    assert "from .test_helpers import _DUP_RANGES, _DUP_SOURCE" in result


def test_add_re_exports_entity_not_in_map_falls_back_to_entity_name():
    # Entity name in group is missing from entity_map → falls back to entity name.
    source = "import os\n\nghost()\n"  # 'ghost' is still referenced
    placement = GroupPlacement(group=["ghost"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {}, {})
    assert "from .utils import ghost" in result


def test_add_re_exports_top_level_block_private_names_not_referenced():
    # TOP_LEVEL block entity whose defined name is private and not used → no import.
    source = "import os\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    placement = GroupPlacement(group=["_block_1"], target_file="constants.py")
    result = _add_re_exports(source, [placement], {"_block_1": entity}, {})
    assert result == source


def test_add_re_exports_indented_local_import_not_treated_as_last_import():
    # Functions with local (indented) imports must not cause re-exports to be
    # inserted inside the function body.  The re-export should appear after the
    # top-level "import os" line, not after the indented "from x import y".
    source = (
        "import os\n"
        "\n"
        "def foo():\n"
        "    from unittest.mock import MagicMock\n"
        "    MagicMock()\n"
        "\n"
        "def bar():\n"
        "    pass\n"
    )
    entity = _make_entity("baz", 7, 8)
    placement = GroupPlacement(group=["baz"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"baz": entity}, {})
    # Re-export must appear immediately after "import os", not inside foo().
    lines = result.splitlines()
    os_idx = next(i for i, l in enumerate(lines) if l == "import os")
    reexport_idx = next(i for i, l in enumerate(lines) if "from .utils import baz" in l)
    assert reexport_idx == os_idx + 1
    # The function body must remain intact (local import line must still be there).
    assert "    from unittest.mock import MagicMock" in result


def test_add_re_exports_syntax_error_returns_source_unchanged():
    # If the source has a SyntaxError, _add_re_exports cannot determine where
    # to insert re-exports and must return the source unchanged.
    source = "import os\ndef (invalid\n"
    entity = _make_entity("baz", 1, 1)
    placement = GroupPlacement(group=["baz"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"baz": entity}, {})
    assert result == source


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


def test_add_re_exports_mode_always_public_always_reexported():
    # "always" mode: public names are unconditionally re-exported (current behaviour).
    source = "import os\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(
        source, [placement], {"foo": entity}, {}, reexport_mode="always"
    )
    assert "from .utils import foo" in result


def test_add_re_exports_mode_application_non_test_public_reexported():
    # "application" mode + non-test file: public names are re-exported.
    source = "import os\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(
        source,
        [placement],
        {"foo": entity},
        {},
        reexport_mode="application",
        is_test_file=False,
    )
    assert "from .utils import foo" in result


def test_add_re_exports_mode_application_test_file_public_not_reexported():
    # "application" mode + test file: public names are NOT unconditionally re-exported.
    source = "import os\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(
        source,
        [placement],
        {"foo": entity},
        {},
        reexport_mode="application",
        is_test_file=True,
    )
    assert result == source


def test_add_re_exports_mode_application_test_file_in_external_loads_reexported():
    # "application" mode + test file: public name IS re-exported when in external_loads.
    source = "import os\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(
        source,
        [placement],
        {"foo": entity},
        {},
        external_loads={"foo"},
        reexport_mode="application",
        is_test_file=True,
    )
    assert "from .utils import foo" in result


def test_add_re_exports_mode_application_test_file_public_in_still_loaded_reexported():
    # "application" mode + test file: public name IS re-exported when still referenced.
    source = "import os\n\nfoo()\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(
        source,
        [placement],
        {"foo": entity},
        {},
        reexport_mode="application",
        is_test_file=True,
    )
    assert "from .utils import foo" in result


def test_add_re_exports_mode_imported_public_not_in_external_loads_not_reexported():
    # "imported" mode: public name is NOT re-exported if absent from external_loads.
    source = "import os\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(
        source, [placement], {"foo": entity}, {}, reexport_mode="imported"
    )
    assert result == source


def test_add_re_exports_mode_imported_public_in_external_loads_reexported():
    # "imported" mode: public name IS re-exported when in external_loads.
    source = "import os\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(
        source,
        [placement],
        {"foo": entity},
        {},
        external_loads={"foo"},
        reexport_mode="imported",
    )
    assert "from .utils import foo" in result


def test_add_re_exports_mode_imported_public_in_still_loaded_reexported():
    # "imported" mode: public name IS re-exported when still referenced in source.
    source = "import os\n\nfoo()\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(
        source, [placement], {"foo": entity}, {}, reexport_mode="imported"
    )
    assert "from .utils import foo" in result


def test_add_re_exports_mode_imported_private_in_external_loads_reexported():
    # "imported" mode: private names still follow the same rule (external_loads).
    source = "import os\n"
    entity = _make_entity("_helper", 1, 2)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(
        source,
        [placement],
        {"_helper": entity},
        {},
        external_loads={"_helper"},
        reexport_mode="imported",
    )
    assert "from .utils import _helper" in result


def test_generate_private_entity_reexported_when_external_caller(tmp_path):
    # Private entity is re-exported when an external file imports it.
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    mod = pkg / "big.py"
    mod.write_text("def _helper():\n    pass\n")
    caller = tmp_path / "tests" / "test_big.py"
    caller.parent.mkdir()
    caller.write_text("from mypkg.big import _helper\n")

    source = "def _helper():\n    pass\n"
    entity = _make_entity("_helper", 1, 2)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["_helper"], target_file="private.py")])

    result = generate_file_splits(c, plan, source, str(mod))

    assert "from .private import _helper" in result.original_source


def test_generate_file_splits_reexport_imported_public_not_reexported_without_caller(
    tmp_path,
):
    # "imported" mode: public entity not imported elsewhere → no re-export stub.
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    mod = pkg / "big.py"
    mod.write_text("def foo():\n    pass\n")
    # No external callers import foo.

    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, str(mod), reexport_mode="imported")

    assert "from .helpers import foo" not in result.original_source


def test_generate_file_splits_reexport_mode_imported_public_reexported_with_caller(
    tmp_path,
):
    # "imported" mode: public entity imported elsewhere → re-export stub is added.
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    mod = pkg / "big.py"
    mod.write_text("def foo():\n    pass\n")
    caller = tmp_path / "other.py"
    caller.write_text("from mypkg.big import foo\n")

    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, str(mod), reexport_mode="imported")

    assert "from .helpers import foo" in result.original_source


def test_generate_file_splits_reexport_mode_always_public_reexported_without_caller(
    tmp_path,
):
    # "always" mode: public entity re-exported even when no external callers exist.
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    mod = pkg / "big.py"
    mod.write_text("def foo():\n    pass\n")

    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, str(mod), reexport_mode="always")

    assert "from .helpers import foo" in result.original_source


def test_add_re_exports_private_external_only_gets_noqa():
    # Private name in external_loads but NOT in remaining source → fmt: skip # noqa comment.
    source = "import os\n"
    entity = _make_entity("_helper", 1, 2)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(
        source, [placement], {"_helper": entity}, {}, external_loads={"_helper"}
    )
    assert "from .utils import _helper  # fmt: skip # noqa: F401, E501" in result


def test_add_re_exports_private_in_still_loaded_no_noqa():
    # Private name referenced in remaining source but NOT in external_loads
    # → re-export without noqa (it is actively used; no future-pruning risk).
    source = "import os\n\n_helper()\n"
    entity = _make_entity("_helper", 3, 3)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"_helper": entity}, {})
    assert "from .utils import _helper\n" in result
    assert "# noqa" not in result


def test_add_re_exports_private_in_still_loaded_and_external_loads_gets_noqa():
    # Private name referenced in remaining source AND in external_loads → noqa
    # marker is added even though it is currently "used", because the non-migrated
    # entity that uses it may itself be migrated in a later recursive split, at
    # which point _prune_unused_imports would silently drop an un-annotated stub.
    source = "import os\n\n_helper()\n"
    entity = _make_entity("_helper", 3, 3)
    placement = GroupPlacement(group=["_helper"], target_file="utils.py")
    result = _add_re_exports(
        source, [placement], {"_helper": entity}, {}, external_loads={"_helper"}
    )
    assert "from .utils import _helper  # fmt: skip # noqa: F401, E501" in result


def test_add_re_exports_public_not_in_still_loaded_gets_noqa():
    # Public name migrated but not referenced in remaining source → fmt: skip # noqa.
    source = "import os\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {})
    assert "from .utils import foo  # fmt: skip # noqa: F401, E501" in result


def test_add_re_exports_public_in_still_loaded_no_noqa():
    # Public name still referenced in remaining source → re-export without noqa.
    source = "import os\n\nfoo()\n"
    entity = _make_entity("foo", 3, 3)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {})
    assert "from .utils import foo\n" in result
    assert "# noqa" not in result


def test_add_re_exports_multiple_noqa_each_on_own_line():
    # Two names both need noqa → one import line each so Black can't break the comment.
    source = "import os\n"
    entity = _make_entity("_block", 3, 4, ["_a", "_b"])
    placement = GroupPlacement(group=["_block"], target_file="utils.py")
    result = _add_re_exports(
        source,
        [placement],
        {"_block": entity},
        {},
        external_loads={"_a", "_b"},
    )
    lines = result.splitlines()
    noqa_lines = [line for line in lines if "# fmt: skip # noqa: F401, E501" in line]
    assert len(noqa_lines) == 2
    names = {line.split("import")[1].split("#")[0].strip() for line in noqa_lines}
    assert names == {"_a", "_b"}


def test_add_re_exports_mixed_splits_into_two_lines():
    # One entity defines two names: one in still_loaded, one purely re-exported.
    # Both are in external_loads, so both get # noqa: F401 to protect them from
    # being pruned if the non-migrated entity that currently uses _used is itself
    # migrated in a later recursive split.
    source = "import os\n\n_used()\n"
    entity = _make_entity("_block", 3, 4, ["_used", "_reexport"])
    placement = GroupPlacement(group=["_block"], target_file="utils.py")
    result = _add_re_exports(
        source,
        [placement],
        {"_block": entity},
        {},
        external_loads={"_used", "_reexport"},
    )
    lines = result.splitlines()
    noqa_lines = [line for line in lines if "# fmt: skip # noqa: F401, E501" in line]
    assert len(noqa_lines) == 2
    names = {line.split("import")[1].split("#")[0].strip() for line in noqa_lines}
    assert names == {"_used", "_reexport"}


def test_add_re_exports_mixed_only_still_loaded_in_external_loads_gets_noqa():
    # When only the used name is in external_loads (not the purely re-exported one),
    # verify external_loads membership drives noqa independently of still_loaded.
    source = "import os\n\n_used()\n"
    entity = _make_entity("_block", 3, 4, ["_used", "_reexport"])
    placement = GroupPlacement(group=["_block"], target_file="utils.py")
    result = _add_re_exports(
        source,
        [placement],
        {"_block": entity},
        {},
        external_loads={"_used"},  # only _used is externally imported
    )
    lines = result.splitlines()
    noqa_lines = [line for line in lines if "# fmt: skip # noqa: F401, E501" in line]
    # _used is in still_loaded AND external_loads → gets noqa
    assert len(noqa_lines) == 1
    assert "_used" in noqa_lines[0]
    # _reexport is not in still_loaded and not in external_loads → not re-exported
    assert "_reexport" not in result


def test_add_re_exports_is_test_file_adds_comment_before_first_noqa():
    # is_test_file=True → single explanatory comment inserted before the first
    # F401 import; non-test files and test files with no noqa imports get no comment.
    source = "import os\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(
        source, [placement], {"foo": entity}, {}, is_test_file=True
    )
    lines = result.splitlines()
    comment_idx = next(
        (
            i
            for i, l in enumerate(lines)
            if "Re-exported for backwards compatibility" in l
        ),
        None,
    )
    noqa_idx = next(
        (i for i, l in enumerate(lines) if "# noqa: F401" in l),
        None,
    )
    assert comment_idx is not None
    assert noqa_idx is not None
    assert comment_idx == noqa_idx - 1


def test_add_re_exports_is_test_file_false_no_comment():
    # is_test_file=False (default) → no explanatory comment added.
    source = "import os\n"
    entity = _make_entity("foo", 1, 2)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(source, [placement], {"foo": entity}, {})
    assert "Re-exported for backwards compatibility" not in result


def test_add_re_exports_is_test_file_no_noqa_imports_no_comment():
    # is_test_file=True but all re-exports are already referenced in source
    # (no noqa imports) → comment is not added.
    source = "import os\n\nfoo()\n"
    entity = _make_entity("foo", 3, 3)
    placement = GroupPlacement(group=["foo"], target_file="utils.py")
    result = _add_re_exports(
        source, [placement], {"foo": entity}, {}, is_test_file=True
    )
    assert "Re-exported for backwards compatibility" not in result


def test_add_re_exports_is_test_file_comment_added_once_for_multiple_noqa():
    # Multiple noqa imports → comment appears exactly once, before the first one.
    source = "import os\n"
    entity = _make_entity("_block", 1, 2, ["foo", "bar"])
    placement = GroupPlacement(group=["_block"], target_file="utils.py")
    result = _add_re_exports(
        source, [placement], {"_block": entity}, {}, is_test_file=True
    )
    comment_count = result.count("Re-exported for backwards compatibility")
    assert comment_count == 1


def test_add_re_exports_is_test_file_comment_before_noqa_when_mixed():
    # is_test_file=True with a mix of used (no noqa) and pure re-export (noqa)
    # imports: the comment must appear before the noqa line, not before the used line.
    source = "import os\n\n_used()\n"
    entity = _make_entity("_block", 3, 4, ["_used", "pub"])
    placement = GroupPlacement(group=["_block"], target_file="utils.py")
    result = _add_re_exports(
        source, [placement], {"_block": entity}, {}, is_test_file=True
    )
    lines = result.splitlines()
    comment_idx = next(
        i for i, l in enumerate(lines) if "Re-exported for backwards" in l
    )
    noqa_idx = next(i for i, l in enumerate(lines) if "# noqa: F401" in l)
    used_idx = next(i for i, l in enumerate(lines) if "import _used" in l)
    assert used_idx < comment_idx
    assert comment_idx == noqa_idx - 1


def test_add_re_exports_relative_from_uses_relative_prefix():
    # When relative_from is set, imports are computed via _relative_import_prefix
    # rather than _target_module_name, so "service/__init__.py" → ".utils"
    # (not ".service.utils").
    source = "# stayed\n"
    entity = _make_entity("Foo", 1, 1)
    placements = [GroupPlacement(group=["Foo"], target_file="service/utils.py")]
    entity_map = {"Foo": entity}
    entity_source_map = {"Foo": "class Foo: pass"}

    result = _add_re_exports(
        source,
        placements,
        entity_map,
        entity_source_map,
        relative_from="service/__init__.py",
    )

    assert "from .utils import Foo" in result
    # Must NOT use the fully-qualified form that would be wrong from __init__.py.
    assert "from .service.utils" not in result
