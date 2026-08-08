"""Tests for file_limiter.code_gen — 100% branch coverage."""

from __future__ import annotations

import textwrap

from crispen.file_limiter.advisor import FileLimiterPlan, GroupPlacement
from crispen.file_limiter.classifier import ClassifiedEntities
from crispen.import_sort import _sort_imports_pep8
from crispen.file_limiter.code_gen import (
    ImportInfo,
    _abs_package_for_dir,
    _add_re_exports,
    _bump_relative_imports,
    _class_has_test_methods,
    _collect_external_imported_names,
    _collect_name_loads,
    _collect_quoted_annotation_names,
    _collect_name_stores,
    _inject_module_level_imports,
    _inject_type_checking_imports,
    _test_names_in_decorators,
    _extract_import_info,
    _extract_module_docstring,
    _extract_shared_helpers,
    _find_cross_file_imports,
    _find_cross_file_type_checking_imports,
    _module_import_stmt,
    _rewrite_module_level_stores,
    _rewrite_module_var_names,
    _find_main_block_entity,
    _find_main_direct_callees,
    _find_needed_imports,
    _find_type_checking_needed_imports,
    _narrow_import_source,
    _find_project_root,
    _import_derived_names,
    _import_line_numbers,
    _inject_inline_imports,
    _inject_inline_test_imports_original,
    _file_has_only_fixtures,
    _is_pytest_fixture,
    _is_test_name,
    _merge_conftest_sources,
    _merge_from_imports,
    _module_path_from_file,
    _prune_inline_redundant_imports,
    _prune_unused_imports,
    _relative_import_prefix,
    _remove_entity_lines,
    _split_cross_imports_by_test,
    _source_is_only_docstring,
    _strip_module_docstring,
    _multiline_string_ranges,
    _normalize_blank_lines,
    _sub_skip_strings,
    _strip_orphaned_indented_comments,
    _strip_orphaned_section_headers,
    _strip_top_level_import_lines,
    _target_module_name,
    _topo_depth,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _collect_name_loads
# ---------------------------------------------------------------------------


def test_collect_name_loads_basic():
    source = "x = foo + bar"
    names = _collect_name_loads(source)
    assert "foo" in names
    assert "bar" in names


def test_collect_name_loads_store_not_included():
    source = "x = 1"
    names = _collect_name_loads(source)
    # x is a Store, not a Load
    assert "x" not in names


def test_collect_name_loads_syntax_error():
    assert _collect_name_loads("def (invalid") == set()


def test_collect_name_loads_excludes_function_params():
    # 'client' is a parameter of test_foo — excluded from loads inside the body.
    source = "def test_foo(client):\n    client.call()\n"
    names = _collect_name_loads(source)
    assert "client" not in names


def test_collect_name_loads_includes_non_param_name():
    # 'helper' is not a parameter of test_foo — still counted as a load.
    source = "def test_foo(client):\n    helper(client)\n"
    names = _collect_name_loads(source)
    assert "helper" in names
    assert "client" not in names


def test_collect_name_loads_excludes_nested_function_params():
    # Inner function params are excluded only within that function's own body.
    source = textwrap.dedent(
        """\
        def outer(x):
            def inner(y):
                return y + x
            return inner
        """
    )
    names = _collect_name_loads(source)
    assert "y" not in names  # param of inner — excluded inside inner body
    assert "x" not in names  # param of outer — excluded inside outer body


def test_collect_name_loads_includes_annotation_names():
    # Type annotations are in the outer scope — their names are included.
    source = "def f(x: MyType) -> ReturnType:\n    pass\n"
    names = _collect_name_loads(source)
    assert "MyType" in names
    assert "ReturnType" in names
    assert "x" not in names  # param name itself, not counted


def test_collect_name_loads_includes_decorator_names():
    # Decorator expressions are in the outer scope.
    source = "@pytest.fixture\ndef client():\n    pass\n"
    names = _collect_name_loads(source)
    assert "pytest" in names


def test_collect_name_loads_kw_defaults_none_skipped():
    # kw_defaults may contain None for keyword-only args without defaults.
    # None entries must not cause a crash and are simply skipped.
    source = "def f(*, a, b=DEFAULT):\n    pass\n"
    names = _collect_name_loads(source)
    assert "DEFAULT" in names
    assert "a" not in names
    assert "b" not in names


def test_collect_name_loads_annotated_vararg_kwarg():
    # *args: T and **kwargs: T annotations are in the outer scope.
    source = "def f(*args: VarType, **kwargs: KwType):\n    pass\n"
    names = _collect_name_loads(source)
    assert "VarType" in names
    assert "KwType" in names


def test_collect_name_loads_excludes_local_variable_assignments():
    # A name assigned in the function body is a local variable — not an import.
    # Loads of that name (e.g. attribute access) must not generate cross-file imports.
    source = textwrap.dedent(
        """\
        def test_foo(tmp_path):
            helpers = tmp_path / "helpers.py"
            helpers.write_text("X = 1", encoding="utf-8")
            assert str(helpers.resolve()) == "x"
        """
    )
    names = _collect_name_loads(source)
    assert "helpers" not in names
    assert "tmp_path" not in names  # also a param — still excluded


def test_collect_name_loads_local_store_does_not_suppress_outer_loads():
    # A local assignment in an inner function must not suppress the outer scope's load.
    source = textwrap.dedent(
        """\
        def outer():
            use(helper)
            def inner():
                helper = 1
                use(helper)
        """
    )
    names = _collect_name_loads(source)
    # outer() loads 'helper' (not locally defined there); inner() assigns it locally.
    assert "helper" in names


# ---------------------------------------------------------------------------
# _collect_quoted_annotation_names
# ---------------------------------------------------------------------------


def test_collect_quoted_annotation_names_basic():
    # "MyType" in a string annotation → detected.
    source = 'def f(x: "MyType") -> None:\n    pass\n'
    names = _collect_quoted_annotation_names(source)
    assert "MyType" in names


def test_collect_quoted_annotation_names_optional():
    # Optional["_LLMAccumulator"] — the inner string is parsed.
    source = 'def f(x: Optional["_LLMAccumulator"]) -> None:\n    pass\n'
    names = _collect_quoted_annotation_names(source)
    assert "_LLMAccumulator" in names


def test_collect_quoted_annotation_names_return():
    # Quoted return annotation.
    source = 'def f() -> "ReturnType":\n    pass\n'
    names = _collect_quoted_annotation_names(source)
    assert "ReturnType" in names


def test_collect_quoted_annotation_names_annassign():
    # Variable annotation: x: "MyClass"
    source = 'x: "MyClass"\n'
    names = _collect_quoted_annotation_names(source)
    assert "MyClass" in names


def test_collect_quoted_annotation_names_unquoted_not_included():
    # Normal (unquoted) annotation names are NOT returned by this function.
    source = "def f(x: MyType) -> None:\n    pass\n"
    names = _collect_quoted_annotation_names(source)
    assert "MyType" not in names


def test_collect_quoted_annotation_names_syntax_error():
    # Unparseable source returns empty set (no crash).
    assert _collect_quoted_annotation_names("def (invalid") == set()


def test_collect_quoted_annotation_names_inner_syntax_error():
    # A string annotation that isn't valid Python is silently ignored.
    source = 'def f(x: "not valid python !!") -> None:\n    pass\n'
    names = _collect_quoted_annotation_names(source)
    assert names == set()


def test_collect_quoted_annotation_names_vararg_kwarg():
    # *args and **kwargs with quoted annotations.
    source = 'def f(*args: "VarType", **kwargs: "KwType") -> None:\n    pass\n'
    names = _collect_quoted_annotation_names(source)
    assert "VarType" in names
    assert "KwType" in names


def test_collect_quoted_annotation_names_annassign_with_value():
    # x: "MyClass" = SomeFactory() — annotation has quoted name AND there is a value.
    # The _walk branch for AnnAssign with node.value must execute.
    source = 'x: "MyClass" = object()\n'
    names = _collect_quoted_annotation_names(source)
    assert "MyClass" in names


# ---------------------------------------------------------------------------
# _collect_name_stores
# ---------------------------------------------------------------------------


def test_collect_name_stores_simple_assign():
    assert _collect_name_stores("X = 1\n") == {"X"}


def test_collect_name_stores_multiple_assigns():
    src = "X = 1\nY = 2\n"
    assert _collect_name_stores(src) == {"X", "Y"}


def test_collect_name_stores_augassign():
    assert _collect_name_stores("X += 1\n") == {"X"}


def test_collect_name_stores_annotated_assign_with_value():
    assert _collect_name_stores("X: int = 42\n") == {"X"}


def test_collect_name_stores_annotated_assign_without_value():
    # Declaration only (no assignment) — not a store.
    assert _collect_name_stores("X: int\n") == set()


def test_collect_name_stores_function_body_not_included():
    # Assignments inside function bodies are not module-level stores.
    src = "def f():\n    X = 1\n"
    assert _collect_name_stores(src) == set()


def test_collect_name_stores_load_not_included():
    assert _collect_name_stores("y = X\n") == {"y"}
    assert "X" not in _collect_name_stores("y = X\n")


def test_collect_name_stores_syntax_error():
    assert _collect_name_stores("def (broken:\n") == set()


def test_collect_name_stores_empty():
    assert _collect_name_stores("") == set()


def test_collect_name_stores_non_name_assign_target():
    # Tuple-unpacking targets are not plain Name nodes — must not crash.
    src = "a, b = 1, 2\n"
    result = _collect_name_stores(src)
    assert "a" not in result  # tuple target, not a plain Name store
    assert "b" not in result


def test_collect_name_stores_non_name_augassign_target():
    # Attribute augmented assignment — target is Attribute, not Name.
    src = "obj.x += 1\n"
    result = _collect_name_stores(src)
    assert result == set()


# ---------------------------------------------------------------------------
# _inject_module_level_imports
# ---------------------------------------------------------------------------


def test_inject_module_level_imports_docstring_only():
    # Source with only a docstring and no imports — insert after the docstring.
    src = '"""Module doc."""\n\nx = 1\n'
    result = _inject_module_level_imports(src, ["from . import converters"])
    assert '"""Module doc."""' in result
    assert "from . import converters" in result
    doc_pos = result.index('"""Module doc."""')
    imp_pos = result.index("from . import converters")
    assert doc_pos < imp_pos


def test_inject_module_level_imports_empty_list():
    src = "x = 1\n"
    assert _inject_module_level_imports(src, []) == src


def test_inject_module_level_imports_after_imports():
    src = "import os\n\nx = 1\n"
    result = _inject_module_level_imports(src, ["from . import converters"])
    assert result == "import os\nfrom . import converters\n\nx = 1\n"


def test_inject_module_level_imports_no_existing_imports():
    src = "x = 1\n"
    result = _inject_module_level_imports(src, ["from . import converters"])
    # Prepended before non-import content
    assert "from . import converters" in result
    assert result.index("from . import converters") < result.index("x = 1")


def test_inject_module_level_imports_sorted():
    src = "import os\n\nx = 1\n"
    result = _inject_module_level_imports(
        src, ["from . import z_mod", "from . import a_mod"]
    )
    lines = result.splitlines()
    import_lines = [ln for ln in lines if "import" in ln]
    assert import_lines.index("from . import a_mod") < import_lines.index(
        "from . import z_mod"
    )


def test_inject_module_level_imports_syntax_error_prepends():
    src = "def (broken:\n"
    result = _inject_module_level_imports(src, ["import os"])
    assert result.startswith("import os\n")


# ---------------------------------------------------------------------------
# _inject_type_checking_imports
# ---------------------------------------------------------------------------


def test_inject_type_checking_imports_empty_list():
    src = "import os\n"
    assert _inject_type_checking_imports(src, []) == src


def test_inject_type_checking_imports_syntax_error():
    src = "def (broken:\n"
    assert _inject_type_checking_imports(src, ["from .config import Cfg"]) == src


def test_inject_type_checking_imports_all_already_present():
    # If every requested import is already in an existing TC block, no change.
    src = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .config import Cfg\n"
        "\n"
        "x = 1\n"
    )
    result = _inject_type_checking_imports(src, ["from .config import Cfg"])
    assert result == src


def test_inject_type_checking_imports_appends_to_existing_block():
    # New import should be appended inside the existing TYPE_CHECKING block.
    src = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .config import Cfg\n"
        "\n"
        "x = 1\n"
    )
    result = _inject_type_checking_imports(src, ["from .models import MyModel"])
    assert "from .models import MyModel" in result
    tc_start = result.index("if TYPE_CHECKING:")
    assert result.index("from .models import MyModel") > tc_start
    assert "x = 1" in result


def test_inject_type_checking_imports_creates_block_with_typing_import():
    # No existing TC block and TYPE_CHECKING not imported → add both.
    src = "from typing import List\n\ndef foo(x: 'Cfg') -> None:\n    pass\n"
    result = _inject_type_checking_imports(src, ["from .config import Cfg"])
    assert "from typing import TYPE_CHECKING" in result
    assert "if TYPE_CHECKING:" in result
    assert "    from .config import Cfg" in result


def test_inject_type_checking_imports_creates_block_type_checking_already_imported():
    # TYPE_CHECKING already in typing import → don't add it again.
    src = (
        "from typing import List, TYPE_CHECKING\n"
        "\n"
        "def foo(x: 'Cfg') -> None:\n"
        "    pass\n"
    )
    result = _inject_type_checking_imports(src, ["from .config import Cfg"])
    assert result.count("TYPE_CHECKING") == 2  # one in import, one in if-block
    assert "if TYPE_CHECKING:" in result
    assert "    from .config import Cfg" in result


def test_inject_type_checking_imports_block_after_last_import():
    # The new block should appear after the last import, before other code.
    src = "import os\nimport sys\n\nx = 1\n"
    result = _inject_type_checking_imports(src, ["from .config import Cfg"])
    lines = result.splitlines()
    sys_line = next(i for i, l in enumerate(lines) if "import sys" in l)
    if_line = next(i for i, l in enumerate(lines) if "if TYPE_CHECKING" in l)
    x_line = next(i for i, l in enumerate(lines) if "x = 1" in l)
    assert sys_line < if_line < x_line


# ---------------------------------------------------------------------------
# _test_names_in_decorators
# ---------------------------------------------------------------------------


def test_test_names_in_decorators_finds_name_in_decorator():
    src = (
        "@pytest.mark.parametrize('x', TestFixture.PARAMS)\ndef test_fn(x):\n    pass\n"
    )
    assert _test_names_in_decorators(src, {"TestFixture"}) == {"TestFixture"}


def test_test_names_in_decorators_name_only_in_body_not_found():
    src = "def test_fn():\n    TestFixture.setup()\n"
    assert _test_names_in_decorators(src, {"TestFixture"}) == set()


def test_test_names_in_decorators_syntax_error_returns_empty():
    assert _test_names_in_decorators("def (invalid", {"TestFixture"}) == set()


def test_test_names_in_decorators_class_decorator():
    src = "@TestFixture.mark\nclass TestSomething:\n    pass\n"
    assert _test_names_in_decorators(src, {"TestFixture"}) == {"TestFixture"}


# ---------------------------------------------------------------------------
# _extract_import_info
# ---------------------------------------------------------------------------


def test_extract_import_info_syntax_error():
    assert _extract_import_info("def (invalid") == []


def test_extract_import_info_plain_import():
    infos = _extract_import_info("import os\n")
    assert len(infos) == 1
    assert "os" in infos[0].names
    assert infos[0].is_future is False


def test_extract_import_info_import_with_asname():
    infos = _extract_import_info("import os as operating_system\n")
    assert infos[0].names == ["operating_system"]


def test_extract_import_info_dotted_import():
    infos = _extract_import_info("import os.path\n")
    assert infos[0].names == ["os"]


def test_extract_import_info_from_import():
    infos = _extract_import_info("from pathlib import Path\n")
    assert "Path" in infos[0].names
    assert infos[0].is_future is False


def test_extract_import_info_from_import_with_asname():
    infos = _extract_import_info("from pathlib import Path as P\n")
    assert infos[0].names == ["P"]


def test_extract_import_info_future_import():
    infos = _extract_import_info("from __future__ import annotations\n")
    assert infos[0].is_future is True
    assert "annotations" in infos[0].names


def test_extract_import_info_skips_non_imports():
    infos = _extract_import_info("def foo():\n    pass\n")
    assert infos == []


def test_extract_import_info_multiple():
    source = "import os\nfrom pathlib import Path\n"
    infos = _extract_import_info(source)
    assert len(infos) == 2


def test_extract_import_info_multiline_parens_normalized():
    # Multi-line parenthesized from-import must be normalized to a single line
    # so that _merge_from_imports can process it without producing malformed output.
    source = "from pathlib import (\n    Path,\n    PurePath,\n)\n"
    infos = _extract_import_info(source)
    assert len(infos) == 1
    assert infos[0].source == "from pathlib import Path, PurePath"
    assert "\n" not in infos[0].source
    assert "Path" in infos[0].names
    assert "PurePath" in infos[0].names


def test_extract_import_info_type_checking_from_import():
    # Imports inside `if TYPE_CHECKING:` are extracted with is_type_checking=True.
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .config import MyConfig\n"
    )
    infos = _extract_import_info(source)
    tc = [i for i in infos if i.is_type_checking]
    assert len(tc) == 1
    assert "MyConfig" in tc[0].names
    assert tc[0].source == "from .config import MyConfig"
    assert tc[0].is_future is False


def test_extract_import_info_type_checking_plain_import():
    # Plain `import` inside `if TYPE_CHECKING:` is also captured.
    source = "if TYPE_CHECKING:\n    import sys\n"
    infos = _extract_import_info(source)
    tc = [i for i in infos if i.is_type_checking]
    assert len(tc) == 1
    assert "sys" in tc[0].names
    assert tc[0].is_type_checking is True


def test_extract_import_info_type_checking_not_is_future():
    # TYPE_CHECKING block imports must not be marked as is_future.
    source = "if TYPE_CHECKING:\n    from .foo import Bar\n"
    infos = _extract_import_info(source)
    tc = [i for i in infos if i.is_type_checking]
    assert all(not i.is_future for i in tc)


def test_extract_import_info_type_checking_skips_non_import_children():
    # Non-import statements inside a TYPE_CHECKING block (rare but valid)
    # must not cause errors and must be silently skipped.
    source = "if TYPE_CHECKING:\n    from .foo import Bar\n    x = 1\n"
    infos = _extract_import_info(source)
    tc = [i for i in infos if i.is_type_checking]
    assert len(tc) == 1
    assert "Bar" in tc[0].names


# ---------------------------------------------------------------------------
# _find_needed_imports
# ---------------------------------------------------------------------------


def test_find_needed_imports_referenced_name():
    # Entity references "os"; import for "os" should be included.
    entity_src_map = {"foo": "def foo():\n    os.getcwd()\n"}
    infos = [ImportInfo(names=["os"], source="import os", is_future=False)]
    result = _find_needed_imports(["foo"], entity_src_map, infos, {"foo"})
    assert "import os" in result


def test_find_needed_imports_unreferenced_name():
    # Entity doesn't reference "sys"; import should be excluded.
    entity_src_map = {"foo": "def foo():\n    pass\n"}
    infos = [ImportInfo(names=["sys"], source="import sys", is_future=False)]
    result = _find_needed_imports(["foo"], entity_src_map, infos, {"foo"})
    assert result == []


def test_find_needed_imports_future_always_included():
    # __future__ import is always included regardless of entity references.
    entity_src_map = {"foo": "def foo():\n    pass\n"}
    infos = [
        ImportInfo(
            names=["annotations"],
            source="from __future__ import annotations",
            is_future=True,
        )
    ]
    result = _find_needed_imports(["foo"], entity_src_map, infos, {"foo"})
    assert "from __future__ import annotations" in result


def test_find_needed_imports_deduplicates():
    # Two ImportInfo entries with the same source string → only one included.
    entity_src_map = {"foo": "def foo():\n    os.getcwd()\n"}
    infos = [
        ImportInfo(names=["os"], source="import os", is_future=False),
        ImportInfo(names=["os"], source="import os", is_future=False),  # duplicate
    ]
    result = _find_needed_imports(["foo"], entity_src_map, infos, {"foo"})
    assert result.count("import os") == 1


def test_find_needed_imports_entity_not_in_map():
    # Entity name not in entity_source_map → treated as empty source.
    infos = [ImportInfo(names=["os"], source="import os", is_future=False)]
    result = _find_needed_imports(["ghost"], {}, infos, set())
    assert result == []


def test_find_needed_imports_skips_type_checking():
    # is_type_checking imports must not appear as regular imports.
    entity_src_map = {"foo": 'def foo(x: "MyConfig") -> None:\n    pass\n'}
    infos = [
        ImportInfo(
            names=["MyConfig"],
            source="from .config import MyConfig",
            is_future=False,
            is_type_checking=True,
        )
    ]
    result = _find_needed_imports(["foo"], entity_src_map, infos, {"foo"})
    assert result == []


# ---------------------------------------------------------------------------
# _find_type_checking_needed_imports
# ---------------------------------------------------------------------------


def test_find_type_checking_needed_imports_quoted_only():
    # "MyType" appears only in a quoted annotation, not a runtime load.
    entity_src_map = {"foo": 'def foo(x: Optional["MyType"]) -> None:\n    pass\n'}
    infos = [
        ImportInfo(
            names=["MyType"], source="from models import MyType", is_future=False
        )
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert "from models import MyType" in result


def test_find_type_checking_needed_imports_runtime_excluded():
    # When the name is used at runtime (not just in a quoted annotation),
    # it should NOT appear in the TYPE_CHECKING-only list.
    # annotation_only = quoted - runtime excludes runtime names directly.
    entity_src_map = {"foo": "def foo():\n    return MyType()\n"}
    infos = [
        ImportInfo(
            names=["MyType"], source="from models import MyType", is_future=False
        )
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert result == []


def test_find_type_checking_needed_imports_no_annotations():
    # No quoted annotations → result is empty.
    entity_src_map = {"foo": "def foo():\n    pass\n"}
    infos = [
        ImportInfo(
            names=["MyType"], source="from models import MyType", is_future=False
        )
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert result == []


def test_find_type_checking_needed_imports_future_excluded():
    # __future__ imports are never returned (they're always in regular imports).
    entity_src_map = {"foo": 'def foo(x: "MyType") -> None:\n    pass\n'}
    infos = [
        ImportInfo(
            names=["annotations"],
            source="from __future__ import annotations",
            is_future=True,
        ),
        ImportInfo(
            names=["MyType"], source="from models import MyType", is_future=False
        ),
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert "from __future__ import annotations" not in result
    assert "from models import MyType" in result


def test_find_type_checking_needed_imports_deduplicates():
    # Two ImportInfo entries with the same source → only one returned.
    entity_src_map = {"foo": 'def foo(x: "MyType") -> None:\n    pass\n'}
    infos = [
        ImportInfo(
            names=["MyType"], source="from models import MyType", is_future=False
        ),
        ImportInfo(
            names=["MyType"], source="from models import MyType", is_future=False
        ),
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert result.count("from models import MyType") == 1


def test_find_type_checking_needed_imports_import_names_no_match():
    # annotation_only has "MyType" but the ImportInfo names do not include it →
    # the tc_names check returns False → import is skipped.
    entity_src_map = {"foo": 'def foo(x: "MyType") -> None:\n    pass\n'}
    infos = [
        ImportInfo(
            names=["OtherType"], source="from models import OtherType", is_future=False
        )
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert result == []


def test_find_type_checking_needed_imports_partial_multi_name_import():
    # From a multi-name import, only the annotation-only name should appear in
    # the TYPE_CHECKING block; the other name (not referenced at all) must not.
    entity_src_map = {
        "foo": 'def foo(x: "MyResult") -> None:\n    pass\n',
    }
    infos = [
        ImportInfo(
            names=["MyResult", "run_thing"],
            source="from mymod import MyResult, run_thing",
            is_future=False,
        )
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert len(result) == 1
    assert "MyResult" in result[0]
    assert "run_thing" not in result[0]


def test_find_type_checking_needed_imports_narrowed_src_dedup():
    # When two ImportInfo entries produce the same narrowed source after
    # filtering, only one copy should appear in the result (line 535 branch).
    entity_src_map = {"foo": 'def foo(x: "MyResult") -> None:\n    pass\n'}
    infos = [
        ImportInfo(
            names=["MyResult", "run_thing"],
            source="from mymod import MyResult, run_thing",
            is_future=False,
        ),
        # A second entry with the same source (e.g. two entities requested it).
        ImportInfo(
            names=["MyResult", "run_thing"],
            source="from mymod import MyResult, run_thing",
            is_future=False,
        ),
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert result.count("from mymod import MyResult") == 1


def test_find_type_checking_needed_imports_shared_import_with_runtime_peer():
    # Regression: when an import line covers both a runtime name and an
    # annotation-only name, the annotation-only name must still get a
    # TYPE_CHECKING import even though the import source appears in the
    # regular imports (where _prune_unused_imports will later drop it).
    entity_src_map = {
        "foo": (
            'def foo(_acc: Optional["_LLMAccumulator"] = None) -> None:\n'
            "    call_with_tool(_PLACEMENT_TOOL)\n"
        )
    }
    infos = [
        ImportInfo(
            names=["_LLMAccumulator", "_PLACEMENT_TOOL"],
            source="from .llm_schemas import _LLMAccumulator, _PLACEMENT_TOOL",
            is_future=False,
        )
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    # _LLMAccumulator is only in a quoted annotation → must be in TC block
    assert any("_LLMAccumulator" in r for r in result)
    # _PLACEMENT_TOOL is a runtime reference → must NOT be in TC block
    assert not any("_PLACEMENT_TOOL" in r for r in result)


def test_find_type_checking_needed_imports_uses_is_type_checking_infos():
    # is_type_checking=True ImportInfo entries are used for TC distribution;
    # the function should return them for entities that use the name in a
    # quoted annotation.
    entity_src_map = {"foo": 'def foo(config: "MyConfig") -> None:\n    pass\n'}
    infos = [
        ImportInfo(
            names=["MyConfig"],
            source="from .config import MyConfig",
            is_future=False,
            is_type_checking=True,
        )
    ]
    result = _find_type_checking_needed_imports(["foo"], entity_src_map, infos)
    assert "from .config import MyConfig" in result


# ---------------------------------------------------------------------------
# _narrow_import_source
# ---------------------------------------------------------------------------


def test_narrow_import_source_syntax_error():
    # Invalid Python → original string returned unchanged.
    bad = "from ??? import Foo"
    assert _narrow_import_source(bad, {"Foo"}) == bad


def test_narrow_import_source_plain_import():
    # Non-ImportFrom statement (bare `import X`) → returned unchanged.
    src = "import os"
    assert _narrow_import_source(src, {"os"}) == src


def test_narrow_import_source_empty_keep():
    # keep_names matches nothing → alias_strs is empty → return original.
    src = "from mymod import A, B"
    assert _narrow_import_source(src, {"C"}) == src


# ---------------------------------------------------------------------------
# _target_module_name
# ---------------------------------------------------------------------------


def test_target_module_name_simple():
    assert _target_module_name("utils.py") == "utils"


def test_target_module_name_nested():
    assert _target_module_name("helpers/io.py") == "helpers.io"


def test_target_module_name_init():
    # __init__.py represents the package, not a "__init__" submodule.
    assert _target_module_name("pkg/__init__.py") == "pkg"


# ---------------------------------------------------------------------------
# _remove_entity_lines
# ---------------------------------------------------------------------------


def test_remove_entity_lines_removes_range():
    source = "line1\nline2\nline3\nline4\n"
    entity = _make_entity("foo", 2, 3)
    entity_map = {"foo": entity}
    result = _remove_entity_lines(source, {"foo"}, entity_map, {})
    assert "line1" in result
    assert "line2" not in result
    assert "line3" not in result
    assert "line4" in result


def test_remove_entity_lines_name_not_in_map():
    # Name not in entity_map → nothing removed.
    source = "line1\nline2\n"
    result = _remove_entity_lines(source, {"ghost"}, {}, {})
    assert result == source


def test_remove_entity_lines_top_level_preserves_import_lines():
    # When a TOP_LEVEL entity containing both imports and assignments is
    # migrated, the import lines must be kept in the original file so that
    # the remaining functions still have access to those names.
    source = "import os\n_CONST = 1\n\ndef foo():\n    return os.getcwd()\n"
    entity_src = "import os\n_CONST = 1\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 2, ["os", "_CONST"])
    entity_map = {"_block_1": entity}
    entity_source_map = {"_block_1": entity_src}
    result = _remove_entity_lines(source, {"_block_1"}, entity_map, entity_source_map)
    assert "import os" in result  # import line preserved
    assert "_CONST" not in result  # assignment line removed
    assert "def foo():" in result  # function untouched


def test_remove_entity_lines_top_level_no_source_map_removes_all():
    # Empty entity_source_map → no imports can be identified, all lines removed.
    source = "import os\n_CONST = 1\n\ndef foo():\n    pass\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 2, ["os", "_CONST"])
    entity_map = {"_block_1": entity}
    result = _remove_entity_lines(source, {"_block_1"}, entity_map, {})
    assert "import os" not in result
    assert "_CONST" not in result


# ---------------------------------------------------------------------------
# _import_derived_names
# ---------------------------------------------------------------------------


def test_import_derived_names_plain_import():
    src = "import os\nimport sys\n"
    assert _import_derived_names(src) == {"os", "sys"}


def test_import_derived_names_from_import():
    src = "from typing import Dict, List\n"
    assert _import_derived_names(src) == {"Dict", "List"}


def test_import_derived_names_aliased():
    src = "import libcst as cst\nfrom dataclasses import dataclass\n"
    assert _import_derived_names(src) == {"cst", "dataclass"}


def test_import_derived_names_ignores_assignments():
    src = "_MODEL = 'x'\n_MIN = 3\n"
    assert _import_derived_names(src) == set()


def test_import_derived_names_syntax_error():
    assert _import_derived_names("def (\n") == set()


# ---------------------------------------------------------------------------
# _import_line_numbers
# ---------------------------------------------------------------------------


def test_import_line_numbers_basic():
    src = "import os\n_CONST = 1\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 5, 6, [])
    # Entity starts at line 5; "import os" is relative line 1 → absolute line 5.
    result = _import_line_numbers(entity, src)
    assert result == {5}


def test_import_line_numbers_no_imports():
    src = "_CONST = 1\n_OTHER = 2\n"
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 2, [])
    assert _import_line_numbers(entity, src) == set()


def test_import_line_numbers_syntax_error():
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, [])
    assert _import_line_numbers(entity, "def (\n") == set()


# ---------------------------------------------------------------------------
# _add_re_exports — import-derived name filtering
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _add_re_exports
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# generate_file_splits
# ---------------------------------------------------------------------------


def test_generate_abort_plan():
    plan = _abort_plan()
    c = _classified()
    result = generate_file_splits(c, plan, "def foo():\n    pass\n", "big.py")
    assert result.abort is True
    assert result.new_files == {}
    assert result.original_source == "def foo():\n    pass\n"


def test_generate_empty_placements():
    plan = _plan()  # placements=[]
    c = _classified()
    source = "def foo():\n    pass\n"
    result = generate_file_splits(c, plan, source, "big.py")
    assert result.abort is False
    assert result.new_files == {}
    assert result.original_source == source


def test_generate_single_entity_migration():
    source = "import os\n\ndef foo():\n    os.getcwd()\n"
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert result.abort is False
    assert "utils.py" in result.new_files
    new_src = result.new_files["utils.py"]
    assert "import os" in new_src
    assert "def foo():" in new_src
    # Original should not have foo's def anymore
    assert "def foo():" not in result.original_source
    # But should have a re-export
    assert "from .utils import foo" in result.original_source


def test_generate_private_entity_no_reexport():
    source = "def _helper():\n    pass\n"
    entity = _make_entity("_helper", 1, 2)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["_helper"], target_file="private.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "from .private import" not in result.original_source


def test_generate_entity_not_in_source_map():
    # Group has entity name not in classified.entities → entity skipped in new file.
    source = "def foo():\n    pass\n"
    entity = _make_entity("foo", 1, 2)
    c = _classified(entities=[entity])
    # "ghost" is in the group but has no matching entity
    plan = _plan([GroupPlacement(group=["foo", "ghost"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "utils.py" in result.new_files
    # "ghost" produces no source so only "foo" appears
    new_src = result.new_files["utils.py"]
    assert "def foo():" in new_src


def test_generate_no_imports_needed():
    # Entity uses no imports → no import section in new file.
    source = "def add(a, b):\n    return a + b\n"
    entity = _make_entity("add", 1, 2)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["add"], target_file="math_utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["math_utils.py"]
    # No "import" prefix expected
    assert not new_src.startswith("import")
    assert "def add" in new_src


def test_generate_multiple_groups_same_file():
    source = textwrap.dedent(
        """\
        import os

        def foo():
            pass

        def bar():
            pass
        """
    )
    e_foo = _make_entity("foo", 3, 4)
    e_bar = _make_entity("bar", 6, 7)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan(
        [
            GroupPlacement(group=["foo"], target_file="utils.py"),
            GroupPlacement(group=["bar"], target_file="utils.py"),
        ]
    )
    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert "def foo():" in new_src
    assert "def bar():" in new_src


def test_generate_multiple_different_target_files():
    source = "def foo():\n    pass\n\ndef bar():\n    pass\n"
    e_foo = _make_entity("foo", 1, 2)
    e_bar = _make_entity("bar", 4, 5)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan(
        [
            GroupPlacement(group=["foo"], target_file="foo_module.py"),
            GroupPlacement(group=["bar"], target_file="bar_module.py"),
        ]
    )
    result = generate_file_splits(c, plan, source, "big.py")

    assert "foo_module.py" in result.new_files
    assert "bar_module.py" in result.new_files
    assert "def foo():" in result.new_files["foo_module.py"]
    assert "def bar():" in result.new_files["bar_module.py"]
    assert "from .bar_module import bar" in result.original_source
    assert "from .foo_module import foo" in result.original_source


def test_generate_future_import_not_duplicated_when_in_entity_source():
    # Entity source itself contains `from __future__ import annotations`
    # (e.g. the _block_1 TOP_LEVEL entity which IS the file's import block).
    # It must appear only once at the top of the new file, not again inside
    # the entity source, which would cause a SyntaxError.
    source = textwrap.dedent(
        """\
        from __future__ import annotations

        \"\"\"Module docstring.\"\"\"

        from __future__ import annotations

        import os

        _CONST = 42
    """
    )
    # _block_1 spans the whole file and contains the future import + constants.
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 9, ["_CONST"])
    c = _classified(entities=[e_block])
    plan = _plan([GroupPlacement(group=["_block_1"], target_file="constants.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["constants.py"]
    assert new_src.count("from __future__ import annotations") == 1
    # Must be at the very start of the file (before any other code).
    first_non_blank = next(line for line in new_src.splitlines() if line.strip())
    assert first_non_blank == "from __future__ import annotations"


def test_generate_future_import_always_included():
    source = "from __future__ import annotations\n\ndef foo():\n    pass\n"
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert "from __future__ import annotations" in new_src


# ---------------------------------------------------------------------------
# _find_cross_file_imports
# ---------------------------------------------------------------------------


def test_find_cross_file_imports_basic():
    # fn_a references _MODEL which is defined in block_1.py
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "llm_extract.py"
    )
    assert from_imports == ["from .block_1 import _MODEL"]
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_imports_same_file_excluded():
    # _MODEL goes to the same file as fn_a → no cross-file import needed
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "llm_extract.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "llm_extract.py"
    )
    assert from_imports == []
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_imports_no_match():
    # Referenced name not in name_to_target_file → no cross-file import
    entity_source_map = {"fn_a": "def fn_a():\n    return os.getcwd()\n"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, {}, "utils.py"
    )
    assert from_imports == []
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_imports_entity_not_in_map():
    # Entity name not in entity_source_map → treated as empty source, no imports
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["ghost"], {}, {"x": "other.py"}, "utils.py"
    )
    assert from_imports == []
    assert module_imports == []
    assert rewrites == {}


# ---------------------------------------------------------------------------
# _relative_import_prefix
# ---------------------------------------------------------------------------


def test_relative_import_prefix_same_directory():
    # Both files at the root level → single dot.
    assert _relative_import_prefix("a.py", "b.py") == ".b"


def test_relative_import_prefix_sibling_subdir():
    # from_file is in sub/, to_file is in helpers/ → go up one, then down.
    assert _relative_import_prefix("sub/a.py", "helpers/b.py") == "..helpers.b"


def test_relative_import_prefix_same_subdir():
    # Both in the same subdirectory → single dot.
    assert _relative_import_prefix("sub/a.py", "sub/b.py") == ".b"


def test_relative_import_prefix_to_nested():
    # to_file is in a subdirectory of root while from_file is at root.
    assert _relative_import_prefix("a.py", "helpers/b.py") == ".helpers.b"


def test_relative_import_prefix_to_init_same_dir():
    # to_file is __init__.py in the same directory → "." (the package itself).
    assert _relative_import_prefix("a.py", "__init__.py") == "."


def test_relative_import_prefix_to_init_same_subdir():
    # Both in sub/, to_file is sub/__init__.py → "." (the package itself).
    assert _relative_import_prefix("sub/a.py", "sub/__init__.py") == "."


def test_find_cross_file_imports_cross_directory():
    # fn_a is in tests/test.py; helper is in helpers/entities.py.
    # Cross-directory import needs ".." to go up from tests/ to root.
    entity_source_map = {"fn_a": "def fn_a():\n    return _helper()\n"}
    name_to_target_file = {"_helper": "helpers/entities.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "tests/test.py"
    )
    assert from_imports == ["from ..helpers.entities import _helper"]
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_imports_top_level_var_uses_module_import():
    # SAFE_MODE is a TOP_LEVEL variable in conversion.py; runtime.py references it.
    # Should produce a module-level import (from . import conversion) in
    # module_imports, not a direct name import, so that later mutations to the
    # variable propagate correctly.
    entity_source_map = {
        "create_lua_runtime": (
            "def create_lua_runtime(safe_mode=None):\n"
            "    if safe_mode is None:\n"
            "        safe_mode = SAFE_MODE\n"
        )
    }
    name_to_target_file = {"SAFE_MODE": "conversion.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["create_lua_runtime"],
        entity_source_map,
        name_to_target_file,
        "runtime.py",
        top_level_var_names={"SAFE_MODE"},
    )
    assert from_imports == []
    assert module_imports == ["from . import conversion"]
    assert rewrites == {"SAFE_MODE": "conversion.SAFE_MODE"}


def test_find_cross_file_imports_top_level_var_abs_pkg():
    # Same as above but with abs_pkg set (test-file context).
    # Uses "import pkg.module as local" syntax to avoid test-name misclassification.
    entity_source_map = {"fn_a": "def fn_a():\n    return SAFE_MODE\n"}
    name_to_target_file = {"SAFE_MODE": "conversion.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "test_fn.py",
        abs_pkg="mylib",
        top_level_var_names={"SAFE_MODE"},
    )
    assert from_imports == []
    assert module_imports == ["import mylib.conversion as conversion"]
    assert rewrites == {"SAFE_MODE": "conversion.SAFE_MODE"}


def test_find_cross_file_imports_top_level_var_abs_pkg_empty():
    # abs_pkg="" (root-level test) — no package prefix, plain "import conversion".
    entity_source_map = {"fn_a": "def fn_a():\n    return SAFE_MODE\n"}
    name_to_target_file = {"SAFE_MODE": "conversion.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "test_fn.py",
        abs_pkg="",
        top_level_var_names={"SAFE_MODE"},
    )
    assert from_imports == []
    assert module_imports == ["import conversion"]
    assert rewrites == {"SAFE_MODE": "conversion.SAFE_MODE"}


def test_find_cross_file_imports_top_level_var_cross_directory():
    # TOP_LEVEL var in sub/constants.py, referenced from runtime.py at root.
    entity_source_map = {"fn_a": "def fn_a():\n    return TIMEOUT\n"}
    name_to_target_file = {"TIMEOUT": "sub/constants.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "runtime.py",
        top_level_var_names={"TIMEOUT"},
    )
    assert from_imports == []
    assert module_imports == ["from .sub import constants"]
    assert rewrites == {"TIMEOUT": "constants.TIMEOUT"}


def test_find_cross_file_imports_mixed_top_level_and_function():
    # SAFE_MODE is a TOP_LEVEL var; _helper is a function — mixed case.
    entity_source_map = {
        "fn_a": ("def fn_a():\n" "    if SAFE_MODE:\n" "        return _helper()\n")
    }
    name_to_target_file = {"SAFE_MODE": "conversion.py", "_helper": "helpers.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "runtime.py",
        top_level_var_names={"SAFE_MODE"},
    )
    assert from_imports == ["from .helpers import _helper"]
    assert module_imports == ["from . import conversion"]
    assert rewrites == {"SAFE_MODE": "conversion.SAFE_MODE"}


# ---------------------------------------------------------------------------
# _module_import_stmt
# ---------------------------------------------------------------------------


def test_module_import_stmt_sibling_relative():
    stmt, local = _module_import_stmt("runtime.py", "conversion.py", abs_pkg=None)
    assert stmt == "from . import conversion"
    assert local == "conversion"


def test_module_import_stmt_cross_directory_relative():
    stmt, local = _module_import_stmt("runtime.py", "sub/constants.py", abs_pkg=None)
    assert stmt == "from .sub import constants"
    assert local == "constants"


def test_module_import_stmt_parent_directory_relative():
    # svc/test_fns.py importing from test_svc.py (parent dir)
    stmt, local = _module_import_stmt("svc/test_fns.py", "test_svc.py", abs_pkg=None)
    assert stmt == "from .. import test_svc"
    assert local == "test_svc"


def test_module_import_stmt_abs_pkg_with_prefix():
    # Uses "import pkg.module as local" to avoid test-name collision.
    stmt, local = _module_import_stmt("test_fn.py", "conversion.py", abs_pkg="mylib")
    assert stmt == "import mylib.conversion as conversion"
    assert local == "conversion"


def test_module_import_stmt_abs_pkg_empty():
    # No package prefix → plain "import conversion".
    stmt, local = _module_import_stmt("test_fn.py", "conversion.py", abs_pkg="")
    assert stmt == "import conversion"
    assert local == "conversion"


def test_module_import_stmt_abs_pkg_nested_module():
    # source_file has a nested path within the package
    stmt, local = _module_import_stmt("test_fn.py", "sub/constants.py", abs_pkg="mylib")
    assert stmt == "import mylib.sub.constants as constants"
    assert local == "constants"


# ---------------------------------------------------------------------------
# _rewrite_module_level_stores
# ---------------------------------------------------------------------------


def test_rewrite_module_level_stores_simple():
    src = "_CONST = int('99')\n"
    result = _rewrite_module_level_stores(src, {"_CONST": "constants._CONST"})
    assert result == "constants._CONST = int('99')\n"


def test_rewrite_module_level_stores_augassign():
    src = "X += 1\n"
    result = _rewrite_module_level_stores(src, {"X": "mod.X"})
    assert result == "mod.X += 1\n"


def test_rewrite_module_level_stores_annassign_with_value():
    src = "X: int = 42\n"
    result = _rewrite_module_level_stores(src, {"X": "mod.X"})
    assert result == "mod.X: int = 42\n"


def test_rewrite_module_level_stores_annassign_without_value_skipped():
    # Declaration only — no value, so nothing to rewrite.
    src = "X: int\n"
    result = _rewrite_module_level_stores(src, {"X": "mod.X"})
    assert result == src


def test_rewrite_module_level_stores_function_body_not_rewritten():
    # Assignments inside function bodies must not be touched.
    src = "def f():\n    X = 1\n"
    result = _rewrite_module_level_stores(src, {"X": "mod.X"})
    assert result == src


def test_rewrite_module_level_stores_empty_rewrites():
    src = "X = 1\n"
    assert _rewrite_module_level_stores(src, {}) == src


def test_rewrite_module_level_stores_syntax_error():
    src = "def (broken:\n"
    assert _rewrite_module_level_stores(src, {"X": "mod.X"}) == src


def test_rewrite_module_level_stores_name_not_in_rewrites():
    src = "Y = 1\n"
    result = _rewrite_module_level_stores(src, {"X": "mod.X"})
    assert result == src


def test_rewrite_module_level_stores_augassign_non_name_target():
    # Attribute augmented assignment — target is Attribute, not Name; must be skipped.
    src = "obj.x += 1\n"
    result = _rewrite_module_level_stores(src, {"x": "mod.x"})
    assert result == src


# ---------------------------------------------------------------------------
# _rewrite_module_var_names
# ---------------------------------------------------------------------------


def test_rewrite_module_var_names_basic():
    src = "def fn():\n    if SAFE_MODE:\n        pass\n"
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert "conversion.SAFE_MODE" in result
    # bare SAFE_MODE no longer appears as a standalone Name
    import ast

    tree = ast.parse(result)
    bare = [
        n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "SAFE_MODE"
    ]
    assert bare == []


def test_rewrite_module_var_names_skips_attribute_access():
    # obj.SAFE_MODE must NOT become obj.conversion.SAFE_MODE — the regex approach
    # would corrupt this; the AST approach correctly skips it because 'SAFE_MODE'
    # is the attr string of an Attribute node, not an ast.Name load.
    src = "def fn():\n    return obj.SAFE_MODE\n"
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert result == src


def test_rewrite_module_var_names_skips_strings():
    src = 'x = "SAFE_MODE"\n'
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert result == src


def test_rewrite_module_var_names_skips_comments():
    src = "# use SAFE_MODE here\nx = 1\n"
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert result == src


def test_rewrite_module_var_names_no_partial_name_match():
    # SAFE_MODE_EXTRA is a different identifier and must not be rewritten
    src = "x = SAFE_MODE_EXTRA\ny = SAFE_MODE\n"
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert "SAFE_MODE_EXTRA" in result
    assert "y = conversion.SAFE_MODE" in result


def test_rewrite_module_var_names_empty_rewrites():
    src = "def fn():\n    return SAFE_MODE\n"
    result = _rewrite_module_var_names(src, {})
    assert result == src


def test_rewrite_module_var_names_initial_syntax_error_returns_original():
    # Unparseable source at the start → return unchanged (first ast.parse fails)
    src = "def fn(\n"
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert result == src


def test_rewrite_module_var_names_no_name_nodes_returns_original():
    # Source has no Name nodes for the given key → return unchanged
    src = "x = 1\n"
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert result == src


def test_rewrite_module_var_names_verify_bare_name_survives_returns_original():
    # If a rewrite introduces a new bare Name that itself appears in rewrites,
    # verification catches it and returns the original source.
    # rewrites={"A": "mod.A", "mod": "pkg.mod"}: rewriting "A" → "mod.A" leaves
    # "mod" as a bare Name load, which is in rewrites → verification fails.
    src = "x = A\n"
    result = _rewrite_module_var_names(src, {"A": "mod.A", "mod": "pkg.mod"})
    assert result == src


def test_rewrite_module_var_names_verify_syntax_error_returns_original(monkeypatch):
    # If re-parsing the rewritten result raises SyntaxError (defensive guard),
    # the original source is returned unchanged.
    import crispen.file_limiter.code_gen as _code_gen
    import ast as _ast

    call_count = [0]
    real_parse = _ast.parse

    def patched_parse(src, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] >= 2:  # fail on the verification parse
            raise SyntaxError("synthetic verify failure")
        return real_parse(src, *args, **kwargs)

    monkeypatch.setattr(_code_gen.ast, "parse", patched_parse)
    src = "x = SAFE_MODE\n"
    result = _rewrite_module_var_names(src, {"SAFE_MODE": "conversion.SAFE_MODE"})
    assert result == src


# ---------------------------------------------------------------------------
# _merge_from_imports
# ---------------------------------------------------------------------------


def test_merge_from_imports_no_overlap():
    imports = ["from .a import x", "from .b import y"]
    assert _merge_from_imports(imports) == ["from .a import x", "from .b import y"]


def test_merge_from_imports_overlapping():
    imports = ["from .conv import A, C", "from .conv import B, C"]
    result = _merge_from_imports(imports)
    assert result == ["from .conv import A, B, C"]


def test_merge_from_imports_deduplicates_names():
    imports = ["from .m import foo, bar", "from .m import bar, baz"]
    result = _merge_from_imports(imports)
    assert result == ["from .m import bar, baz, foo"]


def test_merge_from_imports_preserves_plain_imports():
    imports = ["import os", "from .m import x", "import sys"]
    result = _merge_from_imports(imports)
    assert result == ["from .m import x", "import os", "import sys"]


def test_merge_from_imports_empty():
    assert _merge_from_imports([]) == []


# ---------------------------------------------------------------------------
# _sort_imports_pep8
# ---------------------------------------------------------------------------


def test_sort_imports_pep8_basic_ordering():
    # Third-party plain import after relative from-import → should be reordered.
    imports = [
        "from typing import Any",
        "from .conversion import foo",
        "import lupa",
    ]
    result = _sort_imports_pep8(imports)
    assert result == [
        "from typing import Any",
        "import lupa",
        "from .conversion import foo",
    ]


def test_sort_imports_pep8_future_first():
    imports = ["import os", "from __future__ import annotations", "from .x import y"]
    result = _sort_imports_pep8(imports)
    assert result[0] == "from __future__ import annotations"


def test_sort_imports_pep8_preserves_within_group_order():
    imports = ["from .b import y", "from .a import x"]
    result = _sort_imports_pep8(imports)
    # Both are local; original order preserved
    assert result == ["from .b import y", "from .a import x"]


def test_sort_imports_pep8_empty():
    assert _sort_imports_pep8([]) == []


def test_sort_imports_pep8_all_stdlib():
    imports = ["import os", "import sys", "from pathlib import Path"]
    result = _sort_imports_pep8(imports)
    assert result == imports  # already ordered, stable sort keeps original order


# ---------------------------------------------------------------------------
# generate_file_splits — cross-file import integration
# ---------------------------------------------------------------------------


def test_generate_cross_file_import():
    # fn_a goes to fn_module.py; _block_1 (defining _CONST) goes to constants.py.
    # _CONST is a TOP_LEVEL variable that is never reassigned → fn_module.py uses
    # a plain "from .constants import _CONST" (idiomatic Python; no module alias).
    source = "_CONST = 42\n\ndef fn_a():\n    return _CONST\n"
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_fn = _make_entity("fn_a", 3, 4)
    c = _classified(entities=[e_block, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["fn_a"], target_file="fn_module.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    fn_src = result.new_files["fn_module.py"]
    assert "from .constants import _CONST" in fn_src
    assert "from . import constants" not in fn_src
    assert "constants._CONST" not in fn_src
    # constants.py should NOT have a cross-import (it defines _CONST, not uses it)
    const_src = result.new_files["constants.py"]
    assert "from .fn_module" not in const_src


def test_generate_cross_file_import_no_duplicate_names():
    # Two entities (fn_a and fn_b) migrate to the same new file.
    # fn_a uses X and Z from helpers; fn_b uses Y and Z from helpers.
    # X, Y, Z are TOP_LEVEL variables that are never reassigned → the new file
    # gets ONE "from .constants import X, Y, Z" (no module alias needed).
    source = textwrap.dedent(
        """\
        X = 1
        Y = 2
        Z = 3

        def fn_a():
            return X + Z

        def fn_b():
            return Y + Z
        """
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["X", "Y", "Z"])
    e_a = _make_entity("fn_a", 5, 6)
    e_b = _make_entity("fn_b", 8, 9)
    c = _classified(entities=[e_block, e_a, e_b])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["fn_a", "fn_b"], target_file="funcs.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    funcs_src = result.new_files["funcs.py"]
    # Both fn_a and fn_b are present
    assert "def fn_a" in funcs_src
    assert "def fn_b" in funcs_src
    # Plain from-import (no module alias) since none of X/Y/Z are reassigned
    assert "from .constants import" in funcs_src
    assert "from . import constants" not in funcs_src
    # Variables are referenced by their bare names, not as module attributes
    assert "constants.X" not in funcs_src
    assert "constants.Y" not in funcs_src
    assert "constants.Z" not in funcs_src


def test_generate_cross_file_import_reassigned_uses_module_alias():
    # _CONST is defined by _block_1 (→ constants.py) AND reassigned by _block_2
    # (non-migrated, stays in big.py).  Because _CONST is stored by a different
    # entity, fn_module.py must use the module-alias form so that any mutation of
    # _CONST propagates through the module reference rather than a stale copy.
    source = textwrap.dedent(
        """\
        _CONST = 42
        _CONST = int("99")

        def fn_a():
            return _CONST
        """
    )
    e_block1 = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_block2 = Entity(EntityKind.TOP_LEVEL, "_block_2", 2, 2, ["_CONST"])
    e_fn = _make_entity("fn_a", 4, 5)
    c = _classified(entities=[e_block1, e_block2, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["fn_a"], target_file="fn_module.py"),
            # _block_2 stays (non-migrated) — its store makes _CONST "reassigned"
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    fn_src = result.new_files["fn_module.py"]
    # _CONST is reassigned → module-alias import so mutations propagate.
    assert "from . import constants" in fn_src
    assert "constants._CONST" in fn_src
    assert "from .constants import _CONST" not in fn_src


def test_generate_cross_file_reassigned_original_file_uses_module_alias():
    # _CONST is defined by _block_1 (migrated) and reassigned by _block_2
    # (non-migrated).
    # The original file must rewrite both the load in fn_a AND the module-level
    # store in _block_2 to constants._CONST so that the reassignment updates the
    # value in constants.py rather than creating an orphaned local binding.
    source = textwrap.dedent(
        """\
        _CONST = 42
        _CONST = int("99")

        def fn_a():
            return _CONST
        """
    )
    e_block1 = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_block2 = Entity(EntityKind.TOP_LEVEL, "_block_2", 2, 2, ["_CONST"])
    e_fn = _make_entity("fn_a", 4, 5)
    c = _classified(entities=[e_block1, e_block2, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            # _block_2 and fn_a stay (non-migrated)
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    assert not result.abort
    orig = result.original_source
    # Module-level import added for the module alias.
    assert "from . import constants" in orig
    # Both the store (_block_2) and the load (fn_a) are rewritten.
    assert 'constants._CONST = int("99")' in orig
    assert "return constants._CONST" in orig
    # Must NOT bind _CONST as a bare name via from-import (would shadow the rewrite)
    assert "from .constants import _CONST" not in orig


def test_generate_reassigned_all_entities_migrated_no_original_processing():
    # When ALL entities are migrated, non_migrated_entity_names is empty and the
    # original-file module-alias processing block must be skipped without error.
    source = "_CONST = 42\n_CONST = 99\n\ndef fn_a():\n    return _CONST\n"
    e_block1 = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_block2 = Entity(EntityKind.TOP_LEVEL, "_block_2", 2, 2, ["_CONST"])
    e_fn = _make_entity("fn_a", 4, 5)
    c = _classified(entities=[e_block1, e_block2, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["_block_2", "fn_a"], target_file="funcs.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")
    # Does not abort or crash; original source may be minimal.
    assert not result.abort


def test_generate_reassigned_two_entities_same_file_single_module_import():
    # Two entities in the same new file both reference a reassigned constant.
    # The same "from . import constants" import must appear only once
    # (seen_top_cross deduplication).
    source = textwrap.dedent(
        """\
        _CONST = 42
        _CONST = 99

        def fn_a():
            return _CONST

        def fn_b():
            return _CONST
        """
    )
    e_block1 = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_block2 = Entity(EntityKind.TOP_LEVEL, "_block_2", 2, 2, ["_CONST"])
    e_fn_a = _make_entity("fn_a", 4, 5)
    e_fn_b = _make_entity("fn_b", 7, 8)
    c = _classified(entities=[e_block1, e_block2, e_fn_a, e_fn_b])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["fn_a", "fn_b"], target_file="funcs.py"),
            # _block_2 stays non-migrated → makes _CONST "reassigned"
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    funcs_src = result.new_files["funcs.py"]
    # The module import must appear exactly once despite two entities needing it.
    import_lines = [ln for ln in funcs_src.splitlines() if "import constants" in ln]
    assert len(import_lines) == 1


def test_generate_aborts_when_test_class_used_in_decorator():
    # TestFixture (a Test* class) provides PARAMS used in a parametrize decorator
    # on test_fn.  If they are split into different files, TestFixture would need
    # to be imported inline (to avoid pytest duplicate collection), but that
    # import would not be in scope when the decorator is evaluated.
    source = textwrap.dedent(
        """\
        import pytest

        class TestFixture:
            PARAMS = [1, 2, 3]

        @pytest.mark.parametrize("x", TestFixture.PARAMS)
        def test_fn(x):
            assert x
        """
    )
    e_fixture = Entity(EntityKind.CLASS, "TestFixture", 3, 4, ["TestFixture"])
    e_fn = _make_entity("test_fn", 6, 8)
    c = _classified(entities=[e_fixture, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["TestFixture"], target_file="test_fixture.py"),
            GroupPlacement(group=["test_fn"], target_file="test_fns.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "tests/test_original.py")

    assert result.abort
    assert "TestFixture" in result.abort_reason
    assert "decorator" in result.abort_reason


def test_generate_non_migrated_helper_extracted_to_new_file():
    # _run is non-migrated; test_fn is migrated and references _run.
    # _run is extracted into test_helpers.py to prevent an O→F→O cycle.
    source = textwrap.dedent(
        """\
        import textwrap

        def _run(x):
            return x

        def test_fn():
            return _run(1)
    """
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["textwrap"])
    e_run = _make_entity("_run", 3, 4)
    e_test = _make_entity("test_fn", 6, 7)
    c = _classified(entities=[e_block, e_run, e_test])
    plan = _plan([GroupPlacement(group=["test_fn"], target_file="test_helpers.py")])

    result = generate_file_splits(c, plan, source, "original.py")

    new_src = result.new_files["test_helpers.py"]
    # _run is defined in the new file (extracted), not imported from original
    assert "def _run" in new_src
    assert "from .original import _run" not in new_src
    # import textwrap is not referenced by either entity
    assert "from .original import textwrap" not in new_src


def test_generate_self_referential_placement_dropped():
    # LLM names a target file the same as the original → would create a
    # circular import.  The placement must be silently dropped so the entity
    # stays in the original file and no self-import is added.
    source = "class Foo:\n    pass\n\nclass Bar:\n    pass\n"
    e_foo = _make_entity("Foo", 1, 2)
    e_bar = _make_entity("Bar", 4, 5)
    c = _classified(entities=[e_foo, e_bar])
    # "mymodule.py" is also the original filename → self-referential
    plan = _plan(
        [
            GroupPlacement(group=["Foo"], target_file="mymodule.py"),
            GroupPlacement(group=["Bar"], target_file="helpers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "mymodule.py")

    # Foo stays in the original — no circular self-import
    assert "from .mymodule import Foo" not in result.original_source
    assert "mymodule.py" not in result.new_files
    # Bar is still moved normally
    assert "helpers.py" in result.new_files
    assert "class Bar" in result.new_files["helpers.py"]
    # Foo remains in the original source (not removed)
    assert "class Foo" in result.original_source


def test_generate_all_placements_self_referential():
    # All placements target the original file → nothing is moved.
    source = "def foo():\n    pass\n"
    e_foo = _make_entity("foo", 1, 2)
    c = _classified(entities=[e_foo])
    plan = _plan([GroupPlacement(group=["foo"], target_file="original.py")])

    result = generate_file_splits(c, plan, source, "original.py")

    assert result.new_files == {}
    assert "from .original import foo" not in result.original_source
    assert "def foo" in result.original_source


def test_generate_aborts_on_cross_file_import_cycle():
    # fn_a references fn_b (in b.py) and fn_b references fn_a (in a.py).
    # This creates a circular import a.py ↔ b.py that Python cannot load.
    # generate_file_splits must detect the cycle and abort rather than emit
    # broken code.
    source = "def fn_a():\n    return fn_b()\n\ndef fn_b():\n    return fn_a()\n"
    e_a = _make_entity("fn_a", 1, 2)
    e_b = _make_entity("fn_b", 4, 5)
    c = _classified(entities=[e_a, e_b])
    plan = _plan(
        [
            GroupPlacement(group=["fn_a"], target_file="a.py"),
            GroupPlacement(group=["fn_b"], target_file="b.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    assert result.abort is True
    assert result.new_files == {}


def test_generate_aborts_on_cycle_through_original():
    # _CONST is a TOP_LEVEL constant (stays in original).
    # _worker is migrated to helpers.py and references _CONST.
    # main() (non-migrated) calls _worker → original will re-export _worker.
    # Cycle: original → helpers.py (re-export of _worker)
    #              → original (via `from .original import _CONST`).
    source = textwrap.dedent(
        """\
        _CONST = "value"

        def _worker():
            return _CONST

        def main():
            return _worker()
    """
    )
    e_const = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_worker = _make_entity("_worker", 3, 4)
    e_main = _make_entity("main", 6, 7)
    c = _classified(entities=[e_const, e_worker, e_main])
    plan = _plan([GroupPlacement(group=["_worker"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "original.py")

    # helpers.py would need `from .original import _CONST` while original
    # re-exports _worker from helpers.py → circular import → must abort.
    assert result.abort is True
    assert result.new_files == {}


def test_generate_aborts_on_cycle_through_original_test_subdir():
    # In a test-file subdir split non_migrated_home ("test_svc.py") differs
    # from original_basename ("svc/__init__.py").  The cycle detection must
    # treat the original test file as its own graph node:
    #
    # _CONFIG stays in test_svc.py (TOP_LEVEL, non-migrated).
    # _helper is migrated to svc/test_helpers.py and references _CONFIG.
    # test_fn (non-migrated) calls _helper → test_svc.py re-exports _helper.
    # Cycle: test_svc.py → svc/test_helpers.py (re-export of _helper)
    #              → test_svc.py (via `from ..test_svc import _CONFIG`).
    source = textwrap.dedent(
        """\
        _CONFIG = "value"

        def _helper():
            return _CONFIG

        def test_fn():
            return _helper()
    """
    )
    e_config = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONFIG"])
    e_helper = _make_entity("_helper", 3, 4)
    e_test = _make_entity("test_fn", 6, 7)
    c = _classified(entities=[e_config, e_helper, e_test])
    plan = _plan([GroupPlacement(group=["_helper"], target_file="svc/test_helpers.py")])

    result = generate_file_splits(
        c, plan, source, "tests/test_svc.py", subdir_name="svc"
    )

    # svc/test_helpers.py imports _CONFIG from test_svc.py, and test_svc.py
    # re-exports _helper from svc/test_helpers.py → circular import → abort.
    assert result.abort is True
    assert result.new_files == {}


# ---------------------------------------------------------------------------
# generate_file_splits — TYPE_CHECKING imports for quoted annotations
# ---------------------------------------------------------------------------


def test_generate_file_splits_type_checking_for_quoted_annotation():
    # _advise_set3 uses Optional["_LLMAccumulator"] (quoted annotation).
    # _LLMAccumulator is migrated to models.py; _advise_set3 goes to placements.py.
    # placements.py must get:
    #   from typing import TYPE_CHECKING
    #   if TYPE_CHECKING:
    #       from .models import _LLMAccumulator
    source = textwrap.dedent(
        """\
        from typing import Optional

        class _LLMAccumulator:
            pass

        def _advise_set3(acc: Optional["_LLMAccumulator"]) -> None:
            pass
        """
    )
    e_acc = Entity(EntityKind.CLASS, "_LLMAccumulator", 3, 4, ["_LLMAccumulator"])
    e_fn = Entity(EntityKind.FUNCTION, "_advise_set3", 6, 7, ["_advise_set3"])
    c = _classified(entities=[e_acc, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_LLMAccumulator"], target_file="models.py"),
            GroupPlacement(group=["_advise_set3"], target_file="placements.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "advisor.py")

    placements_src = result.new_files["placements.py"]
    # TYPE_CHECKING may be merged into an existing "from typing import ..." line.
    assert "TYPE_CHECKING" in placements_src
    assert "if TYPE_CHECKING:" in placements_src
    assert "from .models import _LLMAccumulator" in placements_src


def test_generate_file_splits_type_checking_deduplication():
    # Two functions in the same target file both reference "_LLMAccumulator"
    # in quoted annotations.  The TYPE_CHECKING import should appear only once
    # even though both entities trigger _find_cross_file_type_checking_imports.
    source = textwrap.dedent(
        """\
        from typing import Optional

        class _LLMAccumulator:
            pass

        def _fn_a(x: Optional["_LLMAccumulator"]) -> None:
            pass

        def _fn_b(y: Optional["_LLMAccumulator"]) -> None:
            pass
        """
    )
    e_acc = Entity(EntityKind.CLASS, "_LLMAccumulator", 3, 4, ["_LLMAccumulator"])
    e_fna = Entity(EntityKind.FUNCTION, "_fn_a", 6, 7, ["_fn_a"])
    e_fnb = Entity(EntityKind.FUNCTION, "_fn_b", 9, 10, ["_fn_b"])
    c = _classified(entities=[e_acc, e_fna, e_fnb])
    plan = _plan(
        [
            GroupPlacement(group=["_LLMAccumulator"], target_file="models.py"),
            GroupPlacement(group=["_fn_a", "_fn_b"], target_file="placements.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "advisor.py")

    placements_src = result.new_files["placements.py"]
    assert placements_src.count("from .models import _LLMAccumulator") == 1


def test_generate_file_splits_tc_dedup_drops_when_already_in_regular():
    # Entity A uses _Acc at runtime (unquoted annotation → regular cross-file import).
    # Entity B uses _Acc only in a quoted annotation → would normally get a TC import.
    # Both go to workers.py.  The dedup step must remove the TC import entirely since
    # _Acc is already covered by the regular import.
    source = textwrap.dedent(
        """\
        from typing import Optional

        class _Acc:
            pass

        def fn_runtime(x) -> None:
            a: _Acc = x

        def fn_quoted(x: Optional["_Acc"]) -> None:
            pass
        """
    )
    e_acc = Entity(EntityKind.CLASS, "_Acc", 3, 4, ["_Acc"])
    e_rt = Entity(EntityKind.FUNCTION, "fn_runtime", 6, 7, ["fn_runtime"])
    e_qt = Entity(EntityKind.FUNCTION, "fn_quoted", 9, 10, ["fn_quoted"])
    c = _classified(entities=[e_acc, e_rt, e_qt])
    plan = _plan(
        [
            GroupPlacement(group=["_Acc"], target_file="models.py"),
            GroupPlacement(group=["fn_runtime", "fn_quoted"], target_file="workers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "advisor.py")

    workers_src = result.new_files["workers.py"]
    # Regular import must be present, TYPE_CHECKING block must NOT be.
    assert "from .models import _Acc" in workers_src
    assert "if TYPE_CHECKING:" not in workers_src


def test_generate_file_splits_tc_dedup_plain_import_branches():
    # Covers the non-from-import branches in the dedup loop:
    #   • "import sys" in needed → _FROM_IMPORT_RE does not match (2633->2631 branch)
    #   • "import typing_extensions" in needed_tc (annotation-only) → TC import is a
    #     plain import statement, not a from-import (2655 branch)
    source = textwrap.dedent(
        """\
        import sys
        import typing_extensions
        from typing import Optional

        class _Acc:
            pass

        def fn(x: Optional["_Acc"]) -> None:
            sys.exit(0)

        def fn2() -> "typing_extensions.Literal":
            pass
        """
    )
    e_acc = Entity(EntityKind.CLASS, "_Acc", 5, 6, ["_Acc"])
    e_fn = Entity(EntityKind.FUNCTION, "fn", 8, 9, ["fn"])
    e_fn2 = Entity(EntityKind.FUNCTION, "fn2", 11, 12, ["fn2"])
    c = _classified(entities=[e_acc, e_fn, e_fn2])
    plan = _plan(
        [
            GroupPlacement(group=["_Acc"], target_file="models.py"),
            GroupPlacement(group=["fn", "fn2"], target_file="workers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "advisor.py")

    workers_src = result.new_files["workers.py"]
    # TC import for _Acc (cross-file, quoted annotation) must still be present.
    assert "if TYPE_CHECKING:" in workers_src
    assert "_Acc" in workers_src
    # Plain import for typing_extensions preserved in TC block.
    assert "typing_extensions" in workers_src


# ---------------------------------------------------------------------------
# _extract_shared_helpers
# ---------------------------------------------------------------------------


def _make_classified(entities, migrated_names=None):
    migrated = set(migrated_names or [])
    return (
        ClassifiedEntities(
            entities=entities,
            entity_class={},
            graph={},
            set_1=[],
            set_2_groups=[],
            set_3_groups=[],
            abort=False,
        ),
        migrated,
    )


# ---------------------------------------------------------------------------
# _topo_depth
# ---------------------------------------------------------------------------


def test_topo_depth_empty():
    assert _topo_depth({}) == {}


def test_topo_depth_dag():
    # Linear chain: a → b → c.  c is the leaf (depth 0), b has depth 1, a depth 2.
    # The outer loop visits a first, which recurses into b then c, memoising both.
    # When the outer loop reaches b and c they are already in depths (True branch).
    graph = {"a": {"b"}, "b": {"c"}, "c": set()}
    assert _topo_depth(graph) == {"a": 2, "b": 1, "c": 0}


def test_topo_depth_cycle():
    graph = {"a": {"b"}, "b": {"a"}}
    assert _topo_depth(graph) == {"a": 0, "b": 0}


# ---------------------------------------------------------------------------
# _extract_shared_helpers
# ---------------------------------------------------------------------------


def test_extract_shared_helpers_extracts_referenced_function():
    # _helper is non-migrated, test_fn (migrated to helpers.py) references it.
    e_helper = Entity(EntityKind.FUNCTION, "_helper", 1, 2, ["_helper"])
    e_test = Entity(EntityKind.FUNCTION, "test_fn", 4, 6, ["test_fn"])
    classified, migrated_names = _make_classified([e_helper, e_test], ["test_fn"])
    entity_map = {"_helper": e_helper, "test_fn": e_test}
    entity_source_map = {
        "_helper": "def _helper():\n    pass",
        "test_fn": "def test_fn():\n    return _helper()",
    }
    file_entity_names = {"helpers.py": ["test_fn"]}
    name_to_target_file = {"_helper": "original.py", "test_fn": "helpers.py"}

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    # _helper extracted into helpers.py (prepended before test_fn)
    assert file_entity_names["helpers.py"] == ["_helper", "test_fn"]
    assert "_helper" in migrated_names
    assert name_to_target_file["_helper"] == "helpers.py"
    assert len(synthetic) == 1
    assert synthetic[0].group == ["_helper"]
    assert synthetic[0].target_file == "helpers.py"


def test_extract_shared_helpers_skips_top_level_entities():
    # TOP_LEVEL entities are not extracted (only FUNCTION/CLASS).
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_test = Entity(EntityKind.FUNCTION, "test_fn", 3, 4, ["test_fn"])
    classified, migrated_names = _make_classified([e_block, e_test], ["test_fn"])
    entity_map = {"_block_1": e_block, "test_fn": e_test}
    entity_source_map = {
        "_block_1": "_CONST = 42",
        "test_fn": "def test_fn():\n    return _CONST",
    }
    file_entity_names = {"helpers.py": ["test_fn"]}
    name_to_target_file = {"_CONST": "original.py", "test_fn": "helpers.py"}

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    assert "_block_1" not in migrated_names
    assert file_entity_names["helpers.py"] == ["test_fn"]
    assert synthetic == []


def test_extract_shared_helpers_extracts_only_once_for_multiple_refs():
    # _helper referenced twice in the same migrated entity → extracted once.
    e_helper = Entity(EntityKind.FUNCTION, "_helper", 1, 2, ["_helper"])
    e_test = Entity(EntityKind.FUNCTION, "test_fn", 4, 6, ["test_fn"])
    classified, migrated_names = _make_classified([e_helper, e_test], ["test_fn"])
    entity_map = {"_helper": e_helper, "test_fn": e_test}
    entity_source_map = {
        "_helper": "def _helper():\n    pass",
        "test_fn": "def test_fn():\n    _helper()\n    _helper()",
    }
    file_entity_names = {"helpers.py": ["test_fn"]}
    name_to_target_file = {"_helper": "original.py", "test_fn": "helpers.py"}

    _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    assert file_entity_names["helpers.py"].count("_helper") == 1


def test_extract_shared_helpers_skips_name_already_pointing_to_other_target():
    # A non-migrated FUNCTION entity whose defined name already points to a
    # non-original target in name_to_target_file (e.g. a migrated entity also
    # defines it) should not be added to defined_to_entity.
    e_helper = Entity(EntityKind.FUNCTION, "_helper", 1, 2, ["_helper"])
    e_test = Entity(EntityKind.FUNCTION, "test_fn", 4, 5, ["test_fn"])
    classified, migrated_names = _make_classified([e_helper, e_test], ["test_fn"])
    entity_map = {"_helper": e_helper, "test_fn": e_test}
    entity_source_map = {
        "_helper": "def _helper(): pass",
        "test_fn": "def test_fn(): return _helper()",
    }
    file_entity_names = {"helpers.py": ["test_fn"]}
    # _helper already points to helpers.py (not original) — skip it
    name_to_target_file = {"_helper": "helpers.py", "test_fn": "helpers.py"}

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    assert "_helper" not in migrated_names
    assert synthetic == []


def test_extract_shared_helpers_no_extraction_when_no_original_dep():
    # test_fn references other_fn which is also migrated → no extraction needed.
    e_other = Entity(EntityKind.FUNCTION, "other_fn", 1, 2, ["other_fn"])
    e_test = Entity(EntityKind.FUNCTION, "test_fn", 4, 5, ["test_fn"])
    classified, migrated_names = _make_classified(
        [e_other, e_test], ["test_fn", "other_fn"]
    )
    entity_map = {"other_fn": e_other, "test_fn": e_test}
    entity_source_map = {
        "other_fn": "def other_fn():\n    pass",
        "test_fn": "def test_fn():\n    return other_fn()",
    }
    file_entity_names = {"helpers.py": ["test_fn", "other_fn"]}
    name_to_target_file = {"other_fn": "helpers.py", "test_fn": "helpers.py"}

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    assert synthetic == []
    assert file_entity_names["helpers.py"] == ["test_fn", "other_fn"]


def test_extract_shared_helpers_transitive_pull_in():
    # _helper_a is directly wanted by fn_a (in f1.py).
    # _helper_a's source calls _helper_b (non-migrated, in original).
    # _helper_b must be transitively extracted into f1.py to prevent an
    # O→f1.py cycle (f1.py imports _helper_a which calls _helper_b in original;
    # original re-exports _helper_a from f1.py → cycle).
    e_a = Entity(EntityKind.FUNCTION, "_helper_a", 1, 2, ["_helper_a"])
    e_b = Entity(EntityKind.FUNCTION, "_helper_b", 3, 4, ["_helper_b"])
    e_fn = Entity(EntityKind.FUNCTION, "fn_a", 6, 7, ["fn_a"])
    classified, migrated_names = _make_classified([e_a, e_b, e_fn], ["fn_a"])
    entity_map = {"_helper_a": e_a, "_helper_b": e_b, "fn_a": e_fn}
    entity_source_map = {
        "_helper_a": "def _helper_a():\n    _helper_b()",
        "_helper_b": "def _helper_b():\n    pass",
        "fn_a": "def fn_a():\n    _helper_a()",
    }
    file_entity_names = {"f1.py": ["fn_a"]}
    name_to_target_file = {
        "_helper_a": "original.py",
        "_helper_b": "original.py",
        "fn_a": "f1.py",
    }

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    # Both helpers extracted into f1.py.
    assert "_helper_a" in file_entity_names["f1.py"]
    assert "_helper_b" in file_entity_names["f1.py"]
    assert "_helper_a" in migrated_names
    assert "_helper_b" in migrated_names
    assert name_to_target_file["_helper_a"] == "f1.py"
    assert name_to_target_file["_helper_b"] == "f1.py"
    assert len(synthetic) == 2


def test_extract_shared_helpers_scc_prevents_new_to_new_cycle():
    # helper_a is wanted by f1.py; helper_b is wanted by f2.py.
    # They mutually reference each other → one SCC → must go to the same file
    # to prevent the F1→F2→F1 import cycle.
    e_a = Entity(EntityKind.FUNCTION, "helper_a", 1, 2, ["helper_a"])
    e_b = Entity(EntityKind.FUNCTION, "helper_b", 3, 4, ["helper_b"])
    e_fn1 = Entity(EntityKind.FUNCTION, "fn_1", 6, 7, ["fn_1"])
    e_fn2 = Entity(EntityKind.FUNCTION, "fn_2", 9, 10, ["fn_2"])
    classified = ClassifiedEntities(
        entities=[e_a, e_b, e_fn1, e_fn2],
        entity_class={},
        graph={
            "helper_a": {"helper_b"},
            "helper_b": {"helper_a"},
            "fn_1": set(),
            "fn_2": set(),
        },
        set_1=[],
        set_2_groups=[],
        set_3_groups=[],
        abort=False,
    )
    migrated_names = {"fn_1", "fn_2"}
    entity_map = {"helper_a": e_a, "helper_b": e_b, "fn_1": e_fn1, "fn_2": e_fn2}
    entity_source_map = {
        "helper_a": "def helper_a():\n    helper_b()",
        "helper_b": "def helper_b():\n    helper_a()",
        "fn_1": "def fn_1():\n    helper_a()",
        "fn_2": "def fn_2():\n    helper_b()",
    }
    file_entity_names = {"f1.py": ["fn_1"], "f2.py": ["fn_2"]}
    name_to_target_file = {
        "helper_a": "original.py",
        "helper_b": "original.py",
        "fn_1": "f1.py",
        "fn_2": "f2.py",
    }

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    # Both helpers must land in the same file (f1.py is first in plan order).
    assert name_to_target_file["helper_a"] == name_to_target_file["helper_b"]
    chosen = name_to_target_file["helper_a"]
    assert "helper_a" in file_entity_names[chosen]
    assert "helper_b" in file_entity_names[chosen]
    assert "helper_a" in migrated_names
    assert "helper_b" in migrated_names
    # One synthetic placement covering both (single SCC).
    assert len(synthetic) == 1
    assert set(synthetic[0].group) == {"helper_a", "helper_b"}


def test_extract_shared_helpers_transitive_dep_already_wanted():
    # helper_a is directly wanted by f1.py; helper_b is directly wanted by f2.py.
    # helper_a's source also references helper_b (transitive), so helper_b's
    # wanting-set grows from {f2.py} to {f1.py, f2.py} — True branch of the
    # transitive update condition.
    e_a = Entity(EntityKind.FUNCTION, "helper_a", 1, 2, ["helper_a"])
    e_b = Entity(EntityKind.FUNCTION, "helper_b", 3, 4, ["helper_b"])
    e_fn1 = Entity(EntityKind.FUNCTION, "fn_1", 6, 7, ["fn_1"])
    e_fn2 = Entity(EntityKind.FUNCTION, "fn_2", 9, 10, ["fn_2"])
    classified, migrated_names = _make_classified(
        [e_a, e_b, e_fn1, e_fn2], ["fn_1", "fn_2"]
    )
    entity_map = {"helper_a": e_a, "helper_b": e_b, "fn_1": e_fn1, "fn_2": e_fn2}
    entity_source_map = {
        "helper_a": "def helper_a():\n    helper_b()",
        "helper_b": "def helper_b():\n    pass",
        "fn_1": "def fn_1():\n    helper_a()",
        "fn_2": "def fn_2():\n    helper_b()",
    }
    file_entity_names = {"f1.py": ["fn_1"], "f2.py": ["fn_2"]}
    name_to_target_file = {
        "helper_a": "original.py",
        "helper_b": "original.py",
        "fn_1": "f1.py",
        "fn_2": "f2.py",
    }

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    # Both helpers are extracted (as separate SCCs since no mutual cycle in graph).
    assert "helper_a" in migrated_names
    assert "helper_b" in migrated_names
    # Two synthetic placements — one for each singleton SCC.
    assert len(synthetic) == 2


def test_extract_shared_helpers_transitive_dep_no_new_targets():
    # fn_1 directly references both helper_a and helper_b.
    # helper_a's source also references helper_b (transitive dep).
    # When the transitive loop processes helper_a, helper_b already has the same
    # wanting-set {f1.py} → new_targets is empty → False branch of update condition.
    e_a = Entity(EntityKind.FUNCTION, "helper_a", 1, 2, ["helper_a"])
    e_b = Entity(EntityKind.FUNCTION, "helper_b", 3, 4, ["helper_b"])
    e_fn = Entity(EntityKind.FUNCTION, "fn_1", 6, 7, ["fn_1"])
    classified, migrated_names = _make_classified([e_a, e_b, e_fn], ["fn_1"])
    entity_map = {"helper_a": e_a, "helper_b": e_b, "fn_1": e_fn}
    entity_source_map = {
        "helper_a": "def helper_a():\n    helper_b()",
        "helper_b": "def helper_b():\n    pass",
        "fn_1": "def fn_1():\n    helper_a()\n    helper_b()",
    }
    file_entity_names = {"f1.py": ["fn_1"]}
    name_to_target_file = {
        "helper_a": "original.py",
        "helper_b": "original.py",
        "fn_1": "f1.py",
    }

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    # Both helpers are still extracted; the transitive dep on helper_b is a no-op
    # because helper_b already has {f1.py} in its wanting-set (direct want).
    assert "helper_a" in migrated_names
    assert "helper_b" in migrated_names
    assert len(synthetic) == 2


def test_extract_shared_helpers_avoids_cycle_by_choosing_downstream_file():
    # _run is wanted by both test_skip.py and test_transformers.py.
    # test_skip.py already imports from test_transformers.py (_RaisingTransformer).
    # Placing _run in test_skip.py would force test_transformers.py to import from
    # test_skip.py → cycle.  The cycle-aware logic must pick test_transformers.py
    # (the downstream file) instead.
    e_raise = Entity(
        EntityKind.FUNCTION, "_RaisingTransformer", 1, 3, ["_RaisingTransformer"]
    )
    e_run = Entity(EntityKind.FUNCTION, "_run", 4, 5, ["_run"])
    e_skip = Entity(EntityKind.FUNCTION, "fn_skip", 7, 9, ["fn_skip"])
    e_transform = Entity(EntityKind.FUNCTION, "fn_transform", 11, 13, ["fn_transform"])
    classified, migrated_names = _make_classified(
        [e_raise, e_run, e_skip, e_transform],
        ["fn_skip", "fn_transform", "_RaisingTransformer"],
    )
    entity_map = {
        "_RaisingTransformer": e_raise,
        "_run": e_run,
        "fn_skip": e_skip,
        "fn_transform": e_transform,
    }
    entity_source_map = {
        "_RaisingTransformer": "def _RaisingTransformer():\n    pass",
        "_run": "def _run(x):\n    return x",
        # fn_skip refs _RaisingTransformer (migrated to test_transformers.py) AND
        # _run (non-migrated) → _run is wanted by test_skip.py.
        "fn_skip": "def fn_skip():\n    _RaisingTransformer()\n    _run(1)",
        # fn_transform also refs _run → _run is wanted by test_transformers.py too.
        "fn_transform": "def fn_transform():\n    _run(2)",
    }
    file_entity_names = {
        "test_skip.py": ["fn_skip"],
        "test_transformers.py": ["fn_transform", "_RaisingTransformer"],
    }
    name_to_target_file = {
        "_RaisingTransformer": "test_transformers.py",
        "_run": "original.py",
        "fn_skip": "test_skip.py",
        "fn_transform": "test_transformers.py",
    }

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    # _run must go to test_transformers.py, not test_skip.py.
    assert name_to_target_file["_run"] == "test_transformers.py"
    assert "_run" in file_entity_names["test_transformers.py"]
    assert "_run" not in file_entity_names["test_skip.py"]
    assert "_run" in migrated_names
    assert len(synthetic) == 1
    assert synthetic[0].group == ["_run"]
    assert synthetic[0].target_file == "test_transformers.py"


def test_extract_shared_helpers_skips_scc_when_no_cycle_free_placement():
    # fn_1 (in f1.py) refs fn_2 (in f2.py) and fn_2 refs fn_1 → pre-existing
    # cycle in file_deps.  fn_1 also refs helper_h (non-migrated), which itself
    # refs fn_2.  The only candidate for helper_h is f1.py; placing it there
    # would still result in a cycle (f1.py→f2.py→f1.py already exists).
    # Since no cycle-free placement exists, the SCC is skipped entirely.
    e_fn1 = Entity(EntityKind.FUNCTION, "fn_1", 1, 2, ["fn_1"])
    e_fn2 = Entity(EntityKind.FUNCTION, "fn_2", 4, 5, ["fn_2"])
    e_h = Entity(EntityKind.FUNCTION, "helper_h", 7, 8, ["helper_h"])
    classified, migrated_names = _make_classified([e_fn1, e_fn2, e_h], ["fn_1", "fn_2"])
    entity_map = {"fn_1": e_fn1, "fn_2": e_fn2, "helper_h": e_h}
    entity_source_map = {
        "fn_1": "def fn_1():\n    fn_2()\n    helper_h()",
        "fn_2": "def fn_2():\n    fn_1()",
        "helper_h": "def helper_h():\n    fn_2()",
    }
    file_entity_names = {"f1.py": ["fn_1"], "f2.py": ["fn_2"]}
    name_to_target_file = {
        "fn_1": "f1.py",
        "fn_2": "f2.py",
        "helper_h": "original.py",
    }

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    # helper_h is skipped — no placement avoids the pre-existing cycle.
    assert "helper_h" not in migrated_names
    assert synthetic == []


def test_extract_shared_helpers_helper_refs_migrated_entity_in_other_file():
    # helper_a (non-migrated) references fn_2 (migrated to f2.py).
    # When placed in f1.py the trial and apply phases must account for the
    # resulting f1.py → f2.py dependency edge.
    e_fn1 = Entity(EntityKind.FUNCTION, "fn_1", 1, 2, ["fn_1"])
    e_fn2 = Entity(EntityKind.FUNCTION, "fn_2", 4, 5, ["fn_2"])
    e_helper = Entity(EntityKind.FUNCTION, "helper_a", 7, 8, ["helper_a"])
    classified, migrated_names = _make_classified(
        [e_fn1, e_fn2, e_helper], ["fn_1", "fn_2"]
    )
    entity_map = {"fn_1": e_fn1, "fn_2": e_fn2, "helper_a": e_helper}
    entity_source_map = {
        "fn_1": "def fn_1():\n    helper_a()",
        "fn_2": "def fn_2():\n    pass",
        "helper_a": "def helper_a():\n    fn_2()",
    }
    file_entity_names = {"f1.py": ["fn_1"], "f2.py": ["fn_2"]}
    name_to_target_file = {
        "fn_1": "f1.py",
        "fn_2": "f2.py",
        "helper_a": "original.py",
    }

    synthetic = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        "original.py",
    )

    # helper_a is extracted to f1.py; its dep on fn_2 (f2.py) is tracked in
    # both the trial and apply dep-file branches.
    assert "helper_a" in migrated_names
    assert name_to_target_file["helper_a"] == "f1.py"
    assert len(synthetic) == 1
    assert synthetic[0].target_file == "f1.py"


def test_generate_no_circular_import_when_helper_referenced_by_migrated():
    # Integration test: _run stays in original and is used by test_fn (migrated).
    # Without the fix: original → helpers.py (re-export) and helpers.py → original.
    # With the fix: _run is moved into helpers.py; original imports _run from helpers.
    source = textwrap.dedent(
        """\
        def _run(x):
            return x

        def test_fn(tmp_path):
            return _run(tmp_path)
    """
    )
    e_run = _make_entity("_run", 1, 2)
    e_test = _make_entity("test_fn", 4, 5)
    c = _classified(entities=[e_run, e_test])
    plan = _plan([GroupPlacement(group=["test_fn"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "original.py")

    helpers_src = result.new_files["helpers.py"]
    # _run is defined in helpers.py (extracted), not imported from original
    assert "def _run" in helpers_src
    assert "from .original import _run" not in helpers_src
    # original re-imports _run from helpers.py (since it's still used there via
    # non-migrated code — but in this minimal example there's nothing left)
    # At minimum, no circular self-import exists
    assert "from .original import" not in helpers_src


# ---------------------------------------------------------------------------
# _prune_unused_imports
# ---------------------------------------------------------------------------


def test_prune_unused_imports_syntax_error():
    # Unparseable source → returned unchanged.
    source = "def (invalid syntax"
    assert _prune_unused_imports(source) == source


def test_prune_unused_imports_no_replacements_needed():
    # All imports are fully used → fast-path returns source unchanged.
    source = "import os\n\ndef f():\n    os.getcwd()\n"
    assert _prune_unused_imports(source) == source


def test_prune_unused_imports_preserves_future_import():
    # __future__ imports are always kept, even when the name isn't referenced.
    source = "from __future__ import annotations\n\ndef f():\n    pass\n"
    result = _prune_unused_imports(source)
    assert "from __future__ import annotations" in result


def test_prune_unused_imports_preserves_star_import():
    # Star imports cannot be pruned — kept as-is.
    source = "from os.path import *\n\ndef f():\n    pass\n"
    result = _prune_unused_imports(source)
    assert "from os.path import *" in result


def test_prune_unused_imports_removes_fully_unused_plain_import():
    # import whose name is never referenced is dropped entirely.
    source = "import sys\n\ndef f():\n    pass\n"
    result = _prune_unused_imports(source)
    assert "import sys" not in result


def test_prune_unused_imports_removes_fully_unused_from_import():
    # from-import whose names are never referenced is dropped entirely.
    source = "from typing import Dict\n\ndef f():\n    return 1\n"
    result = _prune_unused_imports(source)
    assert "from typing import" not in result


def test_prune_unused_imports_narrows_partial_from_import():
    # Only List is used — import narrowed to just List.
    source = "from typing import Dict, List\n\ndef f(x: List):\n    return x\n"
    result = _prune_unused_imports(source)
    assert "from typing import List" in result
    assert "Dict" not in result


def test_prune_unused_imports_narrows_plain_import():
    # import x, y where only y is used → narrowed to import y.
    source = "import os, sys\n\ndef f():\n    sys.exit()\n"
    result = _prune_unused_imports(source)
    assert "import sys" in result
    assert "os" not in result


def test_prune_unused_imports_multiline_import_collapsed():
    # Multi-line parenthesised import is collapsed to a single line.
    source = textwrap.dedent(
        """\
        from typing import (
            Dict,
            List,
        )

        def f(x: List):
            return x
        """
    )
    result = _prune_unused_imports(source)
    assert "from typing import List" in result
    assert "Dict" not in result
    assert "(\n" not in result


def test_prune_unused_imports_relative_import_narrowed():
    # Relative from-import is reconstructed with dots preserved.
    source = "from .utils import foo, bar\n\ndef f():\n    return foo()\n"
    result = _prune_unused_imports(source)
    assert "from .utils import foo" in result
    assert "bar" not in result


def test_prune_unused_imports_preserves_noqa_f401():
    # Imports marked "# noqa: F401" are intentional re-export stubs and must
    # never be pruned, even when the name is unused in the file body.
    source = (
        "from .utils import _helper  # fmt: skip # noqa: F401, E501\n"
        "\n"
        "def f():\n"
        "    pass\n"
    )
    result = _prune_unused_imports(source)
    assert "from .utils import _helper" in result


def test_prune_unused_imports_prunes_unused_without_noqa():
    # Without noqa, unused imports are still removed.
    source = "from .utils import _helper\n\ndef f():\n    pass\n"
    result = _prune_unused_imports(source)
    assert "from .utils import _helper" not in result


# ---------------------------------------------------------------------------
# generate_file_splits — import pruning integration
# ---------------------------------------------------------------------------


def test_generate_prunes_unused_names_from_multiname_import():
    # foo uses only List, not Dict; the new file's import should be narrowed.
    source = "from typing import Dict, List\n\ndef foo(x: List):\n    return x\n"
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert "from typing import List" in new_src
    assert "Dict" not in new_src


def test_generate_prunes_fully_unused_import_from_original():
    # import os is only used by foo; after foo migrates the original no longer
    # needs os, so the import should be removed.
    source = "import os\n\ndef foo():\n    os.getcwd()\n\ndef bar():\n    return 1\n"
    e_foo = _make_entity("foo", 3, 4)
    e_bar = _make_entity("bar", 6, 7)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "from .utils import foo" in result.original_source
    assert "import os" not in result.original_source
    assert "def bar():" in result.original_source


def test_generate_narrows_partial_unused_import_in_original():
    # foo uses Dict; bar uses List.  After foo migrates, Dict should be
    # stripped from the original's import while List is kept.
    source = (
        "from typing import Dict, List\n\n"
        "def foo(x: Dict):\n    return x\n\n"
        "def bar(x: List):\n    return x\n"
    )
    e_foo = _make_entity("foo", 3, 4)
    e_bar = _make_entity("bar", 6, 7)
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "from typing import List" in result.original_source
    assert "Dict" not in result.original_source


def test_generate_migrated_top_level_import_names_not_in_cross_file_imports():
    # Regression: when a TOP_LEVEL entity containing "from dataclasses import
    # dataclass" is migrated, the name "dataclass" must NOT be added to the
    # name→target-file map.  A FUNCTION entity in a separate new file that also
    # uses dataclass should get "from dataclasses import dataclass" (via
    # _find_needed_imports) rather than "from .constants import dataclass" (a
    # spurious cross-file import that would fail at runtime because constants.py
    # never exports dataclass).
    source = (
        "from dataclasses import dataclass\n\n"
        "_CONST = 42\n\n"
        "def make():\n    return dataclass\n"
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["dataclass", "_CONST"])
    e_make = _make_entity("make", 5, 6)
    c = _classified(entities=[e_block, e_make])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="constants.py"),
            GroupPlacement(group=["make"], target_file="utils.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    utils_src = result.new_files["utils.py"]
    # Must import dataclass from the stdlib, not from constants.py
    assert "from dataclasses import dataclass" in utils_src
    assert "from .constants import dataclass" not in utils_src


# ---------------------------------------------------------------------------
# _find_project_root
# ---------------------------------------------------------------------------


def test_find_project_root_finds_pyproject_toml(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    sub = tmp_path / "pkg" / "module.py"
    sub.parent.mkdir()
    sub.write_text("x = 1\n")
    assert _find_project_root(sub) == tmp_path


def test_find_project_root_finds_git(tmp_path):
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "module.py"
    sub.write_text("x = 1\n")
    assert _find_project_root(sub) == tmp_path


def test_find_project_root_called_with_directory(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    assert _find_project_root(tmp_path) == tmp_path


def test_find_project_root_not_found(tmp_path):
    # tmp_path is under /tmp which has no project markers → None.
    sub = tmp_path / "module.py"
    sub.write_text("x = 1\n")
    result = _find_project_root(sub)
    # If the test runner is inside a project that happens to include tmp_path
    # (unlikely but possible with in-tree pytest), just ensure the function
    # returns without crashing.  The important coverage is the happy path above.
    assert result is None or result.exists()


# ---------------------------------------------------------------------------
# _module_path_from_file
# ---------------------------------------------------------------------------


def test_module_path_from_file_success(tmp_path):
    f = tmp_path / "pkg" / "utils.py"
    f.parent.mkdir()
    f.write_text("")
    assert _module_path_from_file(tmp_path, f) == "pkg.utils"


def test_module_path_from_file_top_level(tmp_path):
    f = tmp_path / "module.py"
    f.write_text("")
    assert _module_path_from_file(tmp_path, f) == "module"


def test_module_path_from_file_not_under_root(tmp_path):
    other = tmp_path.parent / "other.py"
    assert _module_path_from_file(tmp_path, other) is None


# ---------------------------------------------------------------------------
# _collect_external_imported_names
# ---------------------------------------------------------------------------


def test_collect_external_imported_names_relative_path():
    # Non-absolute path → empty set (no scan).
    assert _collect_external_imported_names("relative/path.py") == set()


def test_collect_external_imported_names_nonexistent_file(tmp_path):
    # Absolute but non-existent → empty set.
    assert _collect_external_imported_names(str(tmp_path / "ghost.py")) == set()


def test_collect_external_imported_names_no_project_root(tmp_path):
    # File exists but no pyproject.toml/.git above it → empty set.
    f = tmp_path / "module.py"
    f.write_text("x = 1\n")
    # tmp_path is under /tmp which typically has no project markers.
    result = _collect_external_imported_names(str(f))
    # May or may not find a root depending on environment; we just verify no crash.
    assert isinstance(result, set)


def test_collect_external_imported_names_absolute_import(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    mod = pkg / "utils.py"
    mod.write_text("def _helper():\n    pass\n")
    caller = tmp_path / "tests" / "test_utils.py"
    caller.parent.mkdir()
    caller.write_text("from mypkg.utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_relative_import(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    mod = pkg / "utils.py"
    mod.write_text("def _helper():\n    pass\n")
    sibling = pkg / "other.py"
    sibling.write_text("from .utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_self_excluded(tmp_path):
    # The file being scanned is excluded from the search.
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("from module import _x\n")  # self-referential (ignored)
    result = _collect_external_imported_names(str(mod))
    assert "_x" not in result


def test_collect_external_imported_names_syntax_error_skipped(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("def _helper(): pass\n")
    bad = tmp_path / "bad.py"
    bad.write_text("def (invalid\n")
    good = tmp_path / "good.py"
    good.write_text("from module import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_non_matching_import_ignored(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("def _helper(): pass\n")
    other = tmp_path / "other.py"
    other.write_text("from different_module import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" not in result


def test_collect_external_imported_names_non_importfrom_nodes_skipped(tmp_path):
    # Caller file contains a plain `import` statement (not ImportFrom) mixed
    # with a matching `from … import`.  The plain import must be skipped without
    # crashing, and the matching ImportFrom still contributes to the result.
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "module.py"
    mod.write_text("def _helper(): pass\n")
    caller = tmp_path / "caller.py"
    caller.write_text("import os\nfrom module import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_deep_relative_import(tmp_path):
    # Two-level relative import: `from ..utils import _helper`
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "utils.py"
    mod.write_text("def _helper(): pass\n")
    sub = tmp_path / "pkg" / "sub" / "caller.py"
    sub.parent.mkdir(parents=True)
    sub.write_text("from ...utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    assert "_helper" in result


def test_collect_external_imported_names_init_py_at_root(tmp_path):
    # A bare __init__.py at the project root has no package prefix, so no
    # external caller can import from it by package path — returns empty set.
    (tmp_path / "pyproject.toml").write_text("")
    init_py = tmp_path / "__init__.py"
    init_py.write_text("class Foo: pass\n")
    result = _collect_external_imported_names(str(init_py))
    assert result == set()


def test_collect_external_imported_names_init_py(tmp_path):
    # When original_path is an __init__.py, callers import from the package
    # name (e.g. "mypkg.sub"), not "mypkg.sub.__init__".
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg" / "sub"
    pkg.mkdir(parents=True)
    init_py = pkg / "__init__.py"
    init_py.write_text("class Foo: pass\n")
    caller = tmp_path / "caller.py"
    caller.write_text("from mypkg.sub import Foo\n")
    result = _collect_external_imported_names(str(init_py))
    assert "Foo" in result


def test_collect_external_imported_names_init_py_relative_caller(tmp_path):
    # Relative import from sibling module targeting a package __init__.py.
    (tmp_path / "pyproject.toml").write_text("")
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    sub = pkg / "sub"
    sub.mkdir()
    (sub / "__init__.py").write_text("def _helper(): pass\n")
    sibling = pkg / "other.py"
    sibling.write_text("from .sub import _helper\n")
    result = _collect_external_imported_names(str(sub / "__init__.py"))
    assert "_helper" in result


def test_collect_external_imported_names_relative_level_too_deep(tmp_path):
    # Relative import that goes above the project root → skipped without crash.
    (tmp_path / "pyproject.toml").write_text("")
    mod = tmp_path / "utils.py"
    mod.write_text("def _helper(): pass\n")
    # A file at the top level trying to go up 5 packages (impossible).
    top = tmp_path / "top.py"
    top.write_text("from .....utils import _helper\n")
    result = _collect_external_imported_names(str(mod))
    # The over-deep import is silently skipped; no crash.
    assert isinstance(result, set)


# ---------------------------------------------------------------------------
# _add_re_exports — external_loads parameter
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _add_re_exports — reexport_mode parameter
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# generate_file_splits — private entity re-exported for external caller
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _add_re_exports — # fmt: skip # noqa: F401, E501 for pure re-export imports
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _prune_inline_redundant_imports
# ---------------------------------------------------------------------------


def test_prune_inline_syntax_error():
    # Unparseable source → returned unchanged.
    source = "def (invalid syntax"
    assert _prune_inline_redundant_imports(source) == source


def test_prune_inline_no_top_level_imports():
    # No module-level imports → nothing can be redundant, return unchanged.
    source = "def f():\n    from os import path\n    path.join('a', 'b')\n"
    assert _prune_inline_redundant_imports(source) == source


def test_prune_inline_no_inner_imports():
    # Only top-level imports, no function-body imports → unchanged.
    source = "import os\n\ndef f():\n    return os.getcwd()\n"
    assert _prune_inline_redundant_imports(source) == source


def test_prune_inline_no_redundancy():
    # Inner import brings in a different name than the top-level import.
    source = "import os\n\ndef f():\n    from sys import argv\n    return argv\n"
    assert _prune_inline_redundant_imports(source) == source


def test_prune_inline_removes_fully_redundant_from_import():
    # Top-level import covers all names in the inner from-import → remove it.
    source = textwrap.dedent(
        """\
        from unittest.mock import patch
        from mymod import Foo

        def test_thing():
            from mymod import Foo
            assert Foo()
        """
    )
    result = _prune_inline_redundant_imports(source)
    assert result.count("from mymod import Foo") == 1
    assert "assert Foo()" in result


def test_prune_inline_narrows_partially_redundant_from_import():
    # Only one of two inner names is already at top level → narrow the inner import.
    source = textwrap.dedent(
        """\
        from mymod import Foo

        def test_thing():
            from mymod import Foo, Bar
            assert Foo() and Bar()
        """
    )
    result = _prune_inline_redundant_imports(source)
    lines = result.splitlines()
    inner = [ln for ln in lines if "from mymod import" in ln and ln.startswith("    ")]
    assert len(inner) == 1
    assert "Bar" in inner[0]
    assert "Foo" not in inner[0]


def test_prune_inline_removes_fully_redundant_plain_import():
    # Inner ``import x`` where x is already available at top level → removed.
    source = textwrap.dedent(
        """\
        import os

        def f():
            import os
            return os.getcwd()
        """
    )
    result = _prune_inline_redundant_imports(source)
    assert result.count("import os") == 1


def test_prune_inline_narrows_partially_redundant_plain_import():
    # ``import os, sys`` inside function where os is already top-level → narrows to sys.
    source = textwrap.dedent(
        """\
        import os

        def f():
            import os, sys
            return sys.argv
        """
    )
    result = _prune_inline_redundant_imports(source)
    inner = [
        ln
        for ln in result.splitlines()
        if ln.strip().startswith("import") and ln.startswith("    ")
    ]
    assert len(inner) == 1
    assert "sys" in inner[0]
    assert "os" not in inner[0]


def test_prune_inline_preserves_indentation():
    # The narrowed replacement line must preserve the original indentation.
    source = textwrap.dedent(
        """\
        from mymod import Foo

        def test_thing():
            if True:
                from mymod import Foo, Bar
                assert Bar()
        """
    )
    result = _prune_inline_redundant_imports(source)
    inner = [
        ln
        for ln in result.splitlines()
        if "from mymod import" in ln and ln.startswith("        ")
    ]
    assert len(inner) == 1
    assert inner[0].startswith("        from mymod import Bar")


def test_prune_inline_preserves_type_checking_block():
    # Imports inside 'if TYPE_CHECKING:' must never be stripped even when the
    # same name is already imported at module level — removing them would leave
    # an empty (and syntactically invalid) if-block.
    source = textwrap.dedent(
        """\
        from typing import TYPE_CHECKING
        from mymod import Foo

        if TYPE_CHECKING:
            from mymod import Foo
        """
    )
    result = _prune_inline_redundant_imports(source)
    assert result == source


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


# ---------------------------------------------------------------------------
# _find_cross_file_imports — absolute import mode
# ---------------------------------------------------------------------------


def test_find_cross_file_imports_abs_pkg_package_prefix():
    # abs_pkg="tests" → "from tests.block_1 import _MODEL"
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "test_fn.py", abs_pkg="tests"
    )
    assert from_imports == ["from tests.block_1 import _MODEL"]
    assert module_imports == []
    assert rewrites == {}


def test_find_cross_file_imports_abs_pkg_root_level():
    # abs_pkg="" → "from block_1 import _MODEL" (no package prefix)
    entity_source_map = {"fn_a": "def fn_a():\n    return _MODEL\n"}
    name_to_target_file = {"_MODEL": "block_1.py"}
    from_imports, module_imports, rewrites = _find_cross_file_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "test_fn.py", abs_pkg=""
    )
    assert from_imports == ["from block_1 import _MODEL"]
    assert module_imports == []
    assert rewrites == {}


# ---------------------------------------------------------------------------
# _find_cross_file_type_checking_imports
# ---------------------------------------------------------------------------


def test_find_cross_file_type_checking_imports_basic():
    # _LLMAccumulator appears only in a quoted annotation in fn_a.
    # It lives in block_1.py — a TYPE_CHECKING import should be generated.
    entity_source_map = {
        "fn_a": 'def fn_a(x: Optional["_LLMAccumulator"]) -> None:\n    pass\n'
    }
    name_to_target_file = {"_LLMAccumulator": "block_1.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "placements.py"
    )
    assert result == ["from .block_1 import _LLMAccumulator"]


def test_find_cross_file_type_checking_imports_same_file_excluded():
    # _LLMAccumulator goes to the same target file — no import needed.
    entity_source_map = {
        "fn_a": 'def fn_a(x: Optional["_LLMAccumulator"]) -> None:\n    pass\n'
    }
    name_to_target_file = {"_LLMAccumulator": "placements.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "placements.py"
    )
    assert result == []


def test_find_cross_file_type_checking_imports_runtime_excluded():
    # _LLMAccumulator is used at runtime (not just annotation) — excluded.
    entity_source_map = {"fn_a": "def fn_a():\n    return _LLMAccumulator()\n"}
    name_to_target_file = {"_LLMAccumulator": "block_1.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "placements.py"
    )
    assert result == []


def test_find_cross_file_type_checking_imports_no_annotations():
    # No quoted annotations at all → empty result.
    entity_source_map = {"fn_a": "def fn_a():\n    pass\n"}
    name_to_target_file = {"_LLMAccumulator": "block_1.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"], entity_source_map, name_to_target_file, "placements.py"
    )
    assert result == []


def test_find_cross_file_type_checking_imports_not_in_map():
    # Referenced quoted name not in name_to_target_file → no import.
    entity_source_map = {"fn_a": 'def fn_a(x: "UnknownType") -> None:\n    pass\n'}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"], entity_source_map, {}, "placements.py"
    )
    assert result == []


def test_find_cross_file_type_checking_imports_top_level_var_excluded():
    # A name in top_level_var_names is skipped (handled separately).
    entity_source_map = {
        "fn_a": 'def fn_a(x: Optional["SAFE_MODE"]) -> None:\n    pass\n'
    }
    name_to_target_file = {"SAFE_MODE": "constants.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "placements.py",
        top_level_var_names={"SAFE_MODE"},
    )
    assert result == []


def test_find_cross_file_type_checking_imports_abs_pkg():
    # With abs_pkg set, use absolute import style.
    entity_source_map = {
        "fn_a": 'def fn_a(x: Optional["_LLMAccumulator"]) -> None:\n    pass\n'
    }
    name_to_target_file = {"_LLMAccumulator": "block_1.py"}
    result = _find_cross_file_type_checking_imports(
        ["fn_a"],
        entity_source_map,
        name_to_target_file,
        "test_fn.py",
        abs_pkg="tests",
    )
    assert result == ["from tests.block_1 import _LLMAccumulator"]


def test_find_cross_file_type_checking_imports_entity_not_in_map():
    # Entity not in entity_source_map → treated as empty, no imports.
    result = _find_cross_file_type_checking_imports(
        ["ghost"], {}, {"_X": "other.py"}, "placements.py"
    )
    assert result == []


# ---------------------------------------------------------------------------
# _abs_package_for_dir
# ---------------------------------------------------------------------------


def test_abs_package_for_dir_subdir(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_engine.py"
    test_file.touch()
    assert _abs_package_for_dir(str(test_file)) == "tests"


def test_abs_package_for_dir_root_level(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    test_file = tmp_path / "test_engine.py"
    test_file.touch()
    assert _abs_package_for_dir(str(test_file)) == ""


def test_abs_package_for_dir_no_project_root(monkeypatch):
    monkeypatch.setattr(
        "crispen.file_limiter.code_gen._find_project_root", lambda _p: None
    )
    assert _abs_package_for_dir("/some/random/path/test_engine.py") is None


def test_abs_package_for_dir_non_ancestor_root(tmp_path, monkeypatch):
    # Defensive branch: project root is not an ancestor of the file's directory.
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.setattr(
        "crispen.file_limiter.code_gen._find_project_root", lambda _p: other_dir
    )
    test_file = tmp_path / "tests" / "test_engine.py"
    test_file.parent.mkdir()
    test_file.touch()
    assert _abs_package_for_dir(str(test_file)) is None


# ---------------------------------------------------------------------------
# generate_file_splits — test file uses absolute imports
# ---------------------------------------------------------------------------


def test_generate_test_file_reexports_use_absolute_imports(tmp_path):
    # When the original is a test file, re-exports in the updated original
    # must use absolute imports so pytest can load the file.
    (tmp_path / "pyproject.toml").touch()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_engine.py"
    test_file.write_text("")

    source = "import os\n\ndef foo():\n    os.getcwd()\n"
    entity = _make_entity("foo", 3, 4)
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["foo"], target_file="test_helpers.py")])

    result = generate_file_splits(c, plan, source, str(test_file))

    assert "from tests.test_helpers import foo" in result.original_source
    assert "from .test_helpers import foo" not in result.original_source


def test_generate_test_file_cross_imports_use_absolute_imports(tmp_path):
    # Cross-file imports in generated test split files must also be absolute.
    (tmp_path / "pyproject.toml").touch()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_engine.py"
    test_file.write_text("")

    source = "_CONST = 42\n\ndef test_fn():\n    return _CONST\n"
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, ["_CONST"])
    e_fn = _make_entity("test_fn", 3, 4)
    c = _classified(entities=[e_block, e_fn])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="test_constants.py"),
            GroupPlacement(group=["test_fn"], target_file="test_fns.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, str(test_file))

    fn_src = result.new_files["test_fns.py"]
    # _CONST is a TOP_LEVEL variable that is never reassigned → plain absolute
    # from-import (idiomatic Python; module alias only needed if reassigned).
    assert "from tests.test_constants import _CONST" in fn_src
    assert "import tests.test_constants as test_constants" not in fn_src
    assert "test_constants._CONST" not in fn_src


# ---------------------------------------------------------------------------
# _add_re_exports — relative_from (subdir __init__.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# generate_file_splits — subdir_name parameter
# ---------------------------------------------------------------------------


def test_generate_file_splits_subdir_name_uses_init_as_original_basename():
    # When subdir_name="service", the dependency graph treats "service/__init__.py"
    # as the original file node.  Because main (public) is re-exported from
    # __init__, _extract_shared_helpers pulls helper into service/main.py to
    # break the __init__ → main → __init__ cycle.  The split must not abort.
    source = "def helper():\n    return 1\n\ndef main():\n    return helper()\n"
    e_helper = _make_entity("helper", 1, 2)
    e_main = _make_entity("main", 4, 5)
    c = _classified(entities=[e_helper, e_main])
    # Only main is migrated; helper stays in "original" (→ service/__init__.py).
    plan = _plan([GroupPlacement(group=["main"], target_file="service/main.py")])

    result = generate_file_splits(c, plan, source, "service.py", subdir_name="service")

    assert not result.abort
    # helper is extracted into service/main.py to break the re-export cycle.
    main_src = result.new_files["service/main.py"]
    assert "def helper" in main_src
    assert "def main" in main_src
    # Re-exports use the short relative prefix ".main", not ".service.main".
    assert "from .main import" in result.original_source
    assert "from .service.main" not in result.original_source


def test_generate_file_splits_subdir_name_re_exports_use_relative_prefix():
    # With subdir_name set (non-test), re-exports in the "original" source
    # (which becomes __init__.py) use ".utils" not ".service.utils".
    # target_file already has the "service/" prefix (added by runner.py).
    source = "def foo():\n    pass\n"
    e_foo = _make_entity("foo", 1, 2)
    c = _classified(entities=[e_foo])
    plan = _plan([GroupPlacement(group=["foo"], target_file="service/utils.py")])

    result = generate_file_splits(c, plan, source, "service.py", subdir_name="service")

    assert not result.abort
    assert "from .utils import foo" in result.original_source
    assert "from .service.utils" not in result.original_source


def test_generate_file_splits_subdir_name_cross_file_uses_relative():
    # In subdir mode, cross-file imports between new files use relative imports
    # even when the original file is a test (abs_pkg would normally apply).
    # NOTE: runner.py prefixes target_file with subdir_name before this call,
    # so target_files already include "svc/" here.
    source = "def helper():\n    return 1\n\ndef test_fn():\n    return helper()\n"
    e_helper = _make_entity("helper", 1, 2)
    e_test = _make_entity("test_fn", 4, 5)
    c = _classified(entities=[e_helper, e_test])
    plan = _plan(
        [
            GroupPlacement(group=["helper"], target_file="svc/helpers.py"),
            GroupPlacement(group=["test_fn"], target_file="svc/test_fns.py"),
        ]
    )

    # Use a path that looks like a test file so abs_pkg would normally be set.
    result = generate_file_splits(
        c, plan, source, "tests/test_svc.py", subdir_name="svc"
    )

    assert not result.abort
    # Cross-file import from test_fns.py to helpers.py should be relative.
    test_src = result.new_files["svc/test_fns.py"]
    assert "from .helpers import helper" in test_src


def test_generate_file_splits_test_subdir_nonmigrated_imports_from_original():
    # Non-migrated TOP_LEVEL variables (e.g. module-level constants) stay in
    # the original test file.  A new subfile that references a constant that is
    # never reassigned should use a plain ``from`` import (idiomatic Python);
    # module-alias access is only needed when the constant is mutated at runtime.
    source = "_CONFIG = 'val'\n\ndef test_fn():\n    return _CONFIG\n"
    # Use TOP_LEVEL kind so _extract_shared_helpers does not pull _CONFIG into
    # the new file (it only extracts FUNCTION/CLASS entities).
    e_config = Entity(EntityKind.TOP_LEVEL, "_CONFIG", 1, 1, ["_CONFIG"])
    e_test = _make_entity("test_fn", 3, 4)
    c = _classified(entities=[e_config, e_test])
    plan = _plan([GroupPlacement(group=["test_fn"], target_file="svc/test_fns.py")])

    result = generate_file_splits(
        c, plan, source, "tests/test_svc.py", subdir_name="svc"
    )

    assert not result.abort
    test_src = result.new_files["svc/test_fns.py"]
    # _CONFIG is never reassigned → plain from-import (no module alias).
    assert "from ..test_svc import _CONFIG" in test_src
    assert "from .. import test_svc" not in test_src
    assert "test_svc._CONFIG" not in test_src


def test_generate_file_splits_has_main_uses_filename_as_original_basename():
    # When has_main=True, original_basename is the flat filename ("service.py"),
    # not "service_lib/__init__.py".  Re-exports in the original file reference
    # the subdir modules directly (e.g. "from service_lib.utils import foo").
    source = "def foo():\n    pass\n\nif __name__ == '__main__':\n    foo()\n"
    e_foo = _make_entity("foo", 1, 2)
    c = _classified(entities=[e_foo])
    plan = _plan([GroupPlacement(group=["foo"], target_file="service_lib/utils.py")])

    result = generate_file_splits(
        c, plan, source, "service.py", subdir_name="service_lib", has_main=True
    )

    assert not result.abort
    # Re-export in original file uses the subdir module path.
    assert "service_lib" in result.original_source
    # No __init__.py is created by code_gen (the runner handles that decision).
    assert "service_lib/__init__.py" not in result.new_files


# ---------------------------------------------------------------------------
# _bump_relative_imports
# ---------------------------------------------------------------------------


def test_bump_relative_imports_single_dot():
    assert _bump_relative_imports("from .foo import bar") == "from ..foo import bar"


def test_bump_relative_imports_two_dots():
    assert _bump_relative_imports("from .. import baz") == "from ... import baz"


def test_bump_relative_imports_leaves_absolute():
    src = "import os\nfrom typing import List"
    assert _bump_relative_imports(src) == src


def test_bump_relative_imports_multiline():
    src = "from .a import x\nimport sys\nfrom ..b import y\n"
    result = _bump_relative_imports(src)
    assert "from ..a import x" in result
    assert "from ...b import y" in result
    assert "import sys" in result


def test_bump_relative_imports_n_two():
    assert _bump_relative_imports("from .. import foo", n=2) == "from .... import foo"


def test_bump_relative_imports_n_zero():
    src = "from .foo import bar"
    assert _bump_relative_imports(src, n=0) == src


# generate_file_splits — subdir_name bumps relative imports


def test_generate_file_splits_subdir_bumps_needed_imports():
    # In subdir-split mode, relative imports from the original file that appear
    # in new sub-files must be incremented by one level so they still resolve
    # correctly from inside the subdirectory package.
    source = "from .sibling import CONST\n\ndef foo():\n    return CONST\n"
    e_foo = _make_entity("foo", 3, 4)
    c = _classified(entities=[e_foo])
    plan = _plan([GroupPlacement(group=["foo"], target_file="service/utils.py")])

    result = generate_file_splits(c, plan, source, "service.py", subdir_name="service")

    assert not result.abort
    utils_src = result.new_files["service/utils.py"]
    assert "from ..sibling import CONST" in utils_src
    assert "from .sibling import CONST" not in utils_src


def test_generate_file_splits_subdir_bumps_init_imports():
    # In subdir-split mode, relative imports in the non-migrated original source
    # (which becomes subdir/__init__.py) must also be bumped by one level so
    # they still point at the correct modules from inside the package.
    source2 = (
        "from .. import llm_client\n"
        "from .base import Base\n\n"
        "def stayed():\n    return llm_client, Base\n\n"
        "def migrated():\n    pass\n"
    )
    e_stayed2 = _make_entity("stayed", 4, 5)
    e_migrated2 = _make_entity("migrated", 7, 8)
    c = _classified(entities=[e_stayed2, e_migrated2])
    plan = _plan([GroupPlacement(group=["migrated"], target_file="pkg/helpers.py")])

    result = generate_file_splits(c, plan, source2, "pkg.py", subdir_name="pkg")

    assert not result.abort
    init_src = result.original_source
    assert "from ... import llm_client" in init_src
    assert "from ..base import Base" in init_src
    assert "from .. import llm_client" not in init_src
    assert "from .base import Base" not in init_src


def test_generate_file_splits_subdir_bumps_two_levels_deep():
    # When the LLM places a new file two directories deep (e.g.
    # "pkg/pkg/core.py"), relative imports must be bumped by 2 dots, not 1.
    # This matches the real-world scenario where subdir_name="pkg" but the
    # advisor proposes "pkg/pkg/core.py" as a target.
    source = "from .. import llm_client\n\ndef func():\n    return llm_client\n"
    e_func = _make_entity("func", 3, 4)
    c = _classified(entities=[e_func])
    plan = _plan([GroupPlacement(group=["func"], target_file="pkg/pkg/core.py")])

    result = generate_file_splits(c, plan, source, "pkg.py", subdir_name="pkg")

    assert not result.abort
    core_src = result.new_files["pkg/pkg/core.py"]
    # 2 levels deep → original ".." becomes "...." (4 dots)
    assert "from .... import llm_client" in core_src
    assert "from .. import llm_client" not in core_src
    assert "from ... import llm_client" not in core_src


def test_generate_file_splits_subdir_injects_tc_import_for_nonmigrated_entity():
    # When a _block_N TOP_LEVEL entity that holds the `if TYPE_CHECKING:` block
    # is migrated to a sub-file, any non-migrated entity that references the
    # guarded name in a quoted annotation must receive the TYPE_CHECKING import
    # in the updated original (__init__.py).
    #
    # The original file has three entities:
    #   _block_1 — the TYPE_CHECKING block (migrated to sub.py)
    #   helper   — migrated to sub.py
    #   entry    — stays in __init__.py, references "MyConfig" in annotation
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .config import MyConfig\n"
        "\n"
        "def helper():\n"
        "    pass\n"
        "\n"
        "def entry(cfg: 'MyConfig') -> None:\n"
        "    helper()\n"
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, [])
    e_helper = _make_entity("helper", 5, 6)
    e_entry = _make_entity("entry", 8, 9)
    c = _classified(entities=[e_block, e_helper, e_entry])
    plan = _plan(
        [GroupPlacement(group=["_block_1", "helper"], target_file="pkg/sub.py")]
    )

    result = generate_file_splits(c, plan, source, "pkg.py", subdir_name="pkg")

    assert not result.abort
    init_src = result.original_source
    # The TYPE_CHECKING import must be injected and bumped for the new depth.
    assert "if TYPE_CHECKING:" in init_src
    assert "from ..config import MyConfig" in init_src


# ---------------------------------------------------------------------------
# _strip_top_level_import_lines
# ---------------------------------------------------------------------------


def test_strip_top_level_import_lines_removes_imports():
    src = "import os\nfrom typing import List\n\n_CONST = 1\n"
    result = _strip_top_level_import_lines(src)
    assert "import os" not in result
    assert "from typing import List" not in result
    assert "_CONST = 1" in result


def test_strip_top_level_import_lines_no_imports():
    src = "_CONST = 1\n"
    assert _strip_top_level_import_lines(src) == src


def test_strip_top_level_import_lines_syntax_error():
    src = "def (\n"
    assert _strip_top_level_import_lines(src) == src


def test_strip_top_level_import_lines_strips_type_checking_block():
    # `if TYPE_CHECKING:` blocks must be stripped so that their imports are
    # not emitted verbatim in sub-files (wrong path, wrong file).
    src = "if TYPE_CHECKING:\n" "    from .config import MyConfig\n" "\n" "_CONST = 1\n"
    result = _strip_top_level_import_lines(src)
    assert "TYPE_CHECKING" not in result
    assert "MyConfig" not in result
    assert "_CONST = 1" in result


# ---------------------------------------------------------------------------
# _extract_module_docstring
# ---------------------------------------------------------------------------


def test_extract_module_docstring_present():
    src = '"""My module."""\n\nimport os\n'
    assert _extract_module_docstring(src) == '"""My module."""'


def test_extract_module_docstring_absent():
    src = "import os\n\ndef foo():\n    pass\n"
    assert _extract_module_docstring(src) is None


def test_extract_module_docstring_syntax_error():
    assert _extract_module_docstring("def (\n") is None


def test_extract_module_docstring_non_string_expr():
    # First statement is an expression but not a string constant.
    src = "1 + 1\n\ndef foo():\n    pass\n"
    assert _extract_module_docstring(src) is None


# ---------------------------------------------------------------------------
# _strip_module_docstring
# ---------------------------------------------------------------------------


def test_strip_module_docstring_removes_docstring():
    src = '"""My module."""\n\n_CONST = 1\n'
    result = _strip_module_docstring(src)
    assert '"""My module."""' not in result
    assert "_CONST = 1" in result


def test_strip_module_docstring_no_docstring():
    src = "_CONST = 1\n"
    assert _strip_module_docstring(src) == src


def test_strip_module_docstring_syntax_error():
    src = "def (\n"
    assert _strip_module_docstring(src) == src


# ---------------------------------------------------------------------------
# _source_is_only_docstring
# ---------------------------------------------------------------------------


def test_source_is_only_docstring_true():
    assert _source_is_only_docstring('"""Just a docstring."""\n') is True


def test_source_is_only_docstring_with_other_content():
    assert _source_is_only_docstring('"""Doc."""\n\nimport os\n') is False


def test_source_is_only_docstring_no_docstring():
    assert _source_is_only_docstring("import os\n") is False


def test_source_is_only_docstring_syntax_error():
    assert _source_is_only_docstring("def (\n") is False


# ---------------------------------------------------------------------------
# generate_file_splits — TOP_LEVEL entity import deduplication
# ---------------------------------------------------------------------------


def test_generate_top_level_entity_imports_not_duplicated():
    # When a TOP_LEVEL entity source contains regular imports (e.g. `import os`)
    # those must NOT appear twice in the generated file: once from
    # _find_needed_imports and again from the entity source itself.
    source = "import os\n\n_CONST = os.sep\n\ndef foo():\n    return os.getcwd()\n"
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["os", "_CONST"])
    e_foo = _make_entity("foo", 5, 6)
    c = _classified(entities=[e_block, e_foo])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1", "foo"], target_file="utils.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert new_src.count("import os") == 1


# ---------------------------------------------------------------------------
# generate_file_splits — module docstring placement in subdir-split mode
# ---------------------------------------------------------------------------


def test_generate_subdir_module_docstring_goes_to_init():
    # In subdir-split mode the module docstring belongs in __init__.py, not
    # in the split-off child module.  Migrate the preamble entity (_block_1)
    # along with foo so the docstring is removed from the original source,
    # triggering the restore-to-__init__ logic.
    source = textwrap.dedent(
        """\
        \"\"\"Top-level module doc.\"\"\"

        import os

        def foo():
            return os.sep

        def bar():
            return foo()
        """
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["os"])
    e_foo = _make_entity("foo", 5, 6)
    e_bar = _make_entity("bar", 8, 9)
    c = _classified(entities=[e_block, e_foo, e_bar])
    plan = _plan(
        [GroupPlacement(group=["_block_1", "foo"], target_file="pkg/helpers.py")]
    )

    result = generate_file_splits(c, plan, source, "pkg.py", subdir_name="pkg")

    assert not result.abort
    init_src = result.original_source
    helpers_src = result.new_files["pkg/helpers.py"]
    # Docstring belongs in __init__.py.
    assert '"""Top-level module doc."""' in init_src
    # Docstring must NOT appear in the child module.
    assert '"""Top-level module doc."""' not in helpers_src


def test_generate_subdir_docstring_already_in_init_not_duplicated():
    # If the TOP_LEVEL entity stays in the original (not migrated), the
    # docstring remains in the updated source and must not be prepended again.
    source = textwrap.dedent(
        """\
        \"\"\"Top-level module doc.\"\"\"

        _CONST = 1

        def stayed():
            return _CONST

        def migrated():
            pass
        """
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["_CONST"])
    e_stayed = _make_entity("stayed", 5, 6)
    e_migrated = _make_entity("migrated", 8, 9)
    c = _classified(entities=[e_block, e_stayed, e_migrated])
    plan = _plan([GroupPlacement(group=["migrated"], target_file="pkg/helpers.py")])

    result = generate_file_splits(c, plan, source, "pkg.py", subdir_name="pkg")

    assert not result.abort
    init_src = result.original_source
    assert init_src.count('"""Top-level module doc."""') == 1


def test_generate_subdir_module_docstring_goes_to_test_init():
    # For test-file subdir splits the module docstring goes into
    # subdir/__init__.py, not into the re-export stub file.
    source = textwrap.dedent(
        """\
        \"\"\"Tests for the runner module.\"\"\"

        import os

        def test_foo():
            return os.sep

        def test_bar():
            return test_foo()
        """
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["os"])
    e_foo = _make_entity("test_foo", 5, 6)
    e_bar = _make_entity("test_bar", 8, 9)
    c = _classified(entities=[e_block, e_foo, e_bar])
    plan = _plan(
        [GroupPlacement(group=["_block_1", "test_foo"], target_file="svc/test_foo.py")]
    )

    result = generate_file_splits(
        c, plan, source, "tests/test_svc.py", subdir_name="svc"
    )

    assert not result.abort
    init_src = result.new_files["svc/__init__.py"]
    child_src = result.new_files["svc/test_foo.py"]
    updated_src = result.original_source
    # Docstring belongs in __init__.py.
    assert '"""Tests for the runner module."""' in init_src
    # Docstring must NOT appear in the child test file or the stub file.
    assert '"""Tests for the runner module."""' not in child_src
    assert '"""Tests for the runner module."""' not in updated_src


def test_generate_subdir_test_docstring_only_remaining_clears_original():
    # Regression: when a test-file subdir split migrates all entities and the
    # only thing left in the original is the module docstring (a TOP_LEVEL
    # entity that is not migrated by _remove_entity_lines), the docstring must
    # be routed to __init__.py and the original file must be cleared for
    # deletion by the engine.
    source = textwrap.dedent(
        """\
        \"\"\"Tests for the widget module.
        Covers edge cases.
        \"\"\"

        def test_alpha():
            pass

        def test_beta():
            pass
        """
    )
    # The module docstring is a TOP_LEVEL entity spanning lines 1-3.
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, [])
    e_alpha = _make_entity("test_alpha", 5, 6)
    e_beta = _make_entity("test_beta", 8, 9)
    c = _classified(entities=[e_block, e_alpha, e_beta])
    # Only the test functions are migrated; the TOP_LEVEL entity stays.
    plan = _plan(
        [
            GroupPlacement(group=["test_alpha"], target_file="widget/test_alpha.py"),
            GroupPlacement(group=["test_beta"], target_file="widget/test_beta.py"),
        ]
    )

    result = generate_file_splits(
        c, plan, source, "tests/test_widget.py", subdir_name="widget"
    )

    assert not result.abort
    # Docstring must end up in __init__.py.
    init_src = result.new_files["widget/__init__.py"]
    assert '"""Tests for the widget module.' in init_src
    # Original source must be empty so the engine deletes it.
    assert result.original_source == ""


def test_generate_subdir_docstring_not_stripped_from_non_subdir_split():
    # Outside subdir-split mode, a TOP_LEVEL entity's docstring is preserved
    # in the new file (only imports are stripped, not docstrings).
    source = '"""Module doc."""\n\nimport os\n\ndef foo():\n    return os.sep\n'
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["os"])
    e_foo = _make_entity("foo", 5, 6)
    c = _classified(entities=[e_block, e_foo])
    plan = _plan([GroupPlacement(group=["_block_1", "foo"], target_file="utils.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    new_src = result.new_files["utils.py"]
    assert '"""Module doc."""' in new_src


def test_generate_vacuous_top_level_block_alone_creates_no_file():
    # Regression: a TOP_LEVEL block that is only import statements plus the
    # module docstring loses all of its content once imports are re-derived
    # and (in subdir mode) the docstring is routed to __init__.py. Placed
    # alone in its own target file, this used to leave behind a new file
    # containing only the boilerplate header, e.g. a bare
    # "from __future__ import annotations". No such file should be created;
    # the entity should stay in the original file instead.
    source = (
        "from __future__ import annotations\n\n"
        '"""Module doc."""\n\n'
        "import os\n\n\n"
        "def helper():\n"
        "    return os.sep\n"
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 5, ["annotations", "os"])
    e_helper = _make_entity("helper", 8, 9)
    c = _classified(entities=[e_block, e_helper])
    plan = _plan(
        [
            GroupPlacement(group=["_block_1"], target_file="foo/meta.py"),
            GroupPlacement(group=["helper"], target_file="foo/helpers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "pkg/foo.py", subdir_name="foo")

    assert not result.abort
    assert "foo/meta.py" not in result.new_files
    # The docstring is neither lost nor left in an empty file — it stays in
    # the (reconstructed) original source alongside the re-export.
    assert '"""Module doc."""' in result.original_source
    helpers_src = result.new_files["foo/helpers.py"]
    assert helpers_src.count("from __future__ import annotations") == 1


def test_generate_vacuous_top_level_block_grouped_with_content_still_moves():
    # When a vacuous TOP_LEVEL block shares a target file with real content,
    # it is still treated as migrated — its re-derived imports and (in
    # subdir mode) docstring routing are still needed by that file.
    source = (
        '"""Module doc."""\n\n'
        "import os\n\n\n"
        "def helper():\n"
        "    return os.sep\n"
    )
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 3, ["os"])
    e_helper = _make_entity("helper", 6, 7)
    c = _classified(entities=[e_block, e_helper])
    plan = _plan(
        [GroupPlacement(group=["_block_1", "helper"], target_file="foo/helpers.py")]
    )

    result = generate_file_splits(c, plan, source, "pkg/foo.py", subdir_name="foo")

    assert not result.abort
    helpers_src = result.new_files["foo/helpers.py"]
    assert "import os" in helpers_src
    assert "def helper" in helpers_src


# ---------------------------------------------------------------------------
# _is_test_name
# ---------------------------------------------------------------------------


def test_is_test_name_test_class():
    assert _is_test_name("TestFoo") is True


def test_is_test_name_test_function():
    assert _is_test_name("test_bar") is True


def test_is_test_name_non_test():
    assert _is_test_name("helper") is False
    assert _is_test_name("Foo") is False
    assert _is_test_name("_test_private") is False


# ---------------------------------------------------------------------------
# _is_pytest_fixture
# ---------------------------------------------------------------------------


def test_is_pytest_fixture_syntax_error():
    assert _is_pytest_fixture("def (") is False


def test_is_pytest_fixture_empty_body():
    # Empty source → empty tree body → not a fixture.
    assert _is_pytest_fixture("") is False


def test_is_pytest_fixture_class_node():
    # Class definition is not a FunctionDef → returns False.
    assert _is_pytest_fixture("class Foo:\n    pass\n") is False


def test_is_pytest_fixture_no_decorator():
    assert _is_pytest_fixture("def client():\n    pass\n") is False


def test_is_pytest_fixture_bare_name():
    # @fixture (plain name, no call)
    src = "@fixture\ndef client():\n    pass\n"
    assert _is_pytest_fixture(src) is True


def test_is_pytest_fixture_bare_name_called():
    # @fixture() (called with no args)
    src = "@fixture()\ndef client():\n    pass\n"
    assert _is_pytest_fixture(src) is True


def test_is_pytest_fixture_attribute():
    # @pytest.fixture (attribute access, no call)
    src = "@pytest.fixture\ndef client():\n    pass\n"
    assert _is_pytest_fixture(src) is True


def test_is_pytest_fixture_attribute_called():
    # @pytest.fixture(scope="session")
    src = '@pytest.fixture(scope="session")\ndef client():\n    pass\n'
    assert _is_pytest_fixture(src) is True


def test_is_pytest_fixture_non_matching_decorator():
    # @other_decorator — Name but id != "fixture"; not an Attribute.
    src = "@other_decorator\ndef client():\n    pass\n"
    assert _is_pytest_fixture(src) is False


# ---------------------------------------------------------------------------
# _split_cross_imports_by_test
# ---------------------------------------------------------------------------


def test_split_cross_imports_by_test_pure_non_test():
    non_test, test_named = _split_cross_imports_by_test(["from .foo import helper"])
    assert non_test == ["from .foo import helper"]
    assert test_named == []


def test_split_cross_imports_by_test_pure_test():
    non_test, test_named = _split_cross_imports_by_test(
        ["from .foo import TestFoo, test_bar"]
    )
    assert non_test == []
    assert test_named == ["from .foo import TestFoo, test_bar"]


def test_split_cross_imports_by_test_mixed():
    non_test, test_named = _split_cross_imports_by_test(
        ["from .foo import TestFoo, helper, test_bar"]
    )
    assert non_test == ["from .foo import helper"]
    assert test_named == ["from .foo import TestFoo, test_bar"]


def test_split_cross_imports_by_test_plain_import_passthrough():
    # Plain "import x" lines (no "from") pass through to non_test unchanged.
    non_test, test_named = _split_cross_imports_by_test(["import os"])
    assert non_test == ["import os"]
    assert test_named == []


# ---------------------------------------------------------------------------
# _inject_inline_imports
# ---------------------------------------------------------------------------


def test_inject_inline_imports_into_function():
    src = "def foo():\n    return 1\n"
    result = _inject_inline_imports(src, ["from .bar import Baz"])
    assert result == "def foo():\n    from .bar import Baz\n    return 1\n"


def test_inject_inline_imports_skips_docstring():
    src = 'def foo():\n    """Doc."""\n    return 1\n'
    result = _inject_inline_imports(src, ["from .bar import Baz"])
    assert (
        result == 'def foo():\n    """Doc."""\n    from .bar import Baz\n    return 1\n'
    )


def test_inject_inline_imports_into_class():
    src = "class Foo:\n    x = 1\n"
    result = _inject_inline_imports(src, ["from .bar import Baz"])
    assert result == "class Foo:\n    from .bar import Baz\n    x = 1\n"


def test_inject_inline_imports_toplevel_noop():
    # TOP_LEVEL entity (bare if-statement): no body scope, returns unchanged.
    src = "if True:\n    pass\n"
    result = _inject_inline_imports(src, ["from .bar import Baz"])
    assert result == src


def test_inject_inline_imports_empty_list_noop():
    src = "def foo():\n    pass\n"
    assert _inject_inline_imports(src, []) == src


def test_inject_inline_imports_syntax_error_noop():
    src = "def (invalid"
    assert _inject_inline_imports(src, ["from .x import Y"]) == src


def test_inject_inline_imports_empty_source_noop():
    # Empty source parses to empty tree.body — returns unchanged.
    assert _inject_inline_imports("", ["from .x import Y"]) == ""


def test_inject_inline_imports_only_docstring_injects_after():
    # Function with only a docstring — inserts after docstring (at body[0] line)
    # since len(body) == 1.
    src = 'def foo():\n    """Only doc."""\n'
    result = _inject_inline_imports(src, ["from .bar import Baz"])
    assert result == 'def foo():\n    from .bar import Baz\n    """Only doc."""\n'


# ---------------------------------------------------------------------------
# _find_main_block_entity
# ---------------------------------------------------------------------------


def test_find_main_block_entity_present():
    from crispen.file_limiter.entity_parser import parse_entities

    source = textwrap.dedent(
        """\
        def run():
            pass

        if __name__ == "__main__":
            run()
        """
    )
    entities = parse_entities(source)
    esmap = {e.name: source.splitlines(keepends=True) for e in entities}
    # Rebuild entity_source_map properly
    lines = source.splitlines(keepends=True)
    esmap = {
        e.name: "".join(lines[e.start_line - 1 : e.end_line]).rstrip() for e in entities
    }
    result = _find_main_block_entity(entities, esmap)
    assert result is not None
    assert result.startswith("_block_")


def test_find_main_block_entity_absent():
    from crispen.file_limiter.entity_parser import parse_entities

    source = "def foo():\n    pass\n"
    entities = parse_entities(source)
    lines = source.splitlines(keepends=True)
    esmap = {
        e.name: "".join(lines[e.start_line - 1 : e.end_line]).rstrip() for e in entities
    }
    assert _find_main_block_entity(entities, esmap) is None


def test_find_main_block_entity_syntax_error_skipped():
    from crispen.file_limiter.entity_parser import Entity, EntityKind

    # Entity whose source is invalid Python: should be skipped gracefully.
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, [])
    result = _find_main_block_entity([entity], {"_block_1": "def (invalid"})
    assert result is None


# ---------------------------------------------------------------------------
# _find_main_direct_callees
# ---------------------------------------------------------------------------


def test_find_main_direct_callees_basic():
    src = 'if __name__ == "__main__":\n    run_tests()\n'
    callees = _find_main_direct_callees(src, {"run_tests", "other"})
    assert callees == {"run_tests"}


def test_find_main_direct_callees_not_in_entity_names():
    src = 'if __name__ == "__main__":\n    unknown()\n'
    callees = _find_main_direct_callees(src, {"run_tests"})
    assert callees == set()


def test_find_main_direct_callees_syntax_error():
    assert _find_main_direct_callees("def (invalid", {"foo"}) == set()


def test_find_main_direct_callees_no_main_block():
    src = "run_tests()\n"
    assert _find_main_direct_callees(src, {"run_tests"}) == set()


# ---------------------------------------------------------------------------
# _inject_inline_test_imports_original
# ---------------------------------------------------------------------------


def test_inject_inline_test_imports_original_basic():
    source = textwrap.dedent(
        """\
        def runner():
            TestFoo()
        """
    )
    migrated = {"TestFoo": "sub/test_foo.py"}
    result = _inject_inline_test_imports_original(
        source, migrated, abs_pkg="pkg.tests", original_basename="test_orig.py"
    )
    assert "from pkg.tests.sub.test_foo import TestFoo" in result
    # Import appears inside the function body, not before the def line.
    lines = result.splitlines()
    def_idx = next(i for i, l in enumerate(lines) if l.startswith("def runner"))
    import_idx = next(i for i, l in enumerate(lines) if "import TestFoo" in l)
    assert import_idx > def_idx


def test_inject_inline_test_imports_original_skips_docstring():
    source = textwrap.dedent(
        """\
        def runner():
            \"\"\"Run tests.\"\"\"
            TestFoo()
        """
    )
    migrated = {"TestFoo": "sub/test_foo.py"}
    result = _inject_inline_test_imports_original(
        source, migrated, abs_pkg="tests", original_basename="test_orig.py"
    )
    lines = result.splitlines()
    doc_idx = next(i for i, l in enumerate(lines) if '"""Run tests."""' in l)
    import_idx = next(i for i, l in enumerate(lines) if "import TestFoo" in l)
    assert import_idx > doc_idx


def test_inject_inline_test_imports_original_no_reference():
    source = "def runner():\n    pass\n"
    migrated = {"TestFoo": "sub/test_foo.py"}
    result = _inject_inline_test_imports_original(
        source, migrated, abs_pkg="tests", original_basename="test_orig.py"
    )
    assert result == source


def test_inject_inline_test_imports_original_empty_map():
    source = "def runner():\n    TestFoo()\n"
    result = _inject_inline_test_imports_original(
        source, {}, abs_pkg="tests", original_basename="test_orig.py"
    )
    assert result == source


def test_inject_inline_test_imports_original_syntax_error():
    result = _inject_inline_test_imports_original(
        "def (invalid",
        {"TestFoo": "sub/test_foo.py"},
        abs_pkg="tests",
        original_basename="test_orig.py",
    )
    assert result == "def (invalid"


def test_inject_inline_test_imports_original_relative_import():
    source = "def runner():\n    TestFoo()\n"
    migrated = {"TestFoo": "sub/test_foo.py"}
    result = _inject_inline_test_imports_original(
        source, migrated, abs_pkg=None, original_basename="test_orig.py"
    )
    assert "from .sub.test_foo import TestFoo" in result


def test_inject_inline_test_imports_original_unreferenced_symbol_skipped():
    # Function references `helper` (not test-named) and `other_func`, neither
    # of which is in migrated_test_symbols — the false branch of `if tfile:`.
    source = "def runner():\n    helper()\n    other_func()\n"
    migrated = {"TestFoo": "sub/test_foo.py"}
    result = _inject_inline_test_imports_original(
        source, migrated, abs_pkg="tests", original_basename="test_orig.py"
    )
    assert result == source


# ---------------------------------------------------------------------------
# generate_file_splits — shebang handling
# ---------------------------------------------------------------------------


def test_generate_shebang_stripped_from_new_file():
    # Shebang on line 1 should NOT appear in generated new files.
    source = "#!/usr/bin/env python3\n\ndef foo():\n    pass\n\ndef bar():\n    foo()\n"
    e_foo = Entity(EntityKind.FUNCTION, "foo", 3, 4, ["foo"])
    e_bar = Entity(EntityKind.FUNCTION, "bar", 6, 7, ["bar"])
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["bar"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert "#!/usr/bin/env python3" not in result.new_files["helpers.py"]


def test_generate_shebang_preserved_in_original_when_entity_migrated():
    # When the entity owning line 1 (with shebang comment) is migrated,
    # the shebang must be restored at the top of the original file.
    source = "#!/usr/bin/env python3\ndef foo():\n    pass\n\ndef bar():\n    pass\n"
    e_foo = Entity(EntityKind.FUNCTION, "foo", 1, 3, ["foo"])
    e_bar = Entity(EntityKind.FUNCTION, "bar", 5, 6, ["bar"])
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["foo"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert result.original_source.startswith("#!/usr/bin/env python3\n")
    assert "#!/usr/bin/env python3" not in result.new_files["helpers.py"]


def test_generate_shebang_preserved_when_not_migrated():
    # When the shebang entity stays in the original, shebang remains at top.
    source = "#!/usr/bin/env python3\ndef foo():\n    pass\n\ndef bar():\n    pass\n"
    e_foo = Entity(EntityKind.FUNCTION, "foo", 1, 3, ["foo"])
    e_bar = Entity(EntityKind.FUNCTION, "bar", 5, 6, ["bar"])
    c = _classified(entities=[e_foo, e_bar])
    plan = _plan([GroupPlacement(group=["bar"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    assert result.original_source.startswith("#!/usr/bin/env python3\n")


# ---------------------------------------------------------------------------
# generate_file_splits — __main__ sticky behaviour
# ---------------------------------------------------------------------------


def test_generate_main_block_stays_in_original():
    source = textwrap.dedent(
        """\
        def run():
            pass

        def other():
            pass

        if __name__ == "__main__":
            run()
        """
    )
    e_run = Entity(EntityKind.FUNCTION, "run", 1, 2, ["run"])
    e_other = Entity(EntityKind.FUNCTION, "other", 4, 5, ["other"])
    e_main = Entity(EntityKind.TOP_LEVEL, "_block_7", 7, 8, [])
    c = _classified(entities=[e_run, e_other, e_main])
    # Plan tries to migrate run + __main__ block and other.
    plan = _plan(
        [
            GroupPlacement(group=["run", "_block_7"], target_file="helpers.py"),
            GroupPlacement(group=["other"], target_file="helpers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    # __main__ block stays in original.
    assert 'if __name__ == "__main__"' in result.original_source
    assert 'if __name__ == "__main__"' not in result.new_files.get("helpers.py", "")


def test_generate_main_callee_stays_in_original():
    source = textwrap.dedent(
        """\
        def run():
            pass

        def other():
            pass

        if __name__ == "__main__":
            run()
        """
    )
    e_run = Entity(EntityKind.FUNCTION, "run", 1, 2, ["run"])
    e_other = Entity(EntityKind.FUNCTION, "other", 4, 5, ["other"])
    e_main = Entity(EntityKind.TOP_LEVEL, "_block_7", 7, 8, [])
    c = _classified(entities=[e_run, e_other, e_main])
    # Plan tries to migrate run (the direct callee of __main__).
    plan = _plan(
        [
            GroupPlacement(group=["run"], target_file="helpers.py"),
            GroupPlacement(group=["other"], target_file="helpers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    # run() is a direct __main__ callee — must stay in original.
    assert "def run():" in result.original_source
    # other() is not a callee — may be migrated.
    assert "helpers.py" in result.new_files


# ---------------------------------------------------------------------------
# generate_file_splits — test-named symbol inline imports
# ---------------------------------------------------------------------------


def test_generate_test_named_cross_import_inlined():
    # TestHelper migrates to helpers.py; runner stays in original and
    # references TestHelper — the import must be injected inside runner's body.
    source = textwrap.dedent(
        """\
        class TestHelper:
            def test_x(self):
                pass

        def runner():
            TestHelper()
        """
    )
    e_cls = Entity(EntityKind.CLASS, "TestHelper", 1, 3, ["TestHelper"])
    e_run = Entity(EntityKind.FUNCTION, "runner", 5, 6, ["runner"])
    c = _classified(entities=[e_cls, e_run])
    plan = _plan([GroupPlacement(group=["TestHelper"], target_file="helpers.py")])

    result = generate_file_splits(c, plan, source, "big.py")

    orig = result.original_source
    # No module-level re-export of TestHelper.
    lines = orig.splitlines()
    top_level_import_lines = [
        ln for ln in lines if ln.startswith("from") and "TestHelper" in ln
    ]
    assert top_level_import_lines == []
    # Import appears inside runner's body.
    assert "    from .helpers import TestHelper" in orig


def test_generate_test_named_inline_not_applied_to_toplevel_entity():
    # A TOP_LEVEL entity referencing a test-named symbol falls back to
    # module-level import since it has no body scope to inject into.
    source = textwrap.dedent(
        """\
        class TestHelper:
            def test_x(self):
                pass

        _inst = TestHelper()
        """
    )
    e_cls = Entity(EntityKind.CLASS, "TestHelper", 1, 3, ["TestHelper"])
    e_block = Entity(EntityKind.TOP_LEVEL, "_block_5", 5, 5, ["_inst"])
    c = _classified(entities=[e_cls, e_block])
    plan = _plan(
        [GroupPlacement(group=["TestHelper", "_block_5"], target_file="helpers.py")]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    # TestHelper and _block_5 were migrated together — no cross-file issue here.
    # Test just ensures no crash and the file is produced.
    assert "helpers.py" in result.new_files


def test_generate_test_named_inlined_in_function_in_new_file():
    # TestA goes to file_a.py; func_b (which calls TestA) goes to file_b.py.
    # The cross-file import of TestA into file_b.py should be injected inline
    # inside func_b's body rather than at the top of file_b.py.
    source = textwrap.dedent(
        """\
        class TestA:
            def test_x(self):
                pass

        def func_b():
            TestA()
        """
    )
    e_a = Entity(EntityKind.CLASS, "TestA", 1, 3, ["TestA"])
    e_b = Entity(EntityKind.FUNCTION, "func_b", 5, 6, ["func_b"])
    c = _classified(entities=[e_a, e_b])
    plan = _plan(
        [
            GroupPlacement(group=["TestA"], target_file="file_a.py"),
            GroupPlacement(group=["func_b"], target_file="file_b.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    file_b = result.new_files["file_b.py"]
    lines = file_b.splitlines()
    # No module-level import of TestA.
    assert not any(ln.startswith("from") and "TestA" in ln for ln in lines)
    # Inline import inside func_b.
    assert "    from .file_a import TestA" in file_b


def test_generate_toplevel_entity_in_new_file_test_import_falls_back_to_module_level():
    # A TOP_LEVEL entity in a new file that references a test-named symbol
    # from another new file: no function body to inject into, falls back to
    # module-level import.  Two TOP_LEVEL entities referencing the same
    # test name exercise the dedup path on the second.
    source = textwrap.dedent(
        """\
        class TestA:
            def test_x(self):
                pass

        _inst1 = TestA()

        def _sep():
            pass

        _inst2 = TestA()
        """
    )
    e_a = Entity(EntityKind.CLASS, "TestA", 1, 3, ["TestA"])
    e_b1 = Entity(EntityKind.TOP_LEVEL, "_block_5", 5, 5, ["_inst1"])
    e_sep = Entity(EntityKind.FUNCTION, "_sep", 7, 8, ["_sep"])
    e_b2 = Entity(EntityKind.TOP_LEVEL, "_block_10", 10, 10, ["_inst2"])
    c = _classified(entities=[e_a, e_b1, e_sep, e_b2])
    plan = _plan(
        [
            GroupPlacement(group=["TestA"], target_file="file_a.py"),
            GroupPlacement(
                group=["_block_5", "_sep", "_block_10"], target_file="file_b.py"
            ),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    file_b = result.new_files["file_b.py"]
    # Module-level import is acceptable for TOP_LEVEL entities (no body scope).
    assert "TestA" in file_b
    # Dedup: the same import appears only once despite two TOP_LEVEL entities
    # both referencing TestA.
    assert file_b.count("import TestA") == 1


def test_generate_cross_import_dedup_across_entities():
    # helper goes to helpers.py; foo and bar both go to workers.py and both
    # reference helper — the cross-file import should appear once (dedup).
    source = textwrap.dedent(
        """\
        def helper():
            pass

        def foo():
            helper()

        def bar():
            helper()
        """
    )
    e_h = Entity(EntityKind.FUNCTION, "helper", 1, 2, ["helper"])
    e_foo = Entity(EntityKind.FUNCTION, "foo", 4, 5, ["foo"])
    e_bar = Entity(EntityKind.FUNCTION, "bar", 7, 8, ["bar"])
    c = _classified(entities=[e_h, e_foo, e_bar])
    plan = _plan(
        [
            GroupPlacement(group=["helper"], target_file="helpers.py"),
            GroupPlacement(group=["foo", "bar"], target_file="workers.py"),
        ]
    )

    result = generate_file_splits(c, plan, source, "big.py")

    workers = result.new_files["workers.py"]
    # "from .helpers import helper" should appear exactly once.
    assert workers.count("import helper") == 1


# ---------------------------------------------------------------------------
# generate_file_splits — pytest conftest routing
# ---------------------------------------------------------------------------


def test_generate_pytest_conftest_disabled_no_conftest():
    # Default (pytest_conftest=False): fixture goes to assigned file, re-exported.
    src = "@pytest.fixture\ndef client():\n    pass\n"
    entity = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["client"], target_file="fixtures.py")])

    result = generate_file_splits(c, plan, src, "test_big.py")

    assert "fixtures.py" in result.new_files
    assert "conftest.py" not in result.new_files
    assert "client" in result.new_files["fixtures.py"]


def test_generate_pytest_conftest_subdir_routes_to_subdir_conftest():
    # With pytest_conftest=True AND subdir_name set, fixtures go to
    # <subdir>/conftest.py (not the parent conftest.py).  This prevents
    # multiple test files in the same directory from conflicting when they
    # each have a fixture of the same name.
    src = "@pytest.fixture\ndef client():\n    pass\n"
    entity = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["client"], target_file="expr/fixtures.py")])

    result = generate_file_splits(
        c, plan, src, "test_big.py", subdir_name="expr", pytest_conftest=True
    )

    assert "expr/conftest.py" in result.new_files
    assert "def client():" in result.new_files["expr/conftest.py"]
    assert "conftest.py" not in result.new_files  # parent conftest untouched
    assert "import client" not in result.original_source


def test_generate_pytest_conftest_subdir_fixture_referenced_in_remaining_goes_to_parent():  # noqa: E501
    # When a fixture is migrated from a subdir split but its name still appears
    # in entities that remain in the original file, route it to the parent
    # conftest.py (not the subdir conftest) so those tests can find it.
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        def test_big(client):
            pass
        """
    )
    e_client = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    e_test = Entity(EntityKind.FUNCTION, "test_big", 5, 6, ["test_big"])
    # Only the fixture is migrated; the test stays in the original.
    c = _classified(entities=[e_client, e_test])
    plan = _plan([GroupPlacement(group=["client"], target_file="expr/fixtures.py")])

    result = generate_file_splits(
        c, plan, src, "test_big.py", subdir_name="expr", pytest_conftest=True
    )

    # Fixture goes to parent conftest.py, not the subdir one.
    assert "conftest.py" in result.new_files
    assert "def client():" in result.new_files["conftest.py"]
    assert "expr/conftest.py" not in result.new_files
    # No import of client back into the original.
    assert "import client" not in result.original_source


def test_generate_pytest_conftest_subdir_fixture_overrides_parent_conftest(tmp_path):
    # When the fixture is referenced in remaining source AND the parent conftest
    # already has a fixture with the same name (the module was overriding it),
    # the fixture is *copied* (not moved) to the subdir conftest so migrated
    # tests get the override; the entity also stays in the original file so
    # the original test discovers it from its own module.
    parent_conftest = tmp_path / "conftest.py"
    parent_conftest.write_text(
        "@pytest.fixture\ndef client():\n    return 'base'\n", encoding="utf-8"
    )
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            return 'override'

        def test_big(client):
            pass
        """
    )
    e_client = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    e_test = Entity(EntityKind.FUNCTION, "test_big", 5, 6, ["test_big"])
    c = _classified(entities=[e_client, e_test])
    plan = _plan([GroupPlacement(group=["client"], target_file="expr/fixtures.py")])
    original_path = str(tmp_path / "test_big.py")

    result = generate_file_splits(
        c, plan, src, original_path, subdir_name="expr", pytest_conftest=True
    )

    # Fixture goes to subdir conftest for migrated tests.
    assert "expr/conftest.py" in result.new_files
    assert "def client():" in result.new_files["expr/conftest.py"]
    assert "return 'override'" in result.new_files["expr/conftest.py"]
    # Parent conftest is NOT modified (would drop the override via merge).
    assert "conftest.py" not in result.new_files
    # Fixture stays in original file so the original test finds the override.
    assert "def client():" in result.original_source
    assert "return 'override'" in result.original_source
    # No re-export import injected.
    assert "import client" not in result.original_source


def test_generate_pytest_conftest_subdir_parent_conftest_imports_only(tmp_path):
    # When parent conftest exists but contains only imports (no function defs),
    # no conflict is detected and the fixture routes to parent conftest normally.
    parent_conftest = tmp_path / "conftest.py"
    parent_conftest.write_text("import pytest\n", encoding="utf-8")
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        def test_big(client):
            pass
        """
    )
    e_client = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    e_test = Entity(EntityKind.FUNCTION, "test_big", 5, 6, ["test_big"])
    c = _classified(entities=[e_client, e_test])
    plan = _plan([GroupPlacement(group=["client"], target_file="expr/fixtures.py")])
    original_path = str(tmp_path / "test_big.py")

    result = generate_file_splits(
        c, plan, src, original_path, subdir_name="expr", pytest_conftest=True
    )

    # No conflict in parent conftest → fixture routes to parent conftest.
    assert "conftest.py" in result.new_files
    assert "def client():" in result.new_files["conftest.py"]
    assert "expr/conftest.py" not in result.new_files


def test_generate_pytest_conftest_subdir_parent_conftest_syntax_error(tmp_path):
    # When parent conftest has a syntax error, the OSError/SyntaxError handler
    # silently ignores it (no names loaded), so no conflict is detected and the
    # fixture routes to parent conftest normally.
    parent_conftest = tmp_path / "conftest.py"
    parent_conftest.write_text("def (broken syntax", encoding="utf-8")
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        def test_big(client):
            pass
        """
    )
    e_client = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    e_test = Entity(EntityKind.FUNCTION, "test_big", 5, 6, ["test_big"])
    c = _classified(entities=[e_client, e_test])
    plan = _plan([GroupPlacement(group=["client"], target_file="expr/fixtures.py")])
    original_path = str(tmp_path / "test_big.py")

    result = generate_file_splits(
        c, plan, src, original_path, subdir_name="expr", pytest_conftest=True
    )

    # Unreadable parent conftest → no conflict detected → parent conftest.
    assert "conftest.py" in result.new_files
    assert "def client():" in result.new_files["conftest.py"]
    assert "expr/conftest.py" not in result.new_files


def test_generate_pytest_conftest_fixture_goes_to_conftest():
    # With pytest_conftest=True, fixture entity lands in conftest.py, not the
    # LLM-assigned file, and no re-export import appears in the original.
    src = "@pytest.fixture\ndef client():\n    pass\n"
    entity = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["client"], target_file="fixtures.py")])

    result = generate_file_splits(c, plan, src, "test_big.py", pytest_conftest=True)

    assert "conftest.py" in result.new_files
    assert "def client():" in result.new_files["conftest.py"]
    # No import of client back into the original (no F401/F811).
    assert "import client" not in result.original_source
    # The LLM-assigned file is dropped (all entities redirected).
    assert "fixtures.py" not in result.new_files


def test_generate_pytest_conftest_mixed_group_splits():
    # Fixture goes to conftest.py; non-fixture stays in the assigned file.
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        def helper():
            pass
        """
    )
    e_client = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    e_helper = Entity(EntityKind.FUNCTION, "helper", 5, 6, ["helper"])
    c = _classified(entities=[e_client, e_helper])
    plan = _plan([GroupPlacement(group=["client", "helper"], target_file="support.py")])

    result = generate_file_splits(c, plan, src, "test_big.py", pytest_conftest=True)

    assert "conftest.py" in result.new_files
    assert "def client():" in result.new_files["conftest.py"]
    assert "support.py" in result.new_files
    assert "def helper():" in result.new_files["support.py"]
    assert "import client" not in result.original_source


def test_generate_pytest_conftest_no_fixtures_no_conftest():
    # pytest_conftest=True but no fixture entities → no conftest.py created.
    src = "def helper():\n    pass\n"
    entity = Entity(EntityKind.FUNCTION, "helper", 1, 2, ["helper"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["helper"], target_file="support.py")])

    result = generate_file_splits(c, plan, src, "test_big.py", pytest_conftest=True)

    assert "conftest.py" not in result.new_files
    assert "support.py" in result.new_files


def test_generate_pytest_conftest_prepends_existing(tmp_path):
    # When conftest.py already exists on disk, its content is prepended.
    existing = tmp_path / "conftest.py"
    existing.write_text(
        "# existing fixture\ndef prior():\n    pass\n", encoding="utf-8"
    )

    src = "@pytest.fixture\ndef client():\n    pass\n"
    entity = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["client"], target_file="fixtures.py")])
    original_path = str(tmp_path / "test_big.py")

    result = generate_file_splits(c, plan, src, original_path, pytest_conftest=True)

    conftest_src = result.new_files["conftest.py"]
    assert "# existing fixture" in conftest_src
    assert "def prior():" in conftest_src
    assert "def client():" in conftest_src
    # Existing content should come first.
    assert conftest_src.index("prior") < conftest_src.index("client")


def test_generate_pytest_conftest_name_conflict_keeps_in_target(tmp_path):
    # When conftest.py already defines a function with the same name as the
    # fixture being routed, the fixture stays in its LLM-assigned target file
    # instead of being dropped by _merge_conftest_sources.  This preserves the
    # entity in the split output so that _verify_preservation passes.
    existing = tmp_path / "conftest.py"
    existing.write_text(
        "@pytest.fixture\nasync def client():\n    return 'old'\n", encoding="utf-8"
    )

    src = "@pytest.fixture\nasync def client():\n    return 'new'\n"
    entity = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["client"], target_file="fixtures.py")])
    original_path = str(tmp_path / "test_big.py")

    result = generate_file_splits(c, plan, src, original_path, pytest_conftest=True)

    # Fixture must appear in the output — in the LLM-assigned file, not conftest.
    assert "fixtures.py" in result.new_files
    assert "def client():" in result.new_files["fixtures.py"]
    # conftest.py should not be created/modified (no new fixtures were routed there).
    assert "conftest.py" not in result.new_files


def test_generate_pytest_conftest_name_conflict_mixed_group(tmp_path):
    # When a placement group contains both a conftest-conflict fixture AND a
    # regular function, the fixture is excluded from re-exports but the regular
    # function is still re-exported.  This covers the branch that rebuilds the
    # GroupPlacement with only the non-conflict names.
    existing = tmp_path / "conftest.py"
    existing.write_text(
        "@pytest.fixture\ndef client():\n    return 'old'\n", encoding="utf-8"
    )

    src = (
        "@pytest.fixture\ndef client():\n    return 'new'\n\n"
        "def helper():\n    pass\n"
    )
    e_client = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    e_helper = Entity(EntityKind.FUNCTION, "helper", 5, 6, ["helper"])
    c = _classified(entities=[e_client, e_helper])
    plan = _plan([GroupPlacement(group=["client", "helper"], target_file="helpers.py")])
    original_path = str(tmp_path / "test_big.py")

    result = generate_file_splits(c, plan, src, original_path, pytest_conftest=True)

    # Both entities migrate to helpers.py.
    assert "helpers.py" in result.new_files
    assert "def client():" in result.new_files["helpers.py"]
    assert "def helper():" in result.new_files["helpers.py"]
    # helper is re-exported (public non-fixture); client is not (conftest conflict).
    assert "helper" in result.original_source
    assert "client" not in result.original_source


def test_generate_pytest_conftest_unreadable_conftest_falls_through(tmp_path):
    # When conftest.py exists but has a syntax error, the OSError/SyntaxError
    # handler silently ignores it and routes the fixture to conftest normally.
    existing = tmp_path / "conftest.py"
    existing.write_text("def (broken syntax", encoding="utf-8")

    src = "@pytest.fixture\ndef client():\n    pass\n"
    entity = Entity(EntityKind.FUNCTION, "client", 1, 3, ["client"])
    c = _classified(entities=[entity])
    plan = _plan([GroupPlacement(group=["client"], target_file="fixtures.py")])
    original_path = str(tmp_path / "test_big.py")

    result = generate_file_splits(c, plan, src, original_path, pytest_conftest=True)

    # With unreadable conftest, routing proceeds normally → fixture goes to conftest.
    assert "conftest.py" in result.new_files
    assert "def client():" in result.new_files["conftest.py"]


# ---------------------------------------------------------------------------
# _file_has_only_fixtures
# ---------------------------------------------------------------------------


def test_file_has_only_fixtures_syntax_error():
    assert _file_has_only_fixtures("def (") is False


def test_file_has_only_fixtures_empty():
    assert _file_has_only_fixtures("") is False


def test_file_has_only_fixtures_no_fixture():
    # Regular function only — not a fixture.
    assert _file_has_only_fixtures("def helper():\n    pass\n") is False


def test_file_has_only_fixtures_with_test_function():
    # Has both a fixture and a test function → not fixture-only.
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        def test_foo(client):
            pass
        """
    )
    assert _file_has_only_fixtures(src) is False


def test_file_has_only_fixtures_with_test_class():
    # Has both a fixture and a Test class → not fixture-only.
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        class TestFoo:
            pass
        """
    )
    assert _file_has_only_fixtures(src) is False


def test_file_has_only_fixtures_with_non_fixture_function():
    # Has a fixture and a plain helper function → not fixture-only.
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        def helper():
            pass
        """
    )
    assert _file_has_only_fixtures(src) is False


def test_file_has_only_fixtures_with_class():
    # Has a fixture and a regular class → not fixture-only.
    src = textwrap.dedent(
        """\
        @pytest.fixture
        def client():
            pass

        class Config:
            pass
        """
    )
    assert _file_has_only_fixtures(src) is False


def test_file_has_only_fixtures_single_fixture():
    # Just a fixture and an import → fixture-only.
    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        def client():
            pass
        """
    )
    assert _file_has_only_fixtures(src) is True


def test_file_has_only_fixtures_multiple_fixtures():
    # Multiple fixtures with no tests → fixture-only.
    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        def client():
            pass

        @pytest.fixture
        def db():
            pass
        """
    )
    assert _file_has_only_fixtures(src) is True


def test_file_has_only_fixtures_async_fixture():
    # Async fixture → fixture-only.
    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        async def client():
            pass
        """
    )
    assert _file_has_only_fixtures(src) is True


def test_file_has_only_fixtures_with_docstring():
    # Module docstring + fixture → fixture-only (docstring is allowed).
    src = textwrap.dedent(
        """\
        \"\"\"Module docstring.\"\"\"

        import pytest

        @pytest.fixture
        def client():
            pass
        """
    )
    assert _file_has_only_fixtures(src) is True


# ---------------------------------------------------------------------------
# generate_file_splits: fixture-only stranded test file cleanup
# ---------------------------------------------------------------------------


def test_generate_stays_fixture_emptied_when_tests_migrated():
    # When a fixture "stays" in the original test file but all tests migrate
    # out, the original becomes fixture-only → route fixture to conftest.py
    # and empty the original so the engine deletes it.
    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        def client():
            pass

        def test_foo(client):
            pass
        """
    )
    e_fixture = Entity(EntityKind.FUNCTION, "client", 3, 5, ["client"])
    e_test = Entity(EntityKind.FUNCTION, "test_foo", 7, 8, ["test_foo"])
    c = _classified(entities=[e_fixture, e_test])
    # Only test_foo is migrated; client "stays" in original.
    plan = _plan(
        [GroupPlacement(group=["test_foo"], target_file="expression/test_foo.py")]
    )

    result = generate_file_splits(c, plan, src, "test_big.py", pytest_conftest=True)

    # Original should be empty (engine will delete it).
    assert result.original_source == ""
    # Fixture should be routed to conftest.py.
    assert "conftest.py" in result.new_files
    assert "def client():" in result.new_files["conftest.py"]


def test_generate_stays_fixture_merged_with_existing_conftest(tmp_path):
    # If conftest.py already exists on disk (e.g. same fixture already there),
    # the merge deduplicates so the fixture is not repeated.
    existing = tmp_path / "conftest.py"
    existing.write_text(
        "import pytest\n\n\n@pytest.fixture\ndef client():\n    pass\n",
        encoding="utf-8",
    )

    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        def client():
            pass

        def test_foo(client):
            pass
        """
    )
    e_fixture = Entity(EntityKind.FUNCTION, "client", 3, 5, ["client"])
    e_test = Entity(EntityKind.FUNCTION, "test_foo", 7, 8, ["test_foo"])
    c = _classified(entities=[e_fixture, e_test])
    plan = _plan(
        [GroupPlacement(group=["test_foo"], target_file="expression/test_foo.py")]
    )
    original_path = str(tmp_path / "test_expression.py")

    result = generate_file_splits(c, plan, src, original_path, pytest_conftest=True)

    assert result.original_source == ""
    # Fixture should appear exactly once in conftest.py (deduplicated).
    assert result.new_files["conftest.py"].count("def client():") == 1


def test_generate_stays_fixture_not_emptied_when_tests_remain():
    # If test functions still remain in the original, the fixture-only cleanup
    # does NOT trigger — the file should keep both fixture and test.
    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        def client():
            pass

        def test_foo(client):
            pass

        def test_bar(client):
            pass
        """
    )
    e_fixture = Entity(EntityKind.FUNCTION, "client", 3, 5, ["client"])
    e_test_foo = Entity(EntityKind.FUNCTION, "test_foo", 7, 8, ["test_foo"])
    e_test_bar = Entity(EntityKind.FUNCTION, "test_bar", 10, 11, ["test_bar"])
    c = _classified(entities=[e_fixture, e_test_foo, e_test_bar])
    # Only test_foo migrates; test_bar stays → original still has a test.
    plan = _plan(
        [GroupPlacement(group=["test_foo"], target_file="expression/test_foo.py")]
    )

    result = generate_file_splits(c, plan, src, "test_big.py", pytest_conftest=True)

    # Original should still contain the remaining test and fixture.
    assert "def test_bar" in result.original_source
    assert result.original_source != ""


def test_generate_stays_fixture_not_emptied_when_conftest_disabled():
    # When pytest_conftest=False, stranded fixtures are left in the original.
    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        def client():
            pass

        def test_foo(client):
            pass
        """
    )
    e_fixture = Entity(EntityKind.FUNCTION, "client", 3, 5, ["client"])
    e_test = Entity(EntityKind.FUNCTION, "test_foo", 7, 8, ["test_foo"])
    c = _classified(entities=[e_fixture, e_test])
    plan = _plan(
        [GroupPlacement(group=["test_foo"], target_file="expression/test_foo.py")]
    )

    result = generate_file_splits(c, plan, src, "test_big.py", pytest_conftest=False)

    # Original keeps the fixture (conftest routing disabled).
    assert "def client():" in result.original_source
    assert "conftest.py" not in result.new_files


def test_generate_stays_fixture_merged_with_already_written_conftest():
    # If conftest.py was already written by this same split run (e.g. another
    # entity was already routed there), merge into it rather than reading disk.
    src = textwrap.dedent(
        """\
        import pytest

        @pytest.fixture
        def client():
            pass

        @pytest.fixture
        def db():
            pass

        def test_foo(client):
            pass
        """
    )
    e_client = Entity(EntityKind.FUNCTION, "client", 3, 5, ["client"])
    e_db = Entity(EntityKind.FUNCTION, "db", 7, 9, ["db"])
    e_test = Entity(EntityKind.FUNCTION, "test_foo", 11, 12, ["test_foo"])
    c = _classified(entities=[e_client, e_db, e_test])
    # db migrates (and goes to conftest.py via pytest routing); test_foo migrates;
    # client stays but is then stranded.
    plan = _plan(
        [
            GroupPlacement(group=["db"], target_file="fixtures.py"),
            GroupPlacement(group=["test_foo"], target_file="expression/test_foo.py"),
        ]
    )

    result = generate_file_splits(c, plan, src, "test_big.py", pytest_conftest=True)

    assert result.original_source == ""
    conftest_src = result.new_files["conftest.py"]
    # Both migrated db and stranded client fixtures should be in conftest.
    assert "def db():" in conftest_src
    assert "def client():" in conftest_src


# ---------------------------------------------------------------------------
# _merge_conftest_sources
# ---------------------------------------------------------------------------


def test_merge_conftest_sources_deduplicates_imports():
    # Imports that already exist are not repeated.
    existing = "import pytest\n\n\n@pytest.fixture\ndef prior():\n    pass\n"
    new = "import pytest\n\n\n@pytest.fixture\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert result.count("import pytest") == 1


def test_merge_conftest_sources_deduplicates_functions():
    # A function already in existing is not appended again.
    existing = "@pytest.fixture\ndef client():\n    return 1\n"
    new = "@pytest.fixture\ndef client():\n    return 2\n"
    result = _merge_conftest_sources(existing, new)
    assert result.count("def client():") == 1
    assert "return 1" in result
    assert "return 2" not in result


def test_merge_conftest_sources_appends_new_fixture():
    # A new fixture not in existing is appended.
    existing = "@pytest.fixture\ndef prior():\n    pass\n"
    new = "@pytest.fixture\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert "def prior():" in result
    assert "def client():" in result
    assert result.index("prior") < result.index("client")


def test_merge_conftest_sources_no_changes_returns_existing():
    # When nothing new to add, return existing unchanged.
    existing = "import pytest\n\n\n@pytest.fixture\ndef client():\n    pass\n"
    new = "import pytest\n\n\n@pytest.fixture\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert result == existing


def test_merge_conftest_sources_inserts_new_imports_before_functions():
    # New imports are inserted after existing imports but before functions — no E402.
    existing = "import pytest\n\n\n@pytest.fixture\ndef prior():\n    pass\n"
    new = "import asyncio\n\n\n@pytest.fixture\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert "import asyncio" in result
    assert "def client():" in result
    # Imports must come before the first function definition.
    assert result.index("import asyncio") < result.index("def prior():")


def test_merge_conftest_sources_syntax_error_fallback():
    # Falls back to simple concatenation when existing cannot be parsed.
    existing = "def (broken"
    new = "import pytest\n"
    result = _merge_conftest_sources(existing, new)
    assert "def (broken" in result
    assert "import pytest" in result


def test_merge_conftest_sources_preserves_comments():
    # Comments in the existing conftest are preserved.
    existing = "# shared fixtures\nimport pytest\n\n\ndef prior():\n    pass\n"
    new = "@pytest.fixture\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert "# shared fixtures" in result
    assert "def client():" in result


def test_merge_conftest_sources_from_import_dedup():
    # from-style imports are also deduplicated via the _import_key F: path.
    existing = "from conftest import setup\n\n\ndef prior():\n    pass\n"
    new = "from conftest import setup\n\n\ndef client():\n    pass\n"
    result = _merge_conftest_sources(existing, new)
    assert result.count("from conftest import setup") == 1
    assert "def client():" in result


def test_merge_conftest_sources_only_new_imports_no_defs():
    # When only new imports are added but no new functions, ends with newline.
    existing = "import pytest\n\n\ndef prior():\n    pass\n"
    new = "import asyncio\n"
    result = _merge_conftest_sources(existing, new)
    assert "import asyncio" in result
    assert result.endswith("\n")
    # No duplicate function definition appended.
    assert result.count("def prior():") == 1


def test_merge_conftest_sources_non_import_non_def_in_new():
    # Bare statements (assignments, expressions) in new_content are silently ignored.
    existing = "def prior():\n    pass\n"
    new = "X = 42\n"
    result = _merge_conftest_sources(existing, new)
    # Nothing to import or define → returns existing unchanged.
    assert result == existing


# ---------------------------------------------------------------------------
# _strip_orphaned_section_headers
# ---------------------------------------------------------------------------


def test_strip_orphaned_3line_header_at_eof():
    """3-line block with no code after it is removed."""
    div = "# ---\n"
    source = "def foo():\n    pass\n\n\n" + div + "# Old Section\n" + div
    result = _strip_orphaned_section_headers(source)
    assert "# Old Section" not in result
    assert "def foo():" in result


def test_strip_orphaned_single_line_header_at_eof():
    """Single-line header with no code after it is removed."""
    source = "def foo():\n    pass\n\n# --- Removed ---\n"
    result = _strip_orphaned_section_headers(source)
    assert "# --- Removed ---" not in result
    assert "def foo():" in result


def test_strip_not_orphaned_3line_header():
    """3-line block followed by substantive code is kept."""
    div = "# ---\n"
    source = div + "# Helpers\n" + div + "\n\ndef helper():\n    pass\n"
    result = _strip_orphaned_section_headers(source)
    assert "# Helpers" in result
    assert "def helper():" in result


def test_strip_not_orphaned_single_line_header():
    """Single-line header followed by substantive code is kept."""
    source = "# --- Tools ---\n\ndef tool():\n    pass\n"
    result = _strip_orphaned_section_headers(source)
    assert "# --- Tools ---" in result


def test_strip_orphaned_header_followed_only_by_another_header():
    """Header followed only by another header (and then nothing) — both orphaned."""
    source = "def foo():\n" "    pass\n" "\n" "# --- First ---\n" "# --- Second ---\n"
    result = _strip_orphaned_section_headers(source)
    assert "# --- First ---" not in result
    assert "# --- Second ---" not in result
    assert "def foo():" in result


def test_strip_partial_orphan():
    """Only the header with no code after it is removed; the other stays."""
    source = (
        "# --- Active ---\n" "\n" "def foo():\n" "    pass\n" "\n" "# --- Empty ---\n"
    )
    result = _strip_orphaned_section_headers(source)
    assert "# --- Active ---" in result
    assert "# --- Empty ---" not in result


def test_strip_no_headers_returns_unchanged():
    """Source with no section headers is returned unchanged."""
    source = "def foo():\n    pass\n"
    assert _strip_orphaned_section_headers(source) == source


def test_strip_all_headers_have_content():
    """When every header has content below it, source is returned unchanged."""
    source = (
        "# --- A ---\n"
        "\n"
        "def a():\n"
        "    pass\n"
        "\n"
        "# --- B ---\n"
        "\n"
        "def b():\n"
        "    pass\n"
    )
    result = _strip_orphaned_section_headers(source)
    assert "# --- A ---" in result
    assert "# --- B ---" in result


def test_strip_equals_single_line_header_orphaned():
    """=== style orphaned header is also removed."""
    source = "def foo():\n    pass\n\n# === OLD SECTION ===\n"
    result = _strip_orphaned_section_headers(source)
    assert "# === OLD SECTION ===" not in result


# ---------------------------------------------------------------------------
# _normalize_blank_lines
# ---------------------------------------------------------------------------


def test_normalize_blank_lines_strips_leading_blanks():
    """Leading blank lines are removed (prevents E303 at top of file)."""
    source = "\n\n\ndef foo():\n    pass\n"
    result = _normalize_blank_lines(source)
    assert result.startswith("def foo():")


def test_normalize_blank_lines_collapses_excess_top_level():
    """4+ consecutive newlines between top-level defs collapse to 3."""
    source = "def foo():\n    pass\n\n\n\n\ndef bar():\n    pass\n"
    result = _normalize_blank_lines(source)
    assert "\n\n\n\n" not in result
    assert "def foo():" in result
    assert "def bar():" in result


def test_normalize_blank_lines_collapses_body_blanks():
    """2+ blank lines inside an indented body collapse to 1 (prevents E303 in body)."""
    source = "def foo():\n    x = 1\n\n\n    y = 2\n"
    result = _normalize_blank_lines(source)
    assert "\n\n\n    y" not in result
    assert "\n\n    y" in result


def test_normalize_blank_lines_empty_source():
    """Whitespace-only source returns empty string."""
    assert _normalize_blank_lines("\n\n\n") == ""


def test_normalize_blank_lines_trailing_newline():
    """Result always ends with exactly one newline."""
    source = "x = 1\n\n\n"
    result = _normalize_blank_lines(source)
    assert result.endswith("\n")
    assert not result.endswith("\n\n")


def test_normalize_blank_lines_preserves_multiline_string_body_blanks():
    """Blank lines inside a multi-line string literal are never collapsed.

    Regression: _EXCESS_BLANK_BODY_RE matched \\n{3,}(?=[ \\t]) inside
    triple-quoted strings, collapsing 2 blank lines before an indented line
    to 1 (e.g. stored source-code fixtures in tests).
    """
    # The triple-quoted string contains 2 blank lines before an indented `def`.
    # That produces the sequence \\n\\n\\n        def inside the raw source,
    # which _EXCESS_BLANK_BODY_RE would collapse to \\n\\n        def.
    source = textwrap.dedent(
        """\
        import textwrap
        def foo():
            src = textwrap.dedent(
                \"\"\"\\
                @dataclass
                class _SplitTask:
                    pass


                def _find_free_vars():
                    x = 1
                \"\"\"
            )
        """
    )
    result = _normalize_blank_lines(source)
    # Two blank lines before the indented `def` inside the string must survive.
    # After outer textwrap.dedent the string content has 8-space indentation.
    assert "\n\n\n        def _find_free_vars" in result


def test_normalize_blank_lines_still_collapses_excess_outside_strings():
    """Blank-line collapsing still fires for code outside string literals."""
    source = "def foo():\n    x = 1\n\n\n    y = 2\n"
    result = _normalize_blank_lines(source)
    assert "\n\n\n    y" not in result
    assert "\n\n    y" in result


# ---------------------------------------------------------------------------
# _multiline_string_ranges
# ---------------------------------------------------------------------------


def test_multiline_string_ranges_triple_quoted():
    """Detects a triple-quoted string spanning multiple lines."""
    source = 'x = """\nhello\n"""\n'
    ranges = _multiline_string_ranges(source)
    assert len(ranges) == 1
    start, end = ranges[0]
    assert source[start:end] == '"""\nhello\n"""'


def test_multiline_string_ranges_single_line_string_ignored():
    """Single-line strings (no literal newline) are not returned."""
    source = 'x = "hello\\n"\n'
    ranges = _multiline_string_ranges(source)
    assert ranges == []


def test_multiline_string_ranges_no_strings():
    """Returns empty list when there are no string literals."""
    source = "x = 1 + 2\n"
    ranges = _multiline_string_ranges(source)
    assert ranges == []


def test_multiline_string_ranges_invalid_source():
    """Falls back to empty list on tokenization error."""
    # Unterminated string triggers TokenError.
    source = 'x = """\nhello\n'
    ranges = _multiline_string_ranges(source)
    assert ranges == []


# ---------------------------------------------------------------------------
# _sub_skip_strings
# ---------------------------------------------------------------------------


def test_sub_skip_strings_does_not_touch_string_content():
    """Pattern match inside a multi-line string is not substituted."""
    import re

    pattern = re.compile(r"\n{3,}(?=[ \t])")
    source = 'def f():\n    s = """\n    a\n\n\n    b\n    """\n'
    result = _sub_skip_strings(pattern, "\n\n", source)
    # The sequence inside the string must survive unchanged.
    assert "\n\n\n    b" in result


def test_sub_skip_strings_applies_outside_strings():
    """Pattern match outside string literals is substituted normally."""
    import re

    pattern = re.compile(r"\n{3,}(?=[ \t])")
    source = "def f():\n    x = 1\n\n\n    y = 2\n"
    result = _sub_skip_strings(pattern, "\n\n", source)
    assert "\n\n\n    y" not in result
    assert "\n\n    y" in result


def test_sub_skip_strings_no_strings_falls_through():
    """When there are no multi-line strings the plain .sub() path is taken."""
    import re

    pattern = re.compile(r"x")
    source = "x = 1\n"
    result = _sub_skip_strings(pattern, "y", source)
    assert result == "y = 1\n"


# ---------------------------------------------------------------------------
# _strip_orphaned_indented_comments
# ---------------------------------------------------------------------------


def test_strip_orphaned_indented_comments_removes_orphan():
    """Indented comment at module level (outside any AST node) is removed."""
    source = "\n\n    # This comment was left behind after function removal\n"
    result = _strip_orphaned_indented_comments(source)
    assert "# This comment was left behind" not in result


def test_strip_orphaned_indented_comments_keeps_inside_function():
    """Indented comment inside a function body is preserved."""
    source = "def foo():\n    # normal comment\n    pass\n"
    result = _strip_orphaned_indented_comments(source)
    assert "# normal comment" in result


def test_strip_orphaned_indented_comments_keeps_module_level_comment():
    """Non-indented module-level comment is preserved."""
    source = "# module comment\ndef foo():\n    pass\n"
    result = _strip_orphaned_indented_comments(source)
    assert "# module comment" in result


def test_strip_orphaned_indented_comments_syntax_error():
    """SyntaxError in source returns source unchanged."""
    source = "    # orphaned\ndef f(: pass\n"
    result = _strip_orphaned_indented_comments(source)
    assert result == source
