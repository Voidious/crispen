from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import (
    _bump_relative_imports,
    _find_main_block_entity,
    _find_main_direct_callees,
    generate_file_splits,
)
from crispen.file_limiter.entity_parser import Entity, EntityKind
from .test_shared_helpers_extraction import _classified, _make_entity, _plan


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

    # Entity whose source is invalid Python: should be skipped gracefully.
    entity = Entity(EntityKind.TOP_LEVEL, "_block_1", 1, 1, [])
    result = _find_main_block_entity([entity], {"_block_1": "def (invalid"})
    assert result is None


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
