from __future__ import annotations
import textwrap
from crispen.file_limiter.advisor import GroupPlacement
from crispen.file_limiter.code_gen import generate_file_splits
from crispen.file_limiter.entity_parser import Entity, EntityKind
from ..test_shared_helpers import _classified, _make_entity, _plan


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
