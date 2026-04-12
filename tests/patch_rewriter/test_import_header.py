from __future__ import annotations
from crispen.patch_rewriter import (
    _CgIndex,
    _FLContext,
    _cg_build_index,
    _cg_file_to_module_and_package,
    _cg_parse_imports,
    _get_external_import_names,
    _import_header,
)


def test_import_header_stops_before_def():
    src = "import os\nfrom x import y\n\ndef foo():\n    pass\n"
    assert _import_header(src) == "import os\nfrom x import y\n"


def test_import_header_stops_before_class():
    src = "import os\n\nclass Foo:\n    pass\n"
    assert _import_header(src) == "import os\n"


def test_import_header_stops_before_async_def():
    src = "import os\nasync def foo(): pass\n"
    assert _import_header(src) == "import os\n"


def test_import_header_no_defs_returns_all():
    src = "import os\nfrom x import y\n"
    assert _import_header(src) == "import os\nfrom x import y\n"


def test_import_header_empty_source():
    assert _import_header("") == ""


def test_import_header_strips_trailing_blanks():
    src = "import os\n\n\ndef foo(): pass\n"
    assert _import_header(src) == "import os\n"


def test_get_external_import_names_absolute():
    src = "from pkg import Foo\nimport os\n"
    names = _get_external_import_names(src)
    assert "Foo" in names
    assert "os" in names


def test_get_external_import_names_level1_skipped():
    src = "from .sub import Bar\nfrom . import Baz\n"
    names = _get_external_import_names(src)
    assert names == set()


def test_get_external_import_names_level2_included():
    src = "from ..pkg import Foo\nfrom ...llm_client import call_with_tool\n"
    names = _get_external_import_names(src)
    assert "Foo" in names
    assert "call_with_tool" in names


def test_get_external_import_names_star_import_skipped():
    src = "from pkg import *\n"
    names = _get_external_import_names(src)
    assert names == set()


def test_get_external_import_names_asname():
    src = "import libcst as cst\nfrom pkg import Foo as F\n"
    names = _get_external_import_names(src)
    assert "cst" in names
    assert "F" in names
    assert "libcst" not in names
    assert "Foo" not in names


def test_get_external_import_names_syntax_error():
    assert _get_external_import_names("def (broken:") == set()


def test_cg_file_to_module_regular(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    f = pkg / "helpers.py"
    f.touch()
    mod, pkg_path = _cg_file_to_module_and_package(f, tmp_path)
    assert mod == "pkg.helpers"
    assert pkg_path == "pkg"


def test_cg_file_to_module_init(tmp_path):
    d = tmp_path / "pkg" / "utils"
    d.mkdir(parents=True)
    f = d / "__init__.py"
    f.touch()
    mod, pkg_path = _cg_file_to_module_and_package(f, tmp_path)
    assert mod == "pkg.utils"
    assert pkg_path == "pkg.utils"


def test_cg_file_to_module_top_level(tmp_path):
    f = tmp_path / "helpers.py"
    f.touch()
    mod, pkg_path = _cg_file_to_module_and_package(f, tmp_path)
    assert mod == "helpers"
    assert pkg_path == ""


def test_cg_file_to_module_nested(tmp_path):
    d = tmp_path / "a" / "b"
    d.mkdir(parents=True)
    f = d / "c.py"
    f.touch()
    mod, pkg_path = _cg_file_to_module_and_package(f, tmp_path)
    assert mod == "a.b.c"
    assert pkg_path == "a.b"


def test_cg_parse_imports_from_import():
    assert _cg_parse_imports("from pkg.sub import foo\n", "pkg") == {
        "foo": ("pkg.sub", "foo")
    }


def test_cg_parse_imports_import_simple():
    result = _cg_parse_imports("import os\n", "pkg")
    assert result["os"] == ("os", "os")


def test_cg_parse_imports_import_dotted():
    result = _cg_parse_imports("import pkg.sub\n", "")
    assert result["pkg"] == ("pkg.sub", "pkg.sub")


def test_cg_parse_imports_import_as():
    result = _cg_parse_imports("import os as o\n", "pkg")
    assert result["o"] == ("os", "os")


def test_cg_parse_imports_from_import_as():
    assert _cg_parse_imports("from pkg import foo as bar\n", "pkg") == {
        "bar": ("pkg", "foo")
    }


def test_cg_parse_imports_relative_level1():
    # `from . import helper` with package "pkg.sub" → mod = "pkg.sub"
    assert _cg_parse_imports("from . import helper\n", "pkg.sub") == {
        "helper": ("pkg.sub", "helper")
    }


def test_cg_parse_imports_relative_level2():
    # `from .. import foo` with package "pkg.sub" → base = "pkg"
    assert _cg_parse_imports("from .. import foo\n", "pkg.sub") == {
        "foo": ("pkg", "foo")
    }


def test_cg_parse_imports_relative_with_module():
    # `from .utils import helper` with package "pkg" → mod = "pkg.utils"
    assert _cg_parse_imports("from .utils import helper\n", "pkg") == {
        "helper": ("pkg.utils", "helper")
    }


def test_cg_parse_imports_relative_with_empty_base():
    # `from .sub import foo` with empty package → base="" → mod = "sub"
    assert _cg_parse_imports("from .sub import foo\n", "") == {"foo": ("sub", "foo")}


def test_cg_parse_imports_relative_no_module():
    # `from . import bar` with package "pkg.sub" → mod = "pkg.sub"
    assert _cg_parse_imports("from . import bar\n", "pkg.sub") == {
        "bar": ("pkg.sub", "bar")
    }


def test_cg_parse_imports_star_skipped():
    assert _cg_parse_imports("from pkg import *\n", "pkg") == {}


def test_cg_parse_imports_syntax_error():
    assert _cg_parse_imports("def f(:\n", "pkg") == {}


def test_cg_parse_imports_too_deep_relative():
    # level=3 with package="pkg" → go_up=2 > len(["pkg"])=1 → skipped
    assert _cg_parse_imports("from ... import foo\n", "pkg") == {}


def test_cg_parse_imports_level2_with_submodule():
    # `from ..utils import foo` with package "pkg.sub" → base="pkg" → "pkg.utils"
    assert _cg_parse_imports("from ..utils import foo\n", "pkg.sub") == {
        "foo": ("pkg.utils", "foo")
    }


def test_cg_index_get_imports_cached():
    index = _CgIndex(
        module_to_source={"pkg.mod": "from pkg.sub import foo\n"},
        module_to_package={"pkg.mod": "pkg"},
        module_to_defs={"pkg.mod": set()},
        file_to_module={},
    )
    r1 = index.get_imports("pkg.mod")
    r2 = index.get_imports("pkg.mod")  # second call — cached
    assert r1 == r2 == {"foo": ("pkg.sub", "foo")}
    assert "pkg.mod" in index._import_cache


def test_cg_index_get_imports_missing_module():
    index = _CgIndex(
        module_to_source={},
        module_to_package={},
        module_to_defs={},
        file_to_module={},
    )
    assert index.get_imports("nonexistent") == {}


def test_cg_build_index_from_repo(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text("def foo(): pass\n", encoding="utf-8")
    index = _cg_build_index(str(tmp_path), {}, [])
    assert "pkg.mod" in index.module_to_source
    assert "foo" in index.module_to_defs["pkg.mod"]
    assert index.module_to_package["pkg.mod"] == "pkg"


def test_cg_build_index_per_file_override(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    f = pkg / "mod.py"
    f.write_text("def old(): pass\n", encoding="utf-8")
    abs_path = str(f.resolve())
    index = _cg_build_index(str(tmp_path), {abs_path: "def new(): pass\n"}, [])
    assert "new" in index.module_to_defs["pkg.mod"]
    assert "old" not in index.module_to_defs["pkg.mod"]


def test_cg_build_index_no_repo_root():
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="",
        modified_source="",
        new_files={"placement.py": "def helper(): pass\n"},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths=set(),
    )
    index = _cg_build_index(None, {}, [ctx])
    assert "pkg.placement" in index.module_to_source
    assert index.file_to_module == {}


def test_cg_build_index_excluded_dirs(tmp_path):
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "mod.py").write_text("def foo(): pass\n", encoding="utf-8")
    index = _cg_build_index(str(tmp_path), {}, [])
    assert "mod" not in index.module_to_source


def test_cg_build_index_new_files_from_context():
    ctx = _FLContext(
        filepath="/proj/pkg/orig.py",
        old_module="pkg.orig",
        original_source="",
        modified_source="",
        new_files={"placement.py": "def helper(): pass\n"},
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths=set(),
    )
    index = _cg_build_index(None, {}, [ctx])
    assert "helper" in index.module_to_defs["pkg.placement"]
    assert index.module_to_package["pkg.placement"] == "pkg"


def test_cg_build_index_init_package(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .sub import foo\n", encoding="utf-8")
    (pkg / "sub.py").write_text("def foo(): pass\n", encoding="utf-8")
    index = _cg_build_index(str(tmp_path), {}, [])
    assert "pkg" in index.module_to_source
    assert index.module_to_package["pkg"] == "pkg"


def test_cg_build_index_already_in_index():
    ctx1 = _FLContext(
        filepath="/proj/orig.py",
        old_module="orig",
        original_source="",
        modified_source="",
        new_files={"placement.py": "def first(): pass\n"},
        new_module_paths={"placement.py": "pkg.shared"},
        entity_to_target={},
        forking_old_paths=set(),
    )
    ctx2 = _FLContext(
        filepath="/proj/orig2.py",
        old_module="orig2",
        original_source="",
        modified_source="",
        new_files={"placement.py": "def second(): pass\n"},
        new_module_paths={"placement.py": "pkg.shared"},  # same module path
        entity_to_target={},
        forking_old_paths=set(),
    )
    index = _cg_build_index(None, {}, [ctx1, ctx2])
    assert "first" in index.module_to_defs["pkg.shared"]
    assert "second" not in index.module_to_defs["pkg.shared"]


def test_cg_build_index_oserror(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    bad = pkg / "bad.py"
    bad.write_text("def foo(): pass\n", encoding="utf-8")
    bad.chmod(0o000)
    try:
        index = _cg_build_index(str(tmp_path), {}, [])
        assert "pkg.bad" not in index.module_to_source
    finally:
        bad.chmod(0o644)


def test_cg_build_index_missing_module_path():
    ctx = _FLContext(
        filepath="/proj/orig.py",
        old_module="orig",
        original_source="",
        modified_source="",
        new_files={"placement.py": "def helper(): pass\n"},
        new_module_paths={},  # rel_path missing → new_mod = None → skip
        entity_to_target={},
        forking_old_paths=set(),
    )
    index = _cg_build_index(None, {}, [ctx])
    assert "placement.py" not in index.module_to_source


def test_cg_build_index_empty_src():
    ctx = _FLContext(
        filepath="/proj/orig.py",
        old_module="orig",
        original_source="",
        modified_source="",
        new_files={"placement.py": ""},  # empty src → skipped
        new_module_paths={"placement.py": "pkg.placement"},
        entity_to_target={},
        forking_old_paths=set(),
    )
    index = _cg_build_index(None, {}, [ctx])
    assert "pkg.placement" not in index.module_to_source


def test_cg_build_index_init_package_new_file():
    # __init__.py as a new file: pkg = new_mod (not rsplit)
    ctx = _FLContext(
        filepath="/proj/orig.py",
        old_module="orig",
        original_source="",
        modified_source="",
        new_files={"__init__.py": "def init_fn(): pass\n"},
        new_module_paths={"__init__.py": "pkg.sub"},
        entity_to_target={},
        forking_old_paths=set(),
    )
    index = _cg_build_index(None, {}, [ctx])
    assert index.module_to_package["pkg.sub"] == "pkg.sub"


def test_cg_build_index_top_level_new_file():
    # new_mod without a dot → package = ""
    ctx = _FLContext(
        filepath="/proj/orig.py",
        old_module="orig",
        original_source="",
        modified_source="",
        new_files={"placement.py": "def helper(): pass\n"},
        new_module_paths={"placement.py": "placement"},  # no dot
        entity_to_target={},
        forking_old_paths=set(),
    )
    index = _cg_build_index(None, {}, [ctx])
    assert index.module_to_package.get("placement") == ""
