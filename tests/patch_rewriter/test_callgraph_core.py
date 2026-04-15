from __future__ import annotations
from crispen.patch_rewriter import (
    _CgIndex,
    _FLContext,
    _cg_build_index,
    _cg_collect_called_names,
    _cg_collect_defined_names,
    _cg_collect_func_body_calls,
    _cg_file_to_module_and_package,
    _cg_parse_imports,
    _cg_resolve_call_to_import,
    _resolve_import_to_file,
)


def test_resolve_relative_level1_py(tmp_path):
    # from .sub import NAME — sub.py exists
    (tmp_path / "sub.py").write_text("X = 1\n", encoding="utf-8")
    scan = str(tmp_path / "test_foo.py")
    result = _resolve_import_to_file("sub", 1, scan, None)
    assert result == str(tmp_path / "sub.py")


def test_resolve_relative_level1_init(tmp_path):
    # from .pkg import NAME — pkg/__init__.py exists
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    scan = str(tmp_path / "test_foo.py")
    result = _resolve_import_to_file("pkg", 1, scan, None)
    assert result == str(pkg / "__init__.py")


def test_resolve_relative_level1_no_module(tmp_path):
    # from . import NAME — finds __init__.py in same dir
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    scan = str(tmp_path / "test_foo.py")
    result = _resolve_import_to_file(None, 1, scan, None)
    assert result == str(tmp_path / "__init__.py")


def test_resolve_relative_level2(tmp_path):
    # from ..sub import NAME — goes up one level
    parent = tmp_path / "parent"
    parent.mkdir()
    child = parent / "child"
    child.mkdir()
    (parent / "sub.py").write_text("X = 1\n", encoding="utf-8")
    scan = str(child / "test_foo.py")
    result = _resolve_import_to_file("sub", 2, scan, None)
    assert result == str(parent / "sub.py")


def test_resolve_relative_not_found(tmp_path):
    scan = str(tmp_path / "test_foo.py")
    assert _resolve_import_to_file("missing", 1, scan, None) is None


def test_resolve_relative_no_module_no_init(tmp_path):
    scan = str(tmp_path / "test_foo.py")
    assert _resolve_import_to_file(None, 1, scan, None) is None


def test_resolve_absolute_found(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "helpers.py").write_text("X = 1\n", encoding="utf-8")
    scan = str(tmp_path / "tests" / "test_foo.py")
    result = _resolve_import_to_file("mypkg.helpers", 0, scan, str(tmp_path))
    assert result == str(pkg / "helpers.py")


def test_resolve_absolute_no_repo_root(tmp_path):
    scan = str(tmp_path / "test_foo.py")
    assert _resolve_import_to_file("mypkg.helpers", 0, scan, None) is None


def test_resolve_absolute_no_module(tmp_path):
    scan = str(tmp_path / "test_foo.py")
    assert _resolve_import_to_file(None, 0, scan, str(tmp_path)) is None


def test_resolve_absolute_not_found(tmp_path):
    scan = str(tmp_path / "test_foo.py")
    assert _resolve_import_to_file("no.such.module", 0, scan, str(tmp_path)) is None


def test_cg_collect_called_names_name_and_attr():
    src = "foo()\nobj.bar()\n"
    result = _cg_collect_called_names(src)
    assert "foo" in result
    assert "bar" in result


def test_cg_collect_called_names_complex_func():
    # f()() — outer call's func is a Call node (neither Name nor Attribute).
    src = "f()()\n"
    result = _cg_collect_called_names(src)
    # Only the inner call's name is collected (f), the outer call is skipped.
    assert "f" in result


def test_cg_collect_called_names_parse_error():
    assert _cg_collect_called_names("def f(:\n") == set()


def test_cg_collect_called_names_no_calls():
    assert _cg_collect_called_names("x = 1\n") == set()


def test_cg_collect_func_body_calls_found():
    src = "def helper(): foo()\ndef other(): bar()\n"
    result = _cg_collect_func_body_calls(src, "helper")
    assert "foo" in result
    assert "bar" not in result


def test_cg_collect_func_body_calls_not_found():
    src = "def helper(): foo()\n"
    assert _cg_collect_func_body_calls(src, "missing") == set()


def test_cg_collect_func_body_calls_parse_error():
    assert _cg_collect_func_body_calls("def f(:\n", "f") == set()


def test_cg_collect_func_body_calls_attribute_call():
    src = "def helper(): obj.method()\n"
    result = _cg_collect_func_body_calls(src, "helper")
    assert "method" in result


def test_cg_collect_func_body_calls_complex_func():
    # f()() inside a function body — outer call's func is a Call, not Name/Attribute.
    src = "def helper(): f()()\n"
    result = _cg_collect_func_body_calls(src, "helper")
    assert "f" in result  # inner call collected; outer (complex func) silently skipped


def test_cg_collect_func_body_calls_skips_non_function_nodes():
    # Module-level assignment before the function — should be skipped.
    src = "X = 1\ndef helper(): foo()\n"
    result = _cg_collect_func_body_calls(src, "helper")
    assert "foo" in result


def test_cg_collect_called_names_alias_access_emits_pair():
    # ``m.func()`` should emit both ``"func"`` and ``"m.func"``.
    src = "import mymod as m\nm.func()\n"
    result = _cg_collect_called_names(src)
    assert "func" in result
    assert "m.func" in result


def test_cg_collect_called_names_nested_attr_no_alias_pair():
    # ``a.b.c()`` — the receiver of ``.c`` is itself an Attribute, not a Name;
    # only the bare attr name is emitted (no alias pair for chained access).
    src = "a.b.c()\n"
    result = _cg_collect_called_names(src)
    assert "c" in result
    assert "b.c" not in result  # receiver is Attribute, not Name


def test_cg_collect_func_body_calls_alias_access_emits_pair():
    src = "import mymod as m\ndef helper(): m.process()\n"
    result = _cg_collect_func_body_calls(src, "helper")
    assert "process" in result
    assert "m.process" in result


def test_cg_collect_func_body_calls_nested_attr_no_alias_pair():
    # Chained access ``a.b.c()`` — receiver of attr c is not a Name.
    src = "def helper(): a.b.c()\n"
    result = _cg_collect_func_body_calls(src, "helper")
    assert "c" in result
    assert "b.c" not in result


def test_cg_resolve_call_plain_name():
    imports = {"foo": ("pkg.sub", "foo"), "bar": ("pkg.other", "bar")}
    assert _cg_resolve_call_to_import("foo", imports) == ("pkg.sub", "foo")


def test_cg_resolve_call_alias_attr():
    # ``m.process()`` — alias ``m`` maps to module ``mymod``; resolves to
    # ``(mymod, "process")``.
    imports = {"m": ("mymod", "mymod")}
    assert _cg_resolve_call_to_import("m.process", imports) == ("mymod", "process")


def test_cg_resolve_call_alias_attr_unknown_alias():
    # Alias not in imports → None.
    assert _cg_resolve_call_to_import("unknown.func", {"m": ("mymod", "mymod")}) is None


def test_cg_resolve_call_plain_not_found():
    assert _cg_resolve_call_to_import("missing", {"foo": ("pkg", "foo")}) is None


def test_cg_collect_defined_names_functions_and_classes():
    src = "def foo(): pass\nclass Bar: pass\nasync def baz(): pass\n"
    result = _cg_collect_defined_names(src)
    assert result == {"foo", "Bar", "baz"}


def test_cg_collect_defined_names_parse_error():
    assert _cg_collect_defined_names("def f(:\n") == set()


def test_cg_collect_defined_names_empty():
    assert _cg_collect_defined_names("x = 1\n") == set()


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
