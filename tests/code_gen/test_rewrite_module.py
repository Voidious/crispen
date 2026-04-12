from __future__ import annotations
from crispen.import_sort import _sort_imports_pep8
from crispen.file_limiter.code_gen import (
    _merge_from_imports,
    _rewrite_module_level_stores,
    _rewrite_module_var_names,
)


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
