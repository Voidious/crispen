"""Tests for patch_updater — 100% branch coverage."""

from __future__ import annotations

from crispen.patch_updater import apply_patch_strings


# ---------------------------------------------------------------------------
# apply_patch_strings
# ---------------------------------------------------------------------------


def test_empty_patch_map_returns_source_unchanged():
    src = 'patch("old.module.MyClass")\n'
    assert apply_patch_strings(src, {}) == src


def test_non_matching_string_unchanged():
    src = 'patch("other.module.OtherClass")\n'
    assert (
        apply_patch_strings(src, {"old.module.MyClass": "old.module.helpers.MyClass"})
        == src
    )


def test_bom_source_no_match_returns_unchanged():
    # Regression: libcst's parser drops a leading BOM but its renderer does
    # not restore it, so a naive parse-then-render round trip through a
    # non-matching file used to report a change (and would have silently
    # stripped the BOM from disk) even though no @patch string was touched.
    src = '﻿patch("other.module.OtherClass")\n'
    result = apply_patch_strings(
        src, {"old.module.MyClass": "old.module.helpers.MyClass"}
    )
    assert result == src


def test_bom_source_with_match_preserves_bom():
    src = '﻿patch("old.module.MyClass")\n'
    result = apply_patch_strings(
        src, {"old.module.MyClass": "old.module.helpers.MyClass"}
    )
    assert result.startswith("﻿")
    assert '"old.module.helpers.MyClass"' in result


def test_exact_match_double_quoted():
    src = 'patch("old.module.MyClass")\n'
    result = apply_patch_strings(
        src, {"old.module.MyClass": "old.module.helpers.MyClass"}
    )
    assert '"old.module.helpers.MyClass"' in result


def test_prefix_match_preserves_suffix():
    src = 'patch("old.module.MyClass.method")\n'
    result = apply_patch_strings(
        src, {"old.module.MyClass": "old.module.helpers.MyClass"}
    )
    assert '"old.module.helpers.MyClass.method"' in result


def test_no_false_prefix_match():
    # "old.FooExtra" should NOT match key "old.Foo"
    src = 'patch("old.FooExtra")\n'
    result = apply_patch_strings(src, {"old.Foo": "old.helpers.Foo"})
    assert result == src


def test_triple_quoted_string_skipped():
    src = 'x = """old.module.MyClass"""\n'
    result = apply_patch_strings(
        src, {"old.module.MyClass": "old.module.helpers.MyClass"}
    )
    assert result == src


def test_prefixed_string_skipped():
    # b"..." — raw[0] is 'b', not a quote
    src = 'patch(b"old.module.MyClass")\n'
    result = apply_patch_strings(
        src, {"old.module.MyClass": "old.module.helpers.MyClass"}
    )
    assert result == src


def test_parse_error_returns_source_unchanged():
    src = "def f(:\n    pass\n"
    result = apply_patch_strings(
        src, {"old.module.MyClass": "old.module.helpers.MyClass"}
    )
    assert result == src


def test_multiple_occurrences_all_updated():
    src = 'patch("old.module.MyClass")\n' 'patch("old.module.MyClass")\n'
    result = apply_patch_strings(
        src, {"old.module.MyClass": "old.module.helpers.MyClass"}
    )
    assert result.count('"old.module.helpers.MyClass"') == 2


def test_single_quoted_string_updated():
    src = "patch('old.module.MyClass')\n"
    result = apply_patch_strings(
        src, {"old.module.MyClass": "old.module.helpers.MyClass"}
    )
    assert "'old.module.helpers.MyClass'" in result


def test_longest_key_matches_first():
    # "a.b.C.method" should match "a.b.C" (longer) before "a.b"
    src = 'patch("a.b.C.method")\n'
    patch_map = {
        "a.b.C": "a.b.sub.C",
        "a.b": "a.x",
    }
    result = apply_patch_strings(src, patch_map)
    assert '"a.b.sub.C.method"' in result
    assert '"a.x' not in result


def test_single_quoted_triple_skipped():
    src = "x = '''old.module.MyClass'''\n"
    result = apply_patch_strings(
        src, {"old.module.MyClass": "old.module.helpers.MyClass"}
    )
    assert result == src
