"""Tests for diff_parser module."""

from crispen.diff_parser import parse_diff

SIMPLE_DIFF = """\
--- a/foo.py
+++ b/foo.py
@@ -1,2 +1,3 @@
 def main():
-    pass
+    print("hello")
+    return 0
"""


def test_parse_simple_diff():
    result = parse_diff(SIMPLE_DIFF)
    assert "foo.py" in result
    # Lines 2 and 3 are added (print and return)
    ranges = result["foo.py"]
    assert len(ranges) >= 1
    covered = set()
    for start, end in ranges:
        covered.update(range(start, end + 1))
    assert 2 in covered
    assert 3 in covered


MULTI_FILE_DIFF = """\
--- a/a.py
+++ b/a.py
@@ -1,2 +1,3 @@
 x = 1
+y = 2
 z = 3
--- a/b.py
+++ b/b.py
@@ -5,2 +5,3 @@
 def foo():
+    pass
 return 1
"""


def test_parse_multi_file_diff():
    result = parse_diff(MULTI_FILE_DIFF)
    assert "a.py" in result
    assert "b.py" in result


def test_parse_empty_diff():
    result = parse_diff("")
    assert result == {}


def test_parse_no_additions():
    diff = """\
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,2 @@
 x = 1
-y = 2
 z = 3
"""
    result = parse_diff(diff)
    assert result == {}


def test_non_consecutive_lines_two_ranges():
    # Source has 1 context line; target adds lines 1-2, keeps line 3 as context,
    # then adds lines 4-5. The range merger should split into [(1,2), (4,5)].
    diff = """\
--- a/foo.py
+++ b/foo.py
@@ -3,1 +1,5 @@
+line1
+line2
 context_line
+line4
+line5
"""
    result = parse_diff(diff)
    assert "foo.py" in result
    ranges = result["foo.py"]
    assert len(ranges) == 2
    assert ranges[0] == (1, 2)
    assert ranges[1] == (4, 5)


def test_non_python_files_excluded():
    diff = """\
--- a/config.json
+++ b/config.json
@@ -1,2 +1,3 @@
 {
+"key": "value",
 }
"""
    result = parse_diff(diff)
    assert result == {}


def test_consecutive_lines_merged():
    diff = """\
--- a/foo.py
+++ b/foo.py
@@ -0,0 +1,4 @@
+line1
+line2
+line3
+line4
"""
    result = parse_diff(diff)
    assert "foo.py" in result
    ranges = result["foo.py"]
    # Should be merged into a single range
    assert len(ranges) == 1
    assert ranges[0] == (1, 4)
