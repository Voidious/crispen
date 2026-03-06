from crispen.refactors.duplicate_extractor import _apply_edits


def test_apply_edits_no_edits():
    source = "a = 1\nb = 2\n"
    assert _apply_edits(source, []) == source


def test_apply_edits_replacement():
    source = "a = 1\nb = 2\nc = 3\n"
    # Replace line index 1 (b = 2) with new content
    result = _apply_edits(source, [(1, 2, "x = 99\n")])
    assert result == "a = 1\nx = 99\nc = 3\n"


def test_apply_edits_insertion():
    source = "a = 1\nb = 2\n"
    # Insert before line index 1 (b = 2)
    result = _apply_edits(source, [(1, 1, "INSERTED\n")])
    assert result == "a = 1\nINSERTED\nb = 2\n"


def test_apply_edits_overlapping_skipped():
    source = "a = 1\nb = 2\nc = 3\n"
    edits = [
        (0, 2, "FIRST\n"),
        (1, 3, "SECOND\n"),  # overlaps with first
    ]
    result = _apply_edits(source, edits)
    # Higher-start edit (SECOND) wins; FIRST overlaps and is skipped
    assert "SECOND" in result
    assert "FIRST" not in result


def test_apply_edits_no_trailing_newline_source():
    source = "a = 1"  # no trailing newline
    result = _apply_edits(source, [(0, 1, "b = 2\n")])
    assert result == "b = 2\n"


def test_apply_edits_no_trailing_newline_text():
    source = "a = 1\nb = 2\n"
    # Replacement text without trailing newline
    result = _apply_edits(source, [(0, 1, "x = 99")])
    assert result == "x = 99\nb = 2\n"
