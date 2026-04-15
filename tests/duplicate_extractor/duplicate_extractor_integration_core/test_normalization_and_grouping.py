from crispen.refactors.duplicate_extractor import (
    _SeqInfo,
    _filter_maximal_groups,
    _find_duplicate_groups,
    _has_internal_overlap,
    _normalize_source,
    _overlaps_diff,
)


def test_normalize_source_normalizes_vars():
    src = "result = compute(data)\noutput = transform(result)\n"
    norm = _normalize_source(src)
    # All names (both assigned and free) are replaced with positional placeholders
    assert "result" not in norm
    assert "output" not in norm
    assert "compute" not in norm
    assert "data" not in norm


def test_normalize_source_same_fingerprint():
    src_a = "x = compute(data)\ny = transform(x)\n"
    src_b = "val = compute(data)\nres = transform(val)\n"
    assert _normalize_source(src_a) == _normalize_source(src_b)


def test_normalize_source_different_ops():
    # Structurally different code (different number of statements) should differ
    src_a = "x = a + b\n"
    src_b = "x = a + b\ny = x * 2\n"
    assert _normalize_source(src_a) != _normalize_source(src_b)


def test_normalize_source_invalid_syntax():
    src = "def f(: pass"
    # Falls back to original source
    assert _normalize_source(src) == src


def test_normalize_source_load_context_replaced():
    # Var assigned then used: both should be normalized the same
    src_a = "x = 1\ny = x + 1\n"
    src_b = "a = 1\nb = a + 1\n"
    assert _normalize_source(src_a) == _normalize_source(src_b)


def test_normalize_source_load_not_in_map():
    # Free variables (Load context, never stored) are also normalized,
    # so two blocks with different free variable names get the same fingerprint.
    src_a = "y = a + 1\n"
    src_b = "z = b + 1\n"
    assert _normalize_source(src_a) == _normalize_source(src_b)


def test_normalize_source_repeated_store():
    # Same name assigned twice: _placeholder called with cached key (False branch)
    src = "x = 1\nx = 2\n"
    norm = _normalize_source(src)
    # Both assignments normalize to the same placeholder
    assert norm.count("_v0") == 2


def test_normalize_source_del_context():
    # Del context falls through to return node unchanged
    src = "del x\n"
    norm = _normalize_source(src)
    assert "x" in norm


def test_normalize_source_indented_blocks_match():
    # Source collected from inside a function is indented; dedent must happen
    # before ast.parse so that structurally identical blocks still match.
    src_a = "    p = a * 2\n    if p > 100:\n        p += 1\n"
    src_b = "    q = b * 2\n    if q > 100:\n        q += 1\n"
    assert _normalize_source(src_a) == _normalize_source(src_b)


def _make_seq(start: int, end: int) -> _SeqInfo:
    return _SeqInfo(
        stmts=[],
        start_line=start,
        end_line=end,
        scope="<module>",
        source="",
        fingerprint="",
    )


def test_overlaps_diff_yes():
    seq = _make_seq(5, 10)
    assert _overlaps_diff(seq, [(8, 12)]) is True


def test_overlaps_diff_no():
    seq = _make_seq(5, 10)
    assert _overlaps_diff(seq, [(11, 20)]) is False


def test_overlaps_diff_exact_boundary():
    seq = _make_seq(5, 10)
    assert _overlaps_diff(seq, [(10, 15)]) is True


def test_find_duplicate_groups_empty():
    assert _find_duplicate_groups([], [(1, 5)]) == []


def test_find_duplicate_groups_singleton():
    seq = _make_seq(1, 3)
    seq.fingerprint = "fp1"
    seqs = [seq]
    # Only one seq with this fingerprint — not a duplicate
    assert _find_duplicate_groups(seqs, [(1, 3)]) == []


def test_find_duplicate_groups_no_diff_overlap():
    s1 = _SeqInfo([], 1, 3, "<module>", "", "fp1")
    s2 = _SeqInfo([], 10, 12, "<module>", "", "fp1")
    # Neither overlaps diff range (20, 30)
    assert _find_duplicate_groups([s1, s2], [(20, 30)]) == []


def test_find_duplicate_groups_valid():
    s1 = _SeqInfo([], 1, 3, "<module>", "", "fp1")
    s2 = _SeqInfo([], 10, 12, "<module>", "", "fp1")
    groups = _find_duplicate_groups([s1, s2], [(1, 3)])
    assert len(groups) == 1
    assert set(id(s) for s in groups[0]) == {id(s1), id(s2)}


def test_has_internal_overlap_no_overlap():
    s1 = _SeqInfo([], 1, 3, "<module>", "", "fp1")
    s2 = _SeqInfo([], 10, 12, "<module>", "", "fp1")
    assert not _has_internal_overlap([s1, s2])


def test_has_internal_overlap_adjacent_no_overlap():
    # end_line of s1 == start_line - 1 of s2: not overlapping
    s1 = _SeqInfo([], 1, 5, "<module>", "", "fp1")
    s2 = _SeqInfo([], 6, 10, "<module>", "", "fp1")
    assert not _has_internal_overlap([s1, s2])


def test_has_internal_overlap_touching():
    # end_line of s1 == start_line of s2: overlap (shared boundary line)
    s1 = _SeqInfo([], 1, 5, "<module>", "", "fp1")
    s2 = _SeqInfo([], 5, 9, "<module>", "", "fp1")
    assert _has_internal_overlap([s1, s2])


def test_has_internal_overlap_proper_overlap():
    s1 = _SeqInfo([], 27, 30, "<module>", "", "fp1")
    s2 = _SeqInfo([], 29, 32, "<module>", "", "fp1")
    assert _has_internal_overlap([s1, s2])


def test_has_internal_overlap_unsorted_order():
    # Sequences given in reverse order — function must sort before checking.
    s1 = _SeqInfo([], 29, 32, "<module>", "", "fp1")
    s2 = _SeqInfo([], 27, 30, "<module>", "", "fp1")
    assert _has_internal_overlap([s1, s2])


def test_find_duplicate_groups_skips_internally_overlapping():
    # Simulate the op_range pattern: two pairs [A,B] and [B,C] that share a
    # statement.  The group has internal overlap and must be filtered out.
    s1 = _SeqInfo([], 27, 30, "<module>", "", "fp1")
    s2 = _SeqInfo([], 29, 32, "<module>", "", "fp1")
    # Diff covers both sequences.
    groups = _find_duplicate_groups([s1, s2], [(27, 32)])
    assert groups == []


def test_find_duplicate_groups_caps_at_max_groups():
    sequences = []
    for i in range(6):
        fp = f"fp{i}"
        # Place each group in a disjoint band of 20 lines so _filter_maximal_groups
        # keeps all 6 (none overlap), and the max_groups=3 cap is what limits output.
        sequences.append(_SeqInfo([], i * 20 + 1, i * 20 + 3, "<module>", "", fp))
        sequences.append(_SeqInfo([], i * 20 + 10, i * 20 + 12, "<module>", "", fp))
    # Diff range covers all sequences so the diff-overlap filter passes for all.
    groups = _find_duplicate_groups(sequences, [(1, 130)], max_groups=3)
    assert len(groups) == 3


def test_filter_maximal_groups_empty():
    assert _filter_maximal_groups([]) == []


def test_filter_maximal_groups_single_group():
    s1 = _SeqInfo([], 1, 10, "<module>", "", "fp1")
    s2 = _SeqInfo([], 20, 29, "<module>", "", "fp1")
    group = [s1, s2]
    result = _filter_maximal_groups([group])
    assert result == [group]


def test_filter_maximal_groups_removes_subsumed():
    # Large group spans lines 1-10; small group spans 1-5 (subset).
    # Only the large group should be kept.
    s_large_a = _SeqInfo([], 1, 10, "<module>", "", "fp_large")
    s_large_b = _SeqInfo([], 20, 29, "<module>", "", "fp_large")
    large_group = [s_large_a, s_large_b]

    s_small_a = _SeqInfo([], 1, 5, "<module>", "", "fp_small")
    s_small_b = _SeqInfo([], 20, 24, "<module>", "", "fp_small")
    small_group = [s_small_a, s_small_b]

    result = _filter_maximal_groups([small_group, large_group])
    assert len(result) == 1
    assert result[0] is large_group


def test_filter_maximal_groups_keeps_non_overlapping():
    # Two groups with completely disjoint line ranges — both should be kept.
    s1a = _SeqInfo([], 1, 5, "<module>", "", "fp1")
    s1b = _SeqInfo([], 30, 34, "<module>", "", "fp1")
    group1 = [s1a, s1b]

    s2a = _SeqInfo([], 10, 14, "<module>", "", "fp2")
    s2b = _SeqInfo([], 40, 44, "<module>", "", "fp2")
    group2 = [s2a, s2b]

    result = _filter_maximal_groups([group1, group2])
    assert len(result) == 2
