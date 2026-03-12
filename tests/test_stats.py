"""Tests for crispen.stats.RunStats."""

from crispen.stats import RunStats


def _filled() -> RunStats:
    s = RunStats()
    s.if_not_else = 2
    s.tuple_to_dataclass = 1
    s.duplicate_extracted = 3
    s.duplicate_matched = 1
    s.function_split = 4
    s.file_limiter_edits = 2
    s.algorithmic_rejected = 0
    s.llm_rejected = 1
    s.llm_veto_calls = 4
    s.llm_edit_calls = 7
    s.llm_verify_calls = 3
    s.file_limiter_llm_calls = 5
    s.files_edited = ["foo.py", "bar.py"]
    s.lines_added = 30
    s.lines_deleted = 15
    s.file_limiter_functions_verified = 6
    s.file_limiter_classes_verified = 2
    s.file_limiter_lines_verified = 120
    return s


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def test_merge_adds_all_counters():
    a = RunStats(if_not_else=1, tuple_to_dataclass=2, llm_veto_calls=3)
    b = RunStats(if_not_else=10, duplicate_extracted=5, llm_edit_calls=2)
    a.merge(b)
    assert a.if_not_else == 11
    assert a.tuple_to_dataclass == 2
    assert a.duplicate_extracted == 5
    assert a.llm_veto_calls == 3
    assert a.llm_edit_calls == 2


def test_merge_does_not_merge_files_edited():
    a = RunStats()
    a.files_edited = ["a.py"]
    b = RunStats()
    b.files_edited = ["b.py"]
    a.merge(b)
    assert a.files_edited == ["a.py"]


def test_merge_adds_file_limiter_fields():
    a = RunStats(
        file_limiter_edits=1,
        file_limiter_llm_calls=2,
        file_limiter_functions_verified=3,
        file_limiter_classes_verified=4,
        file_limiter_lines_verified=50,
        lines_added=10,
        lines_deleted=5,
    )
    b = RunStats(
        file_limiter_edits=2,
        file_limiter_llm_calls=3,
        file_limiter_functions_verified=1,
        file_limiter_classes_verified=0,
        file_limiter_lines_verified=20,
        lines_added=5,
        lines_deleted=2,
    )
    a.merge(b)
    assert a.file_limiter_edits == 3
    assert a.file_limiter_llm_calls == 5
    assert a.file_limiter_functions_verified == 4
    assert a.file_limiter_classes_verified == 4
    assert a.file_limiter_lines_verified == 70
    assert a.lines_added == 15
    assert a.lines_deleted == 7


# ---------------------------------------------------------------------------
# property totals
# ---------------------------------------------------------------------------


def test_total_edits():
    s = RunStats(
        if_not_else=2,
        tuple_to_dataclass=1,
        duplicate_extracted=3,
        duplicate_matched=1,
        function_split=4,
        file_limiter_edits=2,
    )
    assert s.total_edits == 13


def test_total_rejected():
    s = RunStats(algorithmic_rejected=2, llm_rejected=3)
    assert s.total_rejected == 5


def test_total_llm_calls():
    s = RunStats(
        llm_veto_calls=4,
        llm_edit_calls=7,
        llm_verify_calls=3,
        file_limiter_llm_calls=5,
    )
    assert s.total_llm_calls == 19


# ---------------------------------------------------------------------------
# count_lines_changed
# ---------------------------------------------------------------------------


def test_count_lines_changed_added_lines():
    s = RunStats()
    s.count_lines_changed("a = 1\n", "a = 1\nb = 2\n")
    assert s.lines_added == 1
    assert s.lines_deleted == 0


def test_count_lines_changed_removed_lines():
    s = RunStats()
    s.count_lines_changed("a = 1\nb = 2\n", "a = 1\n")
    assert s.lines_added == 0
    assert s.lines_deleted == 1


def test_count_lines_changed_no_difference():
    s = RunStats()
    s.count_lines_changed("x = 1\n", "x = 1\n")
    assert s.lines_added == 0
    assert s.lines_deleted == 0


# ---------------------------------------------------------------------------
# format_summary — with files
# ---------------------------------------------------------------------------


def _format_summary_text(stats: RunStats) -> str:
    lines = stats.format_summary()
    return "\n".join(lines)


def test_format_summary_with_files():
    s = _filled()
    text = _format_summary_text(s)
    assert "--- crispen summary ---" in text
    assert "if not/else:" in text
    assert "tuple to dataclass:" in text
    assert "duplicate extracted:" in text
    assert "match existing:" in text
    assert "function split:" in text
    assert "file limiter:        2" in text
    assert "total:               13" in text
    assert "algorithmic:" in text
    assert "LLM:" in text
    assert "veto:" in text
    assert "edit:" in text
    assert "verify:" in text
    assert "file limiter:        5" in text
    assert "total:               19" in text
    assert "files edited (2): foo.py, bar.py" in text
    assert "lines added:           30" in text
    assert "lines deleted:         15" in text
    assert "file limiter verified:" in text
    assert "functions:           6" in text
    assert "classes:             2" in text
    assert "lines:               120" in text


# ---------------------------------------------------------------------------
# format_summary — without files (exercises the else branch)
# ---------------------------------------------------------------------------


def test_format_summary_no_files():
    s = RunStats()
    text = _format_summary_text(s)
    assert "files edited: none" in text
