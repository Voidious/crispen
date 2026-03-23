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
    s.patch_update_edits = 1
    s.algorithmic_rejected = 0
    s.llm_rejected = 1
    s.llm_veto_calls = 4
    s.llm_edit_calls = 7
    s.llm_verify_calls = 3
    s.file_limiter_llm_calls = 5
    s.patch_rewrite_llm_calls = 2
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
    b.llm_elapsed = 1.5
    b.llm_input_tokens = 100
    b.llm_output_tokens = 50
    b.llm_elapsed_by_category["veto"] = 1.5
    b.llm_calls_by_category["veto"] = 1
    b.llm_elapsed_by_refactor["duplicate_extractor"] = 1.5
    b.llm_elapsed_by_file["foo.py"] = 1.5
    a.merge(b)
    assert a.if_not_else == 11
    assert a.tuple_to_dataclass == 2
    assert a.duplicate_extracted == 5
    assert a.llm_veto_calls == 3
    assert a.llm_edit_calls == 2
    assert a.llm_elapsed == 1.5
    assert a.llm_input_tokens == 100
    assert a.llm_elapsed_by_category["veto"] == 1.5
    assert a.llm_calls_by_category["veto"] == 1
    assert a.llm_elapsed_by_refactor["duplicate_extractor"] == 1.5
    assert a.llm_elapsed_by_file["foo.py"] == 1.5


def test_merge_adds_token_breakdown_dicts():
    """Merging RunStats accumulates all per-category/refactor/file token dicts."""
    a = RunStats()
    b = RunStats()
    b.llm_input_tokens_by_category["veto"] = 100
    b.llm_output_tokens_by_category["veto"] = 50
    b.llm_input_tokens_by_refactor["duplicate_extractor"] = 200
    b.llm_output_tokens_by_refactor["duplicate_extractor"] = 80
    b.llm_input_tokens_by_file["foo.py"] = 300
    b.llm_output_tokens_by_file["foo.py"] = 90
    a.merge(b)
    assert a.llm_input_tokens_by_category["veto"] == 100
    assert a.llm_output_tokens_by_category["veto"] == 50
    assert a.llm_input_tokens_by_refactor["duplicate_extractor"] == 200
    assert a.llm_output_tokens_by_refactor["duplicate_extractor"] == 80
    assert a.llm_input_tokens_by_file["foo.py"] == 300
    assert a.llm_output_tokens_by_file["foo.py"] == 90


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


def test_merge_adds_patch_update_fields():
    a = RunStats(patch_update_edits=2, patch_rewrite_llm_calls=3)
    b = RunStats(patch_update_edits=4, patch_rewrite_llm_calls=1)
    a.merge(b)
    assert a.patch_update_edits == 6
    assert a.patch_rewrite_llm_calls == 4


def test_total_edits_includes_patch_update():
    s = RunStats(
        if_not_else=1,
        tuple_to_dataclass=1,
        duplicate_extracted=1,
        duplicate_matched=1,
        function_split=1,
        file_limiter_edits=1,
        patch_update_edits=3,
    )
    assert s.total_edits == 9


def test_total_llm_calls_includes_patch_rewrite():
    s = RunStats(
        llm_veto_calls=1,
        llm_edit_calls=1,
        llm_verify_calls=1,
        file_limiter_llm_calls=1,
        patch_rewrite_llm_calls=5,
    )
    assert s.total_llm_calls == 9


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


def test_format_summary_with_files():
    s = _filled()
    lines = s.format_summary()
    text = "\n".join(lines)
    assert "--- crispen summary ---" in text
    assert "if not/else:" in text
    assert "tuple to dataclass:" in text
    assert "duplicate extracted:" in text
    assert "match existing:" in text
    assert "function split:" in text
    assert "file limiter:        2" in text
    assert "patch update:        1" in text
    assert "total:               14" in text
    assert "algorithmic:" in text
    assert "LLM:" in text
    assert "veto:" in text
    assert "edit:" in text
    assert "verify:" in text
    assert "file limiter:        5" in text
    assert "patch rewrite:       2" in text
    assert "total:               21" in text
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
    lines = s.format_summary()
    text = "\n".join(lines)
    assert "files edited: none" in text


# ---------------------------------------------------------------------------
# record_llm_call
# ---------------------------------------------------------------------------


def test_record_llm_call_basic():
    s = RunStats()
    s.record_llm_call(1.5, 100, 50, "veto", "duplicate_extractor", "foo.py")
    assert s.llm_elapsed == 1.5
    assert s.llm_input_tokens == 100
    assert s.llm_output_tokens == 50
    assert s.llm_elapsed_by_category["veto"] == 1.5
    assert s.llm_calls_by_category["veto"] == 1
    assert s.llm_elapsed_by_refactor["duplicate_extractor"] == 1.5
    assert s.llm_elapsed_by_file["foo.py"] == 1.5


def test_record_llm_call_accumulates():
    s = RunStats()
    s.record_llm_call(1.0, 100, 50, "veto", "duplicate_extractor", "foo.py")
    s.record_llm_call(0.5, 200, 30, "edit", "duplicate_extractor", "foo.py")
    assert s.llm_elapsed == 1.5
    assert s.llm_input_tokens == 300
    assert s.llm_calls_by_category["veto"] == 1
    assert s.llm_calls_by_category["edit"] == 1
    assert s.llm_elapsed_by_refactor["duplicate_extractor"] == 1.5
    assert s.llm_elapsed_by_file["foo.py"] == 1.5


def test_record_llm_call_empty_file_skips_file_dict():
    s = RunStats()
    s.record_llm_call(1.0, 100, 50, "veto", "file_limiter", "")
    assert s.llm_elapsed_by_file == {}


# ---------------------------------------------------------------------------
# format_summary timing section
# ---------------------------------------------------------------------------


def test_format_summary_basic_timing():
    s = RunStats()
    s.total_elapsed = 3.0
    s.llm_elapsed = 2.0
    s.llm_input_tokens = 1000
    s.llm_output_tokens = 500
    lines = s.format_summary(timing="basic")
    text = "\n".join(lines)
    assert "timing:" in text
    assert "total:" in text
    assert "3.00s" in text
    assert "2.00s" in text
    assert "67%" in text
    assert "1,000 in" in text
    assert "500 out" in text


def test_format_summary_off_timing():
    s = RunStats()
    s.total_elapsed = 3.0
    lines = s.format_summary(timing="off")
    text = "\n".join(lines)
    assert "timing:" not in text


def test_format_summary_detailed_timing():
    s = RunStats()
    s.total_elapsed = 3.0
    s.llm_elapsed = 2.0
    s.llm_input_tokens = 1000
    s.llm_output_tokens = 500
    s.record_llm_call(1.0, 600, 300, "veto", "duplicate_extractor", "foo.py")
    s.record_llm_call(1.0, 400, 200, "edit", "function_splitter", "bar.py")
    lines = s.format_summary(timing="detailed")
    text = "\n".join(lines)
    assert "LLM by call type:" in text
    assert "veto" in text
    assert "LLM by refactor:" in text
    assert "duplicate_extractor" in text
    assert "LLM by file:" in text
    assert "foo.py" in text


def test_format_summary_timing_zero_total_elapsed():
    """When total_elapsed=0 we don't divide, just show 0.00s."""
    s = RunStats()
    s.llm_elapsed = 0.0
    lines = s.format_summary(timing="basic")
    text = "\n".join(lines)
    assert "timing:" in text
    assert "total:               0.00s" in text
    assert "LLM:                 0.00s" in text


def test_format_summary_detailed_no_breakdowns():
    """detailed timing with no calls shows timing header but no breakdowns."""
    s = RunStats()
    s.total_elapsed = 1.0
    lines = s.format_summary(timing="detailed")
    text = "\n".join(lines)
    assert "timing:" in text
    assert "LLM by call type:" not in text
    assert "LLM by refactor:" not in text
    assert "LLM by file:" not in text
