"""Cumulative statistics for a single Crispen run."""

import difflib
from dataclasses import dataclass, field
from typing import List


@dataclass
class RunStats:
    """Holds cumulative counts for a single crispen run."""

    # Edit counts by refactor type
    if_not_else: int = 0
    tuple_to_dataclass: int = 0
    duplicate_extracted: int = 0
    duplicate_matched: int = 0
    function_split: int = 0

    # Rejection counts
    algorithmic_rejected: int = 0
    llm_rejected: int = 0

    # LLM call counts
    llm_veto_calls: int = 0
    llm_edit_calls: int = 0
    llm_verify_calls: int = 0

    # File and line tracking
    files_edited: List[str] = field(default_factory=list)
    lines_changed: int = 0

    def merge(self, other: "RunStats") -> None:
        """Add all counters from *other* into self (files_edited is not merged)."""
        self.if_not_else += other.if_not_else
        self.tuple_to_dataclass += other.tuple_to_dataclass
        self.duplicate_extracted += other.duplicate_extracted
        self.duplicate_matched += other.duplicate_matched
        self.function_split += other.function_split
        self.algorithmic_rejected += other.algorithmic_rejected
        self.llm_rejected += other.llm_rejected
        self.llm_veto_calls += other.llm_veto_calls
        self.llm_edit_calls += other.llm_edit_calls
        self.llm_verify_calls += other.llm_verify_calls

    @property
    def total_edits(self) -> int:
        return (
            self.if_not_else
            + self.tuple_to_dataclass
            + self.duplicate_extracted
            + self.duplicate_matched
            + self.function_split
        )

    @property
    def total_rejected(self) -> int:
        return self.algorithmic_rejected + self.llm_rejected

    @property
    def total_llm_calls(self) -> int:
        return self.llm_veto_calls + self.llm_edit_calls + self.llm_verify_calls

    def count_lines_changed(self, original: str, new: str) -> None:
        """Add the number of added/removed lines between *original* and *new*."""
        orig_lines = original.splitlines()
        new_lines = new.splitlines()
        diff = difflib.unified_diff(orig_lines, new_lines)
        for line in diff:
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                self.lines_changed += 1

    def format_summary(self) -> List[str]:
        """Return a list of lines forming the human-readable run summary."""
        lines = ["--- crispen summary ---"]
        lines.append("edits:")
        lines.append(f"  if not/else:         {self.if_not_else}")
        lines.append(f"  tuple to dataclass:  {self.tuple_to_dataclass}")
        lines.append(f"  duplicate extracted: {self.duplicate_extracted}")
        lines.append(f"  match existing:      {self.duplicate_matched}")
        lines.append(f"  function split:      {self.function_split}")
        lines.append(f"  total:               {self.total_edits}")
        lines.append("rejected:")
        lines.append(f"  algorithmic:         {self.algorithmic_rejected}")
        lines.append(f"  LLM:                 {self.llm_rejected}")
        lines.append(f"  total:               {self.total_rejected}")
        lines.append("LLM calls:")
        lines.append(f"  veto:                {self.llm_veto_calls}")
        lines.append(f"  edit:                {self.llm_edit_calls}")
        lines.append(f"  verify:              {self.llm_verify_calls}")
        lines.append(f"  total:               {self.total_llm_calls}")
        if self.files_edited:
            flist = ", ".join(self.files_edited)
            lines.append(f"files edited ({len(self.files_edited)}): {flist}")
        else:
            lines.append("files edited: none")
        lines.append(f"lines changed: {self.lines_changed}")
        return lines
