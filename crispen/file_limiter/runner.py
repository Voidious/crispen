"""FileLimiter runner: orchestrates phases 1–4 for a single file."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import CrispenConfig
from .advisor import GroupPlacement, advise_file_limiter, resolve_naming_conflicts
from .classifier import classify_entities
from .code_gen import SplitResult, generate_file_splits
from .entity_parser import Entity, EntityKind


@dataclass
class FileLimiterResult:
    """Output of :func:`run_file_limiter` for a single file."""

    original_source: str  # updated source for the original file
    new_files: Dict[str, str]  # {relative_path: source_code}
    messages: List[str] = field(default_factory=list)
    abort: bool = False
    subdir_name: Optional[str] = None  # set when files go into a subdirectory
    llm_calls: int = 0
    verified_functions: int = 0
    verified_classes: int = 0
    verified_lines: int = 0


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _strip_imports_by_line(src: str) -> str:
    """Return *src* with every import statement removed.

    Uses AST to locate the exact line range of each import node (correctly
    handling multi-line imports).  Returns *src* unchanged when it cannot be
    parsed as Python.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    remove: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for ln in range(node.lineno, node.end_lineno + 1):
                remove.add(ln)
    if not remove:
        return src
    lines = src.splitlines(keepends=True)
    return "".join(ln_text for i, ln_text in enumerate(lines, 1) if i not in remove)


def _verify_preservation(
    entities: List[Entity],
    split: SplitResult,
    post_source: str,
    placements: List[GroupPlacement],
) -> List[str]:
    """Return a list of failure descriptions (empty = all entities preserved).

    Checks that each entity's source (minus import statements) is present in
    the import-stripped combined output (original + new files).
    Empty entity sources (e.g. blank-line blocks) are skipped.
    TOP_LEVEL entities (import/docstring blocks) are always skipped because
    they are intentionally restructured during a split.

    Both sides have imports stripped before comparison so that post-generation
    pruning (``_prune_inline_redundant_imports``) — which removes function-body
    imports made redundant by file-level imports — does not produce false
    failures while still catching dropped functions or classes.

    Each failure is annotated with where the entity was expected to appear:
    ``migrated → <target>`` or ``stayed in original``.
    """
    lines = post_source.splitlines(keepends=True)
    combined_no_imports = _strip_imports_by_line(split.original_source)
    for content in split.new_files.values():
        combined_no_imports += _strip_imports_by_line(content)
    name_to_file: Dict[str, str] = {
        name: p.target_file for p in placements for name in p.group
    }
    failures: List[str] = []

    for entity in entities:
        if entity.kind == EntityKind.TOP_LEVEL:
            continue  # import/docstring blocks are intentionally restructured
        entity_src = "".join(lines[entity.start_line - 1 : entity.end_line]).rstrip()
        if not entity_src:
            continue
        entity_no_imports = _strip_imports_by_line(entity_src)
        if entity_no_imports not in combined_no_imports:
            preview_lines = entity_src.splitlines()[:3]
            preview = "\n    ".join(preview_lines)
            if len(entity_src.splitlines()) > 3:
                preview += "\n    ..."
            target = name_to_file.get(entity.name)
            loc = f"migrated \u2192 {target}" if target else "stayed in original"
            failures.append(
                f"  entity {entity.name!r} ({entity.kind.value},"
                f" lines {entity.start_line}\u2013{entity.end_line}) [{loc}]:\n"
                f"    {preview}"
            )

    return failures


def _is_whole_file_diff(diff_ranges: List[Tuple[int, int]], n_lines: int) -> bool:
    """Return True if *diff_ranges* cover every line from 1 to *n_lines*.

    A "whole file" diff occurs when the diff adds or replaces every line —
    e.g. when a file is brand-new or completely rewritten.  In this case
    FileLimiter can redirect all output to a subdirectory package instead of
    placing sibling files next to the original.
    """
    if not diff_ranges or n_lines == 0:
        return False
    covered_end = 0
    for start, end in sorted(diff_ranges):
        if start > covered_end + 1:
            return False  # gap — some lines are not in the diff
        covered_end = max(covered_end, end)
    return covered_end >= n_lines


def _detect_naming_conflicts(
    placements: List[GroupPlacement],
    existing_files: frozenset,
    existing_dirs: frozenset,
) -> List[str]:
    """Return naming conflict descriptions (empty = no conflicts).

    A conflict arises when a flat module ``foo.py`` and a package directory
    ``foo/`` would share the same Python import name, making them impossible
    to import together.  Checks conflicts both within the plan itself and
    against the existing filesystem.
    """
    file_stems: set = set()  # stems from flat *.py targets in the plan
    dir_tops: set = set()  # top-level directory names from subdir targets

    for p in placements:
        parts = Path(p.target_file).parts
        if len(parts) == 1:
            file_stems.add(Path(parts[0]).stem)
        else:
            dir_tops.add(parts[0])

    conflicts: List[str] = []

    # Plan-vs-plan: same name used as both a flat file and a directory.
    for stem in sorted(file_stems & dir_tops):
        conflicts.append(f"'{stem}.py' and '{stem}/' directory both appear in the plan")

    # Plan flat file vs. existing directory on disk.
    for stem in sorted(file_stems):
        if stem in existing_dirs:
            conflicts.append(
                f"target '{stem}.py' conflicts with existing directory '{stem}/'"
            )

    # Plan subdirectory vs. existing flat file on disk.
    for top in sorted(dir_tops):
        if f"{top}.py" in existing_files:
            conflicts.append(
                f"target '{top}/' directory conflicts with existing file '{top}.py'"
            )

    return conflicts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_file_limiter(
    filepath: str,
    original_source: str,
    post_source: str,
    diff_ranges: List[Tuple[int, int]],
    config: CrispenConfig,
    verbose: bool = False,
) -> FileLimiterResult:
    """Run all FileLimiter phases on a single file.

    1. Classify entities (Phases 1+2).
    2. Ask the LLM for a placement plan (Phase 3).
    3. Generate new file contents (Phase 4).
    4. Verify all entity sources are preserved.

    LLM-dependent steps (2 and 3) are retried up to
    ``config.file_limiter_retries`` additional times on transient failures.
    Deterministic failures (e.g. single-SCC abort) are never retried.
    On each failed attempt the SKIP message is accumulated; all attempt
    messages are returned when every attempt fails.  On success only the
    success messages are returned.

    Returns a :class:`FileLimiterResult` with ``abort=True`` when the file
    cannot be split or verification fails.  :class:`CrispenAPIError` from
    the LLM is propagated to the caller.
    """
    if verbose:
        print(
            f"crispen: FileLimiter: analyzing '{filepath}'",
            file=sys.stderr,
            flush=True,
        )

    classified = classify_entities(original_source, post_source, diff_ranges)
    source_dir = Path(filepath).parent
    source_name = Path(filepath).name
    existing_files: frozenset = frozenset(
        p.name for p in source_dir.glob("*.py") if p.name != source_name
    )
    existing_dirs: frozenset = frozenset(
        p.name for p in source_dir.iterdir() if p.is_dir()
    )

    if verbose:
        n_set2 = len(classified.set_2_groups)
        n_set3 = len(classified.set_3_groups)
        print(
            f"crispen: FileLimiter:   {n_set2} movable group(s),"
            f" {n_set3} modified group(s)",
            file=sys.stderr,
            flush=True,
        )

    # ---- Subdir-split detection ----
    is_test = source_name.startswith("test_")
    subdir_name: Optional[str] = None
    if config.file_limiter_subdir_split:
        n_lines = len(post_source.splitlines())
        if _is_whole_file_diff(diff_ranges, n_lines):
            stem = Path(source_name).stem
            subdir_name = stem[5:] if stem.startswith("test_") else stem
            subdir_path = source_dir / subdir_name
            if subdir_path.exists():
                return FileLimiterResult(
                    original_source=post_source,
                    new_files={},
                    messages=[
                        f"SKIP {filepath} (FileLimiter): target subdirectory"
                        f" '{subdir_name}/' already exists"
                    ],
                    abort=True,
                )
            # Brand-new directory: no pre-existing files or dirs to conflict with.
            existing_files = frozenset()
            existing_dirs = frozenset()

    # Deterministic failure — retrying the LLM would not help.
    if classified.abort:
        reason = f": {classified.abort_reason}" if classified.abort_reason else ""
        return FileLimiterResult(
            original_source=post_source,
            new_files={},
            messages=[f"SKIP {filepath} (FileLimiter): file cannot be split{reason}"],
            abort=True,
        )

    max_attempts = 1 + config.file_limiter_retries
    retry_msgs: List[str] = []
    last_abort: bool = True
    plan: Optional[object] = None
    split: Optional[SplitResult] = None
    prev_set3_failure: str = ""
    prev_placement_failure: str = ""
    total_llm_calls: int = 0
    # Mutable counter shared with resolve_naming_conflicts LLM calls.
    resolve_counter: List[int] = [0]

    for _attempt in range(max_attempts):
        plan = advise_file_limiter(
            classified,
            filepath,
            config,
            existing_files,
            prev_set3_failure=prev_set3_failure,
            prev_placement_failure=prev_placement_failure,
            verbose=verbose,
        )
        total_llm_calls += plan.llm_calls

        if plan.abort:
            reason = f": {plan.abort_reason}" if plan.abort_reason else ""
            retry_msgs.append(
                f"SKIP {filepath} (FileLimiter): file cannot be split{reason}"
            )
            last_abort = True
            if "set-3 groups" in plan.abort_reason:
                prev_set3_failure = (
                    "Your previous response was incomplete. "
                    "Please return a decision for every group."
                )
            else:
                prev_placement_failure = (
                    "Your previous response was incomplete. "
                    "Please return a target_file for every group, "
                    "do not use an existing filename."
                )
            continue

        if not plan.placements:
            if classified.set_2_groups or classified.set_3_groups:
                retry_msgs.append(
                    f"SKIP {filepath} (FileLimiter): no entities selected for migration"
                )
                last_abort = False
                prev_set3_failure = (
                    "In your previous attempt, you assigned 'stay' to all groups. "
                    "This is not acceptable. "
                    "You MUST assign 'migrate' to at least one group."
                )
                continue
            # Nothing movable regardless of LLM output — don't retry.
            return FileLimiterResult(
                original_source=post_source,
                new_files={},
                abort=False,
                messages=[],
                llm_calls=total_llm_calls,
            )

        # Apply test_ prefix so pytest can discover moved test files.
        if source_name.startswith("test_"):
            for p in plan.placements:
                target = Path(p.target_file)
                if not target.name.startswith("test_") and any(
                    name.startswith("test_") for name in p.group
                ):
                    p.target_file = str(target.parent / ("test_" + target.name))

        # In subdir-split mode, prefix all target files with the subdirectory.
        # This happens after the test_ prefix so names like "test_utils.py"
        # become "service/test_utils.py".
        if subdir_name is not None:
            for p in plan.placements:
                p.target_file = f"{subdir_name}/{p.target_file}"

        # Check for naming conflicts between flat modules and package directories.
        # foo.py and foo/ share the same import name and cannot coexist.
        conflicts = _detect_naming_conflicts(
            plan.placements, existing_files, existing_dirs
        )
        if conflicts:
            conflict_desc = "; ".join(conflicts)
            resolved = resolve_naming_conflicts(
                plan.placements,
                classified,
                filepath,
                existing_files,
                existing_dirs,
                config,
                verbose=verbose,
                _counter=resolve_counter,
            )
            total_llm_calls += resolve_counter[0]
            resolve_counter[0] = 0
            if resolved is not None:
                plan.placements = resolved  # fall through to generate_file_splits
            else:
                retry_msgs.append(
                    f"SKIP {filepath} (FileLimiter): naming conflicts: {conflict_desc}"
                )
                last_abort = True
                prev_placement_failure = (
                    f"Your previous plan has naming conflicts: {conflict_desc}. "
                    "A Python file 'foo.py' and a package directory 'foo/' share "
                    "the same import name and cannot coexist. "
                    "Please rename the conflicting targets."
                )
                continue

        split = generate_file_splits(
            classified, plan, post_source, filepath, subdir_name=subdir_name
        )

        if split.abort:
            reason = f": {split.abort_reason}" if split.abort_reason else ""
            retry_msgs.append(
                f"SKIP {filepath} (FileLimiter): file cannot be split{reason}"
            )
            last_abort = True
            prev_assignments = "; ".join(
                f"{', '.join(p.group)} \u2192 {p.target_file}" for p in plan.placements
            )
            prev_placement_failure = (
                f"Your previous assignments ({prev_assignments}) caused circular "
                "file imports. Please choose different target filenames."
            )
            continue

        # Success — keep retry_msgs so callers can see which attempts failed.
        break
    else:
        # All attempts exhausted.
        return FileLimiterResult(
            original_source=post_source,
            new_files={},
            messages=retry_msgs,
            abort=last_abort,
            llm_calls=total_llm_calls,
        )

    # For non-test subdir splits, redirect the post-split "original" source to
    # service/__init__.py and leave service.py untouched (it is shadowed by the
    # new package).  For test files the original test_*.py keeps re-export stubs.
    if subdir_name is not None and not is_test:
        split.new_files[f"{subdir_name}/__init__.py"] = split.original_source
        split.original_source = original_source

    if verbose:
        print(
            f"crispen: FileLimiter: verifying entity preservation in '{filepath}'",
            file=sys.stderr,
            flush=True,
        )

    failures = _verify_preservation(
        classified.entities, split, post_source, plan.placements
    )
    if failures:
        detail = "\n".join(failures)
        return FileLimiterResult(
            original_source=post_source,
            new_files={},
            messages=[f"SKIP {filepath} (FileLimiter): verification failed\n{detail}"],
            abort=True,
            llm_calls=total_llm_calls,
        )

    # Count entities that passed verification (non-TOP_LEVEL, non-empty).
    post_lines = post_source.splitlines(keepends=True)
    verified_functions = 0
    verified_classes = 0
    verified_lines = 0
    for entity in classified.entities:
        if entity.kind == EntityKind.TOP_LEVEL:
            continue
        entity_src = "".join(
            post_lines[entity.start_line - 1 : entity.end_line]
        ).rstrip()
        if not entity_src:
            continue
        if entity.kind == EntityKind.FUNCTION:
            verified_functions += 1
        if entity.kind == EntityKind.CLASS:
            verified_classes += 1
        verified_lines += entity.end_line - entity.start_line + 1

    msgs = [
        f"{filepath}: FileLimiter: moved {', '.join(p.group)} \u2192 {p.target_file}"
        for p in plan.placements
    ]
    return FileLimiterResult(
        original_source=split.original_source,
        new_files=split.new_files,
        messages=retry_msgs + msgs,
        abort=False,
        subdir_name=subdir_name,
        llm_calls=total_llm_calls,
        verified_functions=verified_functions,
        verified_classes=verified_classes,
        verified_lines=verified_lines,
    )
