"""FileLimiter runner: orchestrates phases 1–4 for a single file."""

from __future__ import annotations

import ast
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
    classified = classify_entities(original_source, post_source, diff_ranges)
    source_dir = Path(filepath).parent
    source_name = Path(filepath).name
    existing_files: frozenset = frozenset(
        p.name for p in source_dir.glob("*.py") if p.name != source_name
    )
    existing_dirs: frozenset = frozenset(
        p.name for p in source_dir.iterdir() if p.is_dir()
    )

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

    for _attempt in range(max_attempts):
        plan = advise_file_limiter(
            classified,
            filepath,
            config,
            existing_files,
            prev_set3_failure=prev_set3_failure,
            prev_placement_failure=prev_placement_failure,
        )

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
            )

        # Apply test_ prefix so pytest can discover moved test files.
        if source_name.startswith("test_"):
            for p in plan.placements:
                target = Path(p.target_file)
                if not target.name.startswith("test_") and any(
                    name.startswith("test_") for name in p.group
                ):
                    p.target_file = str(target.parent / ("test_" + target.name))

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
            )
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

        split = generate_file_splits(classified, plan, post_source, filepath)

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
        )

    msgs = [
        f"{filepath}: FileLimiter: moved {', '.join(p.group)} \u2192 {p.target_file}"
        for p in plan.placements
    ]
    return FileLimiterResult(
        original_source=split.original_source,
        new_files=split.new_files,
        messages=retry_msgs + msgs,
        abort=False,
    )
