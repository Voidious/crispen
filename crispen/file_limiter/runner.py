"""FileLimiter runner: orchestrates phases 1–4 for a single file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import CrispenConfig
from .advisor import GroupPlacement, advise_file_limiter
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


def _verify_preservation(
    entities: List[Entity],
    split: SplitResult,
    post_source: str,
    placements: List[GroupPlacement],
) -> List[str]:
    """Return a list of failure descriptions (empty = all entities preserved).

    Checks that each entity's source text from *post_source* is present in
    either ``split.original_source`` or one of ``split.new_files.values()``.
    Empty entity sources (e.g. blank-line blocks) are skipped.
    TOP_LEVEL entities (import/docstring blocks) are always skipped because
    they are intentionally restructured during a split.
    Each failure is annotated with where the entity was expected to appear:
    ``migrated → <target>`` or ``stayed in original``.
    """
    lines = post_source.splitlines(keepends=True)
    combined = split.original_source + "".join(split.new_files.values())
    name_to_file: Dict[str, str] = {
        name: p.target_file for p in placements for name in p.group
    }
    failures: List[str] = []

    for entity in entities:
        if entity.kind == EntityKind.TOP_LEVEL:
            continue  # import/docstring blocks are intentionally restructured
        entity_src = "".join(lines[entity.start_line - 1 : entity.end_line]).rstrip()
        if entity_src and entity_src not in combined:
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
