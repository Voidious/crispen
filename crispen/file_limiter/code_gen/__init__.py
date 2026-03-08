"""Code generation for FileLimiter: build new files and update original source."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

from ..advisor import FileLimiterPlan, GroupPlacement
from ..classifier import ClassifiedEntities
from ..dep_graph import find_sccs
from ..entity_parser import EntityKind
from .splitting import (
    SplitResult,
    _add_re_exports,
    _find_main_block_entity,
    _find_main_direct_callees,
    _inject_inline_imports,
    _inject_inline_test_imports_original,
    _is_pytest_fixture,
    _is_test_name,
    _merge_conftest_sources,
    _split_cross_imports_by_test,
)
from .utils import (
    _FUTURE_IMPORT_LINE_RE,
    _abs_package_for_dir,
    _bump_relative_imports,
    _collect_external_imported_names,
    _collect_name_loads,
    _extract_import_info,
    _extract_module_docstring,
    _extract_shared_helpers,
    _find_cross_file_imports,
    _find_needed_imports,
    _merge_from_imports,
    _normalize_blank_lines,
    _prune_inline_redundant_imports,
    _prune_unused_imports,
    _remove_entity_lines,
    _strip_module_docstring,
    _strip_orphaned_section_headers,
    _strip_top_level_import_lines,
)
from .utils import ImportInfo  # fmt: skip # noqa: F401, E501
from .utils import _class_has_test_methods  # fmt: skip # noqa: F401, E501
from .utils import _find_project_root  # fmt: skip # noqa: F401, E501
from .utils import _import_derived_names  # fmt: skip # noqa: F401, E501
from .utils import _import_line_numbers  # fmt: skip # noqa: F401, E501
from .utils import _module_path_from_file  # fmt: skip # noqa: F401, E501
from .utils import _relative_import_prefix  # fmt: skip # noqa: F401, E501
from .utils import _target_module_name  # fmt: skip # noqa: F401, E501
from .utils import _topo_depth  # fmt: skip # noqa: F401, E501


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_file_splits(
    classified: ClassifiedEntities,
    plan: FileLimiterPlan,
    post_source: str,
    original_path: str,
    subdir_name: Optional[str] = None,
    pytest_conftest: bool = False,
    has_main: bool = False,
) -> SplitResult:
    """Generate new file contents and the updated original source.

    When *plan* is aborted or has no placements, returns :class:`SplitResult`
    with the original source unchanged (``abort`` mirrors ``plan.abort``).

    When *subdir_name* is set (e.g. ``"service"``), the file is being split
    into a package subdirectory.  The "original" file is treated as
    ``service/__init__.py`` for dependency-graph and import-prefix purposes,
    so cross-file imports within the package use correct relative paths.

    When *pytest_conftest* is True, any entity decorated with
    ``@pytest.fixture`` (or ``@fixture``) is redirected to ``conftest.py``
    instead of the LLM-assigned target file.  pytest auto-discovers fixtures
    from ``conftest.py``, so no re-export import is added to the original
    file — eliminating both the F401 and F811 flake8 warnings that arise
    when a fixture name is used as a test function parameter.
    """
    if plan.abort:
        return SplitResult(
            new_files={},
            original_source=post_source,
            abort=True,
            abort_reason=plan.abort_reason,
        )

    if not plan.placements:
        return SplitResult(new_files={}, original_source=post_source, abort=False)

    # Detect shebang on line 1 so it can be stripped from new files and
    # preserved (or restored) at the top of the original.
    shebang: Optional[str] = None
    if post_source.startswith("#!"):
        nl = post_source.find("\n")
        shebang = post_source[: nl + 1] if nl != -1 else post_source + "\n"

    lines = post_source.splitlines(keepends=True)
    entity_map = {e.name: e for e in classified.entities}

    # Build entity source map (name → stripped source string).
    entity_source_map: Dict[str, str] = {}
    for entity in classified.entities:
        entity_source_map[entity.name] = "".join(
            lines[entity.start_line - 1 : entity.end_line]
        ).rstrip()

    # All entity-defined names (used to limit import matching scope).
    all_entity_names: Set[str] = {
        name for e in classified.entities for name in e.names_defined
    }

    # Extract import info from post-refactor source.
    import_infos = _extract_import_info(post_source)

    # Placements whose target_file matches the original filename would create a
    # self-referential import (e.g. `from .duplicate_extractor import Foo` inside
    # duplicate_extractor.py).  Drop them — entities stay in the original file.
    # In subdir-split mode the "original" is the package __init__.py; use that
    # name throughout so dependency-graph edges and import prefixes are correct.
    original_basename = (
        f"{subdir_name}/__init__.py"
        if subdir_name and not has_main
        else Path(original_path).name
    )
    is_test_file = Path(original_path).name.startswith("test_")
    # For test-file subdir splits the original test file stays on disk (runner.py
    # does not redirect it to __init__.py), so non-migrated names still live in
    # the original file (e.g. "test_runner.py"), not in the package __init__.py.
    non_migrated_home = (
        Path(original_path).name
        if (subdir_name and is_test_file)
        else original_basename
    )
    # Identify the __main__ block and any functions it calls directly.
    # These stay in the original file unconditionally: the __main__ block
    # is an entry point the user expects to keep working, and its direct
    # callees must live in the same file to avoid module-level test-class
    # imports that would cause pytest double-discovery.
    main_entity = _find_main_block_entity(classified.entities, entity_source_map)
    main_sticky: Set[str] = set()
    if main_entity is not None:
        main_sticky.add(main_entity)
        function_entity_names = {
            e.name for e in classified.entities if e.kind == EntityKind.FUNCTION
        }
        main_sticky.update(
            _find_main_direct_callees(
                entity_source_map.get(main_entity, ""), function_entity_names
            )
        )

    valid_placements = [
        p
        for p in plan.placements
        if p.target_file != original_basename
        and not any(name in main_sticky for name in p.group)
    ]

    # --- Pytest conftest routing ---
    # When enabled, entities decorated with @pytest.fixture are redirected to
    # conftest.py instead of the LLM-assigned target file.  pytest discovers
    # fixtures from conftest.py automatically, so no re-export import is added
    # to the original file — eliminating the F401/F811 flake8 false positives.
    fixture_entity_names: Set[str] = set()
    if pytest_conftest:
        conftest_group: List[str] = []
        new_valid: List[GroupPlacement] = []
        for p in valid_placements:
            non_fixture: List[str] = []
            for name in p.group:
                src = entity_source_map.get(name, "")
                if src and _is_pytest_fixture(src):
                    fixture_entity_names.add(name)
                    conftest_group.append(name)
                else:
                    non_fixture.append(name)
            if non_fixture:
                new_valid.append(
                    GroupPlacement(group=non_fixture, target_file=p.target_file)
                )
        if conftest_group:
            new_valid.append(
                GroupPlacement(group=conftest_group, target_file="conftest.py")
            )
        valid_placements = new_valid

    # Group placements by target file (preserving order for topo sort).
    file_entity_names: Dict[str, List[str]] = {}
    for placement in valid_placements:
        file_entity_names.setdefault(placement.target_file, []).extend(placement.group)

    # All migrated entity names.
    migrated_names: Set[str] = {name for p in valid_placements for name in p.group}

    # Build name → target-file map for cross-file import detection.
    # Exclude import-derived names (_find_needed_imports handles those).
    import_defined_names = {name for info in import_infos for name in info.names}
    name_to_target_file: Dict[str, str] = {}
    for target_file, ent_names in file_entity_names.items():
        for ent_name in ent_names:
            entity = entity_map.get(ent_name)
            if entity:
                for defined_name in entity.names_defined:
                    if defined_name not in import_defined_names:
                        name_to_target_file[defined_name] = target_file

    # Also map names from non-migrated entities to the original file so that
    # split files can import helpers (e.g. _run) that stayed behind.
    for entity in classified.entities:
        if entity.name not in migrated_names:
            for defined_name in entity.names_defined:
                if defined_name not in import_defined_names:
                    name_to_target_file.setdefault(defined_name, non_migrated_home)

    # Extract non-migrated FUNCTION/CLASS entities referenced by migrated ones
    # into the new files that use them, breaking O→F→O import cycles.
    synthetic_placements = _extract_shared_helpers(
        file_entity_names,
        entity_source_map,
        entity_map,
        classified,
        name_to_target_file,
        migrated_names,
        non_migrated_home,
    )

    # Collect names that external files (outside the module being split) import
    # from the original file.  Private symbols in this set must get a re-export
    # proxy even though they are no longer referenced by the remaining source.
    external_loads = _collect_external_imported_names(original_path)

    # Detect circular imports.  Cycles can pass through the original file:
    # a new file that imports a non-migrated name from the original can form a
    # chain back to the original via the re-exports the original adds.
    # Model the original as an explicit node in the dependency graph.
    #
    # Original's outgoing edges: it will re-export a migrated name when the
    # name is public (no _/test_ prefix), referenced by a non-migrated entity,
    # or imported by an external file (external_loads).
    #
    # In a test-file subdir split, non_migrated_home ("test_runner.py") differs
    # from original_basename ("runner/__init__.py").  Re-exports are injected
    # into the original test file, so it—not the __init__.py—gains outgoing
    # import edges and must be a separate node in the dependency graph.
    non_migrated_loads: Set[str] = set()
    for ent_name, src in entity_source_map.items():
        if ent_name not in migrated_names:
            non_migrated_loads |= _collect_name_loads(src)

    reexport_home = non_migrated_home
    all_dep_nodes = set(file_entity_names.keys()) | {original_basename, reexport_home}
    file_deps: Dict[str, Set[str]] = {node: set() for node in all_dep_nodes}
    for target_file, ent_names in file_entity_names.items():
        for ent_name in ent_names:
            src = entity_source_map.get(ent_name, "")
            for ref_name in _collect_name_loads(src):
                dep_file = name_to_target_file.get(ref_name)
                if dep_file and dep_file != target_file and dep_file in file_deps:
                    file_deps[target_file].add(dep_file)
    for placement in valid_placements + synthetic_placements:
        for ent_name in placement.group:
            entity = entity_map.get(ent_name)
            if entity:
                for defined_name in entity.names_defined:
                    if (
                        (
                            not defined_name.startswith("_")
                            and not defined_name.startswith("test_")
                        )
                        or defined_name in non_migrated_loads
                        or defined_name in external_loads
                    ):
                        file_deps[reexport_home].add(placement.target_file)
                        break
    if any(len(scc) > 1 for scc in find_sccs(file_deps)):
        return SplitResult(
            new_files={},
            original_source=post_source,
            abort=True,
            abort_reason="proposed split would create circular file imports",
        )

    # Use absolute imports when the original file is a test file.  Pytest's
    # default import mode loads test files as top-level modules (not package
    # members), so relative imports like `from .helpers import foo` would
    # raise ImportError at collection time.
    abs_pkg: Optional[str] = None
    if Path(original_path).name.startswith("test_"):
        abs_pkg = _abs_package_for_dir(original_path)

    # In subdir-split mode, new files live inside a package subdirectory and
    # can use relative imports for cross-file references within that package.
    abs_pkg_for_new_files: Optional[str] = None if subdir_name else abs_pkg

    # Generate new file contents.
    new_files: Dict[str, str] = {}
    for target_file, ent_names in file_entity_names.items():
        needed = _find_needed_imports(
            ent_names, entity_source_map, import_infos, all_entity_names
        )
        if subdir_name is not None:
            depth = len(Path(target_file).parts) - 1
            needed = [_bump_relative_imports(s, depth) for s in needed]
        entity_srcs = []
        top_cross: List[str] = []
        seen_top_cross: Set[str] = set()
        for _ent_name in ent_names:
            _src = entity_source_map.get(_ent_name)
            if _src is None:
                continue
            _entity = entity_map.get(_ent_name)
            if _entity and _entity.kind == EntityKind.TOP_LEVEL:
                # Imports are emitted separately by _find_needed_imports; strip
                # them from the entity body to prevent duplicate import stmts.
                _src = _strip_top_level_import_lines(_src)
                if subdir_name is not None:
                    # In subdir-split mode the module docstring belongs in
                    # __init__.py rather than in one of the child modules.
                    _src = _strip_module_docstring(_src)
            else:
                _src = _FUTURE_IMPORT_LINE_RE.sub("", _src)
            # Strip shebang from any entity that begins on line 1 of the
            # original source — it must not appear in generated new files.
            if shebang and _entity and _entity.start_line == 1:
                nl = _src.find("\n")
                _src = _src[nl + 1 :] if nl != -1 else ""
            _src = _src.rstrip()
            # Compute cross-file imports for this entity and split off any
            # test-named symbols (Test* / test_*) to be injected inline.
            entity_cross = _find_cross_file_imports(
                [_ent_name],
                entity_source_map,
                name_to_target_file,
                target_file,
                abs_pkg=abs_pkg_for_new_files,
            )
            ent_top_cross, ent_test_imports = _split_cross_imports_by_test(entity_cross)
            for imp in ent_top_cross:
                if imp not in seen_top_cross:
                    seen_top_cross.add(imp)
                    top_cross.append(imp)
            if ent_test_imports and _entity and _entity.kind != EntityKind.TOP_LEVEL:
                _src = _inject_inline_imports(_src, ent_test_imports)
            else:
                # TOP_LEVEL entity: no body scope, fall back to module level.
                for imp in ent_test_imports:
                    if imp not in seen_top_cross:
                        seen_top_cross.add(imp)
                        top_cross.append(imp)
            entity_srcs.append(_src)
        entity_srcs = [s for s in entity_srcs if s]
        parts: List[str] = []
        all_imports = _merge_from_imports(needed + top_cross)
        if all_imports:
            parts.append("\n".join(all_imports))
        parts.extend(entity_srcs)
        pruned = _prune_unused_imports("\n\n\n".join(parts) + "\n")
        new_files[target_file] = _prune_inline_redundant_imports(pruned)

    # If an existing conftest.py is present on disk, merge intelligently so
    # that duplicate imports and fixture definitions are not repeated (which
    # would cause flake8 F811/E402 errors when multiple splits write to the
    # same conftest.py file).
    if "conftest.py" in new_files:
        existing_conftest = Path(original_path).parent / "conftest.py"
        if existing_conftest.exists():
            prior = existing_conftest.read_text(encoding="utf-8")
            new_files["conftest.py"] = _merge_conftest_sources(
                prior, new_files["conftest.py"]
            )

    # Build updated original source.
    updated = _remove_entity_lines(
        post_source, migrated_names, entity_map, entity_source_map
    )
    updated = _prune_unused_imports(updated)
    # For non-test subdir splits, re-exports from the __init__.py use relative
    # import prefixes computed from inside the package (e.g. ".utils" not
    # ".service.utils").  For test files the original keeps existing abs_pkg
    # behaviour so pytest can find the re-exported symbols.
    # In a non-test subdir split the updated source becomes subdir/__init__.py,
    # which sits one directory level deeper than the original file.  Any
    # relative imports it still contains (e.g. ``from .. import llm_client``
    # or ``from .base import Foo``) therefore need one extra dot so they keep
    # pointing at the same modules.  Re-exports added by _add_re_exports below
    # are already computed from the __init__.py's perspective and are correct.
    if subdir_name is not None and not is_test_file and not has_main:
        updated = _bump_relative_imports(updated)
    if subdir_name is not None:
        # If the original file had a module docstring and it was migrated away,
        # place it in subdir/__init__.py in both cases: for non-test splits
        # the docstring is prepended to `updated` which runner.py redirects to
        # __init__.py; for test splits it is written directly to __init__.py
        # (runner.py does not redirect `updated` for test files).
        _module_doc = _extract_module_docstring(post_source)
        if _module_doc and not _extract_module_docstring(updated):
            if is_test_file:
                new_files[f"{subdir_name}/__init__.py"] = _module_doc + "\n"
            else:
                updated = _module_doc + "\n\n" + updated
    relative_from: Optional[str] = (
        f"{subdir_name}/__init__.py"
        if (subdir_name and not is_test_file and not has_main)
        else None
    )
    # Exclude conftest.py from re-exports: fixtures there are auto-discovered
    # by pytest and must not be imported back into the original test file.
    re_export_placements = [
        p
        for p in valid_placements + synthetic_placements
        if p.target_file != "conftest.py"
    ]
    updated = _add_re_exports(
        updated,
        re_export_placements,
        entity_map,
        entity_source_map,
        external_loads=external_loads,
        abs_pkg=abs_pkg,
        relative_from=relative_from,
    )

    # For non-migrated entities that reference test-named symbols now living
    # in new files: _add_re_exports intentionally skips re-exporting them
    # (to avoid double-discovery), so inject those imports inline instead.
    migrated_test_symbols = {
        name: tfile
        for name, tfile in name_to_target_file.items()
        if tfile != original_basename and _is_test_name(name)
    }
    updated = _inject_inline_test_imports_original(
        updated, migrated_test_symbols, abs_pkg, original_basename
    )

    # Restore shebang at line 1 of the original.  It may have been removed
    # by _remove_entity_lines if the entity owning line 1 was migrated.
    if shebang and not updated.startswith("#!"):
        updated = shebang + updated

    # Remove section header comment blocks that became orphaned after entity
    # removal (nothing substantive remains beneath them).
    new_files = {f: _strip_orphaned_section_headers(s) for f, s in new_files.items()}
    updated = _strip_orphaned_section_headers(updated)

    # Normalize blank lines: collapse 3+ consecutive blank lines to 2 and
    # ensure exactly one trailing newline in every generated file.
    new_files = {f: _normalize_blank_lines(s) for f, s in new_files.items()}
    updated = _normalize_blank_lines(updated)

    return SplitResult(new_files=new_files, original_source=updated, abort=False)
