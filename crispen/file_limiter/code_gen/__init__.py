"""Code generation for FileLimiter: build new files and update original source."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from ...import_sort import _sort_imports_pep8
from ..advisor import FileLimiterPlan, GroupPlacement
from ..classifier import ClassifiedEntities
from ..dep_graph import find_sccs
from ..entity_parser import EntityKind
from .conftest_merge import _merge_conftest_sources  # fmt: skip # noqa: F401, E501
from .conftest_merge import _rewrite_module_level_stores  # fmt: skip # noqa: F401, E501
from .conftest_merge import _rewrite_module_var_names  # fmt: skip # noqa: F401, E501
from .import_utils import _FROM_IMPORT_RE
from .import_utils import ImportInfo  # fmt: skip # noqa: F401, E501
from .import_utils import _abs_package_for_dir  # fmt: skip # noqa: F401, E501
from .import_utils import _bump_relative_imports  # fmt: skip # noqa: F401, E501
from .import_utils import _collect_external_imported_names  # fmt: skip # noqa: F401, E501
from .import_utils import _extract_import_info  # fmt: skip # noqa: F401, E501
from .import_utils import _find_cross_file_imports  # fmt: skip # noqa: F401, E501
from .import_utils import _find_cross_file_type_checking_imports  # fmt: skip # noqa: F401, E501
from .import_utils import _find_needed_imports  # fmt: skip # noqa: F401, E501
from .import_utils import _find_project_root  # fmt: skip # noqa: F401, E501
from .import_utils import _find_type_checking_needed_imports  # fmt: skip # noqa: F401, E501
from .import_utils import _import_derived_names  # fmt: skip # noqa: F401, E501
from .import_utils import _inject_module_level_imports  # fmt: skip # noqa: F401, E501
from .import_utils import _inject_type_checking_imports  # fmt: skip # noqa: F401, E501
from .import_utils import _merge_from_imports  # fmt: skip # noqa: F401, E501
from .import_utils import _module_import_stmt  # fmt: skip # noqa: F401, E501
from .import_utils import _module_path_from_file  # fmt: skip # noqa: F401, E501
from .import_utils import _narrow_import_source  # fmt: skip # noqa: F401, E501
from .import_utils import _relative_import_prefix  # fmt: skip # noqa: F401, E501
from .import_utils import _target_module_name  # fmt: skip # noqa: F401, E501
from .main_handling import _class_has_test_methods  # fmt: skip # noqa: F401, E501
from .main_handling import _file_has_only_fixtures  # fmt: skip # noqa: F401, E501
from .main_handling import _find_main_block_entity  # fmt: skip # noqa: F401, E501
from .main_handling import _find_main_direct_callees  # fmt: skip # noqa: F401, E501
from .main_handling import _inject_inline_imports  # fmt: skip # noqa: F401, E501
from .main_handling import _inject_inline_test_imports_original  # fmt: skip # noqa: F401, E501
from .main_handling import _is_pytest_fixture  # fmt: skip # noqa: F401, E501
from .main_handling import _is_test_name  # fmt: skip # noqa: F401, E501
from .main_handling import _split_cross_imports_by_test  # fmt: skip # noqa: F401, E501
from .name_analysis import _collect_name_loads  # fmt: skip # noqa: F401, E501
from .name_analysis import _collect_name_stores  # fmt: skip # noqa: F401, E501
from .name_analysis import _collect_quoted_annotation_names  # fmt: skip # noqa: F401, E501
from .name_analysis import _test_names_in_decorators  # fmt: skip # noqa: F401, E501
from .rewrite_utils import _add_re_exports  # fmt: skip # noqa: F401, E501
from .rewrite_utils import _extract_shared_helpers  # fmt: skip # noqa: F401, E501
from .rewrite_utils import _import_line_numbers  # fmt: skip # noqa: F401, E501
from .rewrite_utils import _prune_inline_redundant_imports  # fmt: skip # noqa: F401, E501
from .rewrite_utils import _prune_unused_imports  # fmt: skip # noqa: F401, E501
from .rewrite_utils import _remove_entity_lines  # fmt: skip # noqa: F401, E501
from .rewrite_utils import _strip_top_level_import_lines  # fmt: skip # noqa: F401, E501
from .rewrite_utils import _topo_depth  # fmt: skip # noqa: F401, E501
from .source_utils import _FUTURE_IMPORT_LINE_RE
from .source_utils import SplitResult  # fmt: skip # noqa: F401, E501
from .source_utils import _EXCESS_BLANK_BODY_RE  # fmt: skip # noqa: F401, E501
from .source_utils import _EXCESS_BLANK_RE  # fmt: skip # noqa: F401, E501
from .source_utils import _extract_module_docstring  # fmt: skip # noqa: F401, E501
from .source_utils import _multiline_string_ranges  # fmt: skip # noqa: F401, E501
from .source_utils import _normalize_blank_lines  # fmt: skip # noqa: F401, E501
from .source_utils import _source_is_only_docstring  # fmt: skip # noqa: F401, E501
from .source_utils import _strip_module_docstring  # fmt: skip # noqa: F401, E501
from .source_utils import _strip_orphaned_indented_comments  # fmt: skip # noqa: F401, E501
from .source_utils import _strip_orphaned_section_headers  # fmt: skip # noqa: F401, E501
from .source_utils import _sub_skip_strings  # fmt: skip # noqa: F401, E501


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
    reexport_mode: str = "always",
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
    #
    # Exception: if conftest.py already defines a function with the same name
    # (e.g. a default fixture that this test file overrides), routing the entity
    # there would cause _merge_conftest_sources to silently drop the new version
    # (keeping the old one), so the entity would disappear from the split output
    # entirely and verification would fail.  In that case the fixture stays in
    # its LLM-assigned target file; re-exports are still suppressed to avoid
    # F401/F811 (the fixture is injected by pytest name-lookup, not by import).
    # For subdir splits, route fixtures into the subdirectory's own conftest.py
    # so that multiple test files in the same parent directory each get an
    # isolated conftest scope and cannot overwrite each other's fixtures.
    # Exception to the exception: if the fixture is still referenced in entities
    # that remain in the original file (i.e. tests that were not migrated), route
    # it to the parent conftest.py instead so those tests can find it.  Tests in
    # the subdirectory also inherit from the parent conftest, so this is safe.
    # Further exception: if the fixture is referenced in remaining source AND the
    # parent conftest already has a fixture with the same name (the module was
    # overriding it), merging into parent conftest would silently keep the old
    # version — the original test would get the wrong fixture and the migrated
    # subdir tests would also inherit the wrong version.  In that case, copy
    # (don't move) the fixture to the subdir conftest so migrated tests get the
    # override; the entity is also kept in the original file so the original
    # test can discover it directly from its own module.
    conftest_target = f"{subdir_name}/conftest.py" if subdir_name else "conftest.py"
    parent_conftest_target = "conftest.py"
    existing_conftest_path = (
        Path(original_path).parent / subdir_name / "conftest.py"
        if subdir_name
        else Path(original_path).parent / "conftest.py"
    )
    fixture_entity_names: Set[str] = set()
    # Names of fixtures kept in their LLM-assigned file because the target
    # conftest.py already defines a symbol with that name.  Fixtures injected
    # by pytest name-lookup need no re-export import in the original file.
    conftest_conflict_names: Set[str] = set()
    # Names of fixtures that are copied to the subdir conftest but also kept in
    # the original file (not removed).  This handles the case where the fixture
    # overrides a parent conftest fixture and is also referenced in remaining
    # entities: the subdir copy ensures migrated tests see the override; the
    # original file copy ensures the original test's module-level fixture takes
    # precedence over the stale parent conftest version.
    copy_not_migrate: Set[str] = set()
    if pytest_conftest:
        existing_conftest_names: Set[str] = set()
        if existing_conftest_path.exists():
            try:
                _ec_tree = ast.parse(existing_conftest_path.read_text(encoding="utf-8"))
                for _ec_node in _ec_tree.body:
                    if isinstance(
                        _ec_node,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                    ):
                        existing_conftest_names.add(_ec_node.name)
            except (SyntaxError, OSError):
                pass

        # For subdir splits: build source of entities that will remain in the
        # original file so we can detect fixtures still needed by those entities.
        # Also load parent conftest names to detect fixtures that override a
        # parent-level fixture (used below to avoid silently keeping the old
        # parent version when merging would drop the new override).
        _migrating_names: Set[str] = {
            name for p in valid_placements for name in p.group
        }
        _remaining_src = "\n".join(
            entity_source_map[e.name]
            for e in classified.entities
            if e.name not in _migrating_names and e.name in entity_source_map
        )
        _parent_conftest_names: Set[str] = set()
        if subdir_name:
            _parent_conftest_path = Path(original_path).parent / "conftest.py"
            if _parent_conftest_path.exists():
                try:
                    _pc_tree = ast.parse(
                        _parent_conftest_path.read_text(encoding="utf-8")
                    )
                    for _pc_node in _pc_tree.body:
                        if isinstance(
                            _pc_node,
                            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                        ):
                            _parent_conftest_names.add(_pc_node.name)
                except (SyntaxError, OSError):
                    pass

        conftest_group: List[str] = []
        conftest_group_parent: List[str] = []
        new_valid: List[GroupPlacement] = []
        for p in valid_placements:
            non_fixture: List[str] = []
            for name in p.group:
                src = entity_source_map.get(name, "")
                if src and _is_pytest_fixture(src):
                    fixture_entity_names.add(name)
                    if name in existing_conftest_names:
                        # conftest already has this name: keep in the
                        # LLM-assigned file to avoid losing the entity.
                        # No re-export needed — pytest discovers fixtures
                        # by name-lookup, not by import.
                        non_fixture.append(name)
                        conftest_conflict_names.add(name)
                    elif subdir_name and re.search(
                        r"\b" + re.escape(name) + r"\b", _remaining_src
                    ):
                        # Fixture still used by non-migrated tests.
                        if name in _parent_conftest_names:
                            # Parent conftest already has this name: the module
                            # was overriding it.  Merging into parent conftest
                            # would silently keep the old version.  Instead,
                            # copy (don't move) to subdir conftest — migrated
                            # tests get the override via the subdir conftest;
                            # the entity stays in the original file so the
                            # original test discovers the override from its own
                            # module rather than the stale parent conftest entry.
                            conftest_group.append(name)
                            copy_not_migrate.add(name)
                        else:
                            # No conflict: route to the parent conftest.py so
                            # both original and subdir tests can find it.
                            conftest_group_parent.append(name)
                    else:
                        conftest_group.append(name)
                else:
                    non_fixture.append(name)
            if non_fixture:
                new_valid.append(
                    GroupPlacement(group=non_fixture, target_file=p.target_file)
                )
        if conftest_group:
            new_valid.append(
                GroupPlacement(group=conftest_group, target_file=conftest_target)
            )
        if conftest_group_parent:
            new_valid.append(
                GroupPlacement(
                    group=conftest_group_parent, target_file=parent_conftest_target
                )
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

    # For TOP_LEVEL names that are reassigned (stored) by a *different* entity,
    # cross-file references must use ``module.NAME`` so that the mutation is
    # visible to all importers.  Names that are only ever defined once and never
    # mutated elsewhere use a plain ``from .module import NAME`` — the idiomatic
    # Python form — since the value is stable after import.
    _top_level_def_entity: Dict[str, str] = {
        defined_name: entity.name
        for entity in classified.entities
        if entity.kind == EntityKind.TOP_LEVEL
        for defined_name in entity.names_defined
        if defined_name not in import_defined_names
    }
    top_level_var_names: Set[str] = {
        name
        for name, def_ent in _top_level_def_entity.items()
        for ent_name, src in entity_source_map.items()
        if ent_name != def_ent and name in _collect_name_stores(src)
    }

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
                    reexport_unconditionally = (
                        not defined_name.startswith("_")
                        and not defined_name.startswith("test_")
                        and (
                            reexport_mode == "always"
                            or (reexport_mode == "application" and not is_test_file)
                        )
                    )
                    if (
                        reexport_unconditionally
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
    entity_name_rewrites: Dict[str, Dict[str, str]] = {}  # per-entity rewrites
    for target_file, ent_names in file_entity_names.items():
        needed = _find_needed_imports(
            ent_names, entity_source_map, import_infos, all_entity_names
        )
        needed_tc = _find_type_checking_needed_imports(
            ent_names, entity_source_map, import_infos
        )
        if subdir_name is not None:
            depth = len(Path(target_file).parts) - 1
            needed = [_bump_relative_imports(s, depth) for s in needed]
            needed_tc = [_bump_relative_imports(s, depth) for s in needed_tc]
        entity_srcs = []
        top_cross: List[str] = []
        seen_top_cross: Set[str] = set()
        all_tc_imports: List[str] = list(needed_tc)
        seen_tc: Set[str] = set(needed_tc)
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
            entity_from, entity_mod, entity_rewrites = _find_cross_file_imports(
                [_ent_name],
                entity_source_map,
                name_to_target_file,
                target_file,
                abs_pkg=abs_pkg_for_new_files,
                top_level_var_names=top_level_var_names,
            )
            for _tc_imp in _find_cross_file_type_checking_imports(
                [_ent_name],
                entity_source_map,
                name_to_target_file,
                target_file,
                abs_pkg=abs_pkg_for_new_files,
                top_level_var_names=top_level_var_names,
            ):
                if _tc_imp not in seen_tc:
                    seen_tc.add(_tc_imp)
                    all_tc_imports.append(_tc_imp)
            if entity_rewrites:
                _src = _rewrite_module_var_names(_src, entity_rewrites)
                entity_name_rewrites[_ent_name] = entity_rewrites
            # Module imports for TOP_LEVEL vars must always be at module level
            # (never inlined) — decorators are evaluated before function bodies run.
            for imp in entity_mod:
                if imp not in seen_top_cross:
                    seen_top_cross.add(imp)
                    top_cross.append(imp)
            ent_top_cross, ent_test_imports = _split_cross_imports_by_test(entity_from)
            for imp in ent_top_cross:
                if imp not in seen_top_cross:
                    seen_top_cross.add(imp)
                    top_cross.append(imp)
            if ent_test_imports and _entity and _entity.kind != EntityKind.TOP_LEVEL:
                # Extract the test-named symbols that would be inlined.
                # All items from _split_cross_imports_by_test are "from X import Y"
                # form, so partitioning on " import " is safe.
                inlined_names: Set[str] = set()
                for _imp in ent_test_imports:
                    _, _, _names_part = _imp.partition(" import ")
                    for _n in _names_part.split(","):
                        inlined_names.add(_n.strip())
                # Decorators are evaluated before function bodies, so a symbol
                # that only arrives via an inline import will not be in scope.
                dec_conflicts = _test_names_in_decorators(_src, inlined_names)
                if dec_conflicts:
                    _names_str = ", ".join(f"'{n}'" for n in sorted(dec_conflicts))
                    return SplitResult(
                        new_files={},
                        original_source=post_source,
                        abort=True,
                        abort_reason=(
                            f"cannot split '{_ent_name}': {_names_str} appear(s) "
                            f"in a decorator but would need to be imported inline "
                            f"to avoid pytest collecting them as duplicate tests — "
                            f"keep the dependent test classes in the same file"
                        ),
                    )
                _src = _inject_inline_imports(_src, ent_test_imports)
            else:
                # TOP_LEVEL entity: no body scope, fall back to module level.
                for imp in ent_test_imports:
                    if imp not in seen_top_cross:
                        seen_top_cross.add(imp)
                        top_cross.append(imp)
            entity_srcs.append(_src)
        entity_srcs = [s for s in entity_srcs if s]
        # Dedup: remove TC imports for names already covered by regular imports.
        # This can happen when one entity uses a name at runtime (→ top_cross)
        # while another entity in the same file only uses it in a quoted
        # annotation (→ all_tc_imports), producing duplicate import statements.
        if all_tc_imports and (needed or top_cross):
            _regular_names: Set[str] = set()
            for _imp in needed + top_cross:
                _m = _FROM_IMPORT_RE.match(_imp)
                if _m:
                    _regular_names.update(
                        n.strip() for n in _m.group(2).split(",") if n.strip()
                    )
            _deduped_tc: List[str] = []
            for _tc in all_tc_imports:
                _m = _FROM_IMPORT_RE.match(_tc)
                if _m:
                    _tc_names = {
                        _n.strip() for _n in _m.group(2).split(",") if _n.strip()
                    }
                    _leftover = _tc_names - _regular_names
                    if _leftover:
                        _deduped_tc.append(
                            _tc
                            if _leftover == _tc_names
                            else _narrow_import_source(_tc, _leftover)
                        )
                else:
                    _deduped_tc.append(_tc)
            all_tc_imports = _deduped_tc
        parts: List[str] = []
        imports_for_sort = list(needed + top_cross)
        if all_tc_imports:
            imports_for_sort.append("from typing import TYPE_CHECKING")
        all_imports = _sort_imports_pep8(_merge_from_imports(imports_for_sort))
        if all_imports:
            parts.append("\n".join(all_imports))
        if all_tc_imports:
            tc_sorted = _sort_imports_pep8(_merge_from_imports(all_tc_imports))
            tc_block = "if TYPE_CHECKING:\n" + "\n".join("    " + s for s in tc_sorted)
            parts.append(tc_block)
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
    # copy_not_migrate fixtures are written to the subdir conftest (so they
    # appear in migrated_names / file_entity_names) but must NOT be removed
    # from the original file — the original test discovers them via the test
    # module itself, which takes precedence over the parent conftest's stale
    # base version.
    updated = _remove_entity_lines(
        post_source, migrated_names - copy_not_migrate, entity_map, entity_source_map
    )
    updated = _prune_unused_imports(updated)
    # Compute TYPE_CHECKING imports needed by non-migrated entities that had
    # their import guard block removed as part of a migrated TOP_LEVEL entity.
    # Injection happens AFTER the relative-import bump below so that bumped
    # import strings are passed to _inject_type_checking_imports rather than
    # relying on the bump (which only matches unindented ``from .`` lines and
    # therefore misses indented imports inside an ``if TYPE_CHECKING:`` block).
    _non_migrated_names = [
        e.name for e in classified.entities if e.name not in migrated_names
    ]
    _tc_to_inject: List[str] = []
    if _non_migrated_names:
        _tc_to_inject = _find_type_checking_needed_imports(
            _non_migrated_names, entity_source_map, import_infos
        )
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
        _tc_to_inject = [_bump_relative_imports(imp) for imp in _tc_to_inject]
    if _tc_to_inject:
        updated = _inject_type_checking_imports(updated, _tc_to_inject)
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
        elif is_test_file and _source_is_only_docstring(updated):
            # All entities migrated; the only thing remaining in the original
            # is the module docstring (a TOP_LEVEL entity that was never
            # removed by _remove_entity_lines).  Route it to __init__.py and
            # clear the original so the engine deletes it.
            new_files[f"{subdir_name}/__init__.py"] = (
                _extract_module_docstring(updated) + "\n"
            )
            updated = _strip_module_docstring(updated)
    relative_from: Optional[str] = (
        f"{subdir_name}/__init__.py"
        if (subdir_name and not is_test_file and not has_main)
        else None
    )
    # Apply module-qualified rewrites to the original file for any non-migrated
    # entity that reassigns a TOP_LEVEL name that was moved to a sub-file.
    # This is symmetric with the same treatment for new sub-files: when a name
    # is in top_level_var_names (i.e. reassigned somewhere), all files that
    # reference it — including the non-migrated home — use ``module.NAME``.
    if top_level_var_names:
        non_migrated_entity_names = [
            e.name for e in classified.entities if e.name not in migrated_names
        ]
        if non_migrated_entity_names:
            _orig_from, orig_mod_imports, orig_rewrites = _find_cross_file_imports(
                non_migrated_entity_names,
                entity_source_map,
                name_to_target_file,
                non_migrated_home,
                abs_pkg=abs_pkg,
                top_level_var_names=top_level_var_names,
            )
            if orig_rewrites:
                updated = _rewrite_module_var_names(updated, orig_rewrites)
                updated = _rewrite_module_level_stores(updated, orig_rewrites)
            if orig_mod_imports:
                updated = _inject_module_level_imports(updated, orig_mod_imports)

    # Exclude conftest.py from re-exports: fixtures there are auto-discovered
    # by pytest and must not be imported back into the original test file.
    # Also exclude conftest-conflict fixtures (kept in LLM-assigned file because
    # conftest already has that name): pytest discovers them by name-lookup, so
    # re-exporting them from the original file would be dead code and prevent
    # the original from being cleaned up / deleted.
    _conftest_files = {conftest_target, parent_conftest_target}
    re_export_placements = []
    for p in valid_placements + synthetic_placements:
        if p.target_file in _conftest_files:
            continue
        if conftest_conflict_names:
            filtered_group = [n for n in p.group if n not in conftest_conflict_names]
            if not filtered_group:
                continue
            p = GroupPlacement(group=filtered_group, target_file=p.target_file)
        re_export_placements.append(p)
    updated = _add_re_exports(
        updated,
        re_export_placements,
        entity_map,
        entity_source_map,
        external_loads=external_loads,
        abs_pkg=abs_pkg,
        relative_from=relative_from,
        is_test_file=is_test_file,
        reexport_mode=reexport_mode,
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

    # Remove indented comment lines that ended up at module level after entity
    # migration (flake8 E116: unexpected indentation: comment).
    new_files = {f: _strip_orphaned_indented_comments(s) for f, s in new_files.items()}
    updated = _strip_orphaned_indented_comments(updated)

    # When pytest_conftest is active and the test file's remaining content
    # contains only fixtures (no test functions, no other definitions), the
    # file has become dead code — all tests migrated away and the fixture is
    # stranded.  Route the remaining content to conftest.py (merging to avoid
    # duplicates if it is already there) and empty the original so the engine
    # deletes it.
    if (
        is_test_file
        and pytest_conftest
        and updated
        and _file_has_only_fixtures(updated)
    ):
        if conftest_target in new_files:
            new_files[conftest_target] = _merge_conftest_sources(
                new_files[conftest_target], updated
            )
        elif existing_conftest_path.exists():
            prior = existing_conftest_path.read_text(encoding="utf-8")
            new_files[conftest_target] = _merge_conftest_sources(prior, updated)
        else:
            new_files[conftest_target] = updated
        updated = ""

    # Normalize blank lines: collapse 3+ consecutive blank lines to 2 and
    # ensure exactly one trailing newline in every generated file.
    new_files = {f: _normalize_blank_lines(s) for f, s in new_files.items()}
    updated = _normalize_blank_lines(updated)

    return SplitResult(
        new_files=new_files,
        original_source=updated,
        abort=False,
        entity_name_rewrites=entity_name_rewrites,
        actual_placements=valid_placements + synthetic_placements,
    )
