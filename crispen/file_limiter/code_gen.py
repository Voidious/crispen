"""Code generation for FileLimiter: build new files and update original source."""

from __future__ import annotations
from .imports import ImportInfo  # noqa F401
from .imports import _add_re_exports  # noqa F401
from .imports import _collect_external_imported_names  # noqa F401
from .imports import _collect_name_loads  # noqa F401
from .imports import _extract_import_info  # noqa F401
from .imports import _find_cross_file_imports  # noqa F401
from .imports import _find_needed_imports  # noqa F401
from .imports import _import_derived_names  # noqa F401
from .imports import _import_line_numbers  # noqa F401
from .imports import _prune_inline_redundant_imports  # noqa F401
from .imports import _prune_unused_imports  # noqa F401
from .module_paths import _abs_package_for_dir  # noqa F401
from .module_paths import _find_project_root  # noqa F401
from .module_paths import _module_path_from_file  # noqa F401
from .module_paths import _relative_import_prefix  # noqa F401
from .module_paths import _target_module_name  # noqa F401
from .split_core import _extract_shared_helpers  # noqa F401
from .split_core import _remove_entity_lines  # noqa F401
from .split_core import _topo_depth  # noqa F401
from .split_core import generate_file_splits  # noqa F401
from .split_models import SplitResult  # noqa F401


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
