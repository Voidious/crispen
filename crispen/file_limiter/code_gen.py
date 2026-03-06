"""Code generation for FileLimiter: build new files and update original source."""

from __future__ import annotations
from .helpers_graph import _extract_shared_helpers  # noqa F401
from .helpers_graph import _topo_depth  # noqa F401
from .imports_utils import ImportInfo  # noqa F401
from .imports_utils import _collect_name_loads  # noqa F401
from .imports_utils import _extract_import_info  # noqa F401
from .imports_utils import _find_cross_file_imports  # noqa F401
from .imports_utils import _find_needed_imports  # noqa F401
from .imports_utils import _import_derived_names  # noqa F401
from .imports_utils import _import_line_numbers  # noqa F401
from .imports_utils import _prune_inline_redundant_imports  # noqa F401
from .imports_utils import _prune_unused_imports  # noqa F401
from .imports_utils import _relative_import_prefix  # noqa F401
from .imports_utils import _target_module_name  # noqa F401
from .project_paths import _abs_package_for_dir  # noqa F401
from .project_paths import _collect_external_imported_names  # noqa F401
from .project_paths import _find_project_root  # noqa F401
from .project_paths import _module_path_from_file  # noqa F401
from .reexports import _add_re_exports  # noqa F401
from .splits_core import SplitResult  # noqa F401
from .splits_core import generate_file_splits  # noqa F401
from .splits_transform import _remove_entity_lines  # noqa F401


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
