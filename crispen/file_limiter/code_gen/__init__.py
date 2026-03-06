"""Code generation for FileLimiter: build new files and update original source."""

from __future__ import annotations
from .imports_analysis import ImportInfo  # noqa F401
from .imports_analysis import _abs_package_for_dir  # noqa F401
from .imports_analysis import _collect_external_imported_names  # noqa F401
from .imports_analysis import _collect_name_loads  # noqa F401
from .imports_analysis import _extract_import_info  # noqa F401
from .imports_analysis import _find_cross_file_imports  # noqa F401
from .imports_analysis import _find_needed_imports  # noqa F401
from .imports_analysis import _find_project_root  # noqa F401
from .imports_analysis import _import_derived_names  # noqa F401
from .imports_analysis import _import_line_numbers  # noqa F401
from .imports_analysis import _module_path_from_file  # noqa F401
from .imports_analysis import _prune_inline_redundant_imports  # noqa F401
from .imports_analysis import _prune_unused_imports  # noqa F401
from .imports_analysis import _relative_import_prefix  # noqa F401
from .imports_analysis import _strip_top_level_import_lines  # noqa F401
from .imports_analysis import _target_module_name  # noqa F401
from .rewrite_utils import SplitResult  # noqa F401
from .rewrite_utils import _add_re_exports  # noqa F401
from .rewrite_utils import _bump_relative_imports  # noqa F401
from .rewrite_utils import _extract_module_docstring  # noqa F401
from .rewrite_utils import _extract_shared_helpers  # noqa F401
from .rewrite_utils import _remove_entity_lines  # noqa F401
from .rewrite_utils import _strip_module_docstring  # noqa F401
from .rewrite_utils import _topo_depth  # noqa F401
from .rewrite_utils import generate_file_splits  # noqa F401


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
