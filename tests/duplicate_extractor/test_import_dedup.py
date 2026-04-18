from crispen.refactors.duplicate_extractor import _lift_and_dedup_imports


def test_lift_and_dedup_no_changes_needed():
    src = "import os\nfrom typing import Any, Dict\nx = 1\n"
    assert _lift_and_dedup_imports(src) == src


def test_lift_and_dedup_exact_from_duplicate():
    src = "from typing import Any\nfrom typing import Any\n"
    assert _lift_and_dedup_imports(src) == "from typing import Any\n"


def test_lift_and_dedup_partial_overlap_adds_new_names():
    # Original F811 trigger: helper adds Any+Dict+Optional, file had Any+Dict
    src = "from typing import Any, Dict\nfrom typing import Any, Dict, Optional\n"
    assert _lift_and_dedup_imports(src) == "from typing import Any, Dict, Optional\n"


def test_lift_and_dedup_second_adds_only_new_names():
    src = "from typing import Any\nfrom typing import Optional\n"
    assert _lift_and_dedup_imports(src) == "from typing import Any, Optional\n"


def test_lift_and_dedup_multiple_modules_independent():
    src = (
        "from typing import Any\n"
        "from os.path import join\n"
        "from typing import Dict\n"
        "from os.path import exists\n"
    )
    result = _lift_and_dedup_imports(src)
    assert result == "from typing import Any, Dict\nfrom os.path import join, exists\n"


def test_lift_and_dedup_plain_import_deduped():
    # Unlike the old _dedup_from_imports, plain 'import X' dups are now removed
    src = "import os\nimport os\n"
    assert _lift_and_dedup_imports(src) == "import os\n"


def test_lift_and_dedup_skips_multiline_parens():
    src = "from typing import (\n    Any,\n    Dict,\n)\nfrom typing import Any\n"
    # Paren form not matched; single-line import stands alone — no change
    assert _lift_and_dedup_imports(src) == src


def test_lift_and_dedup_skips_wildcard():
    src = "from typing import *\nfrom typing import *\n"
    assert _lift_and_dedup_imports(src) == src


def test_lift_and_dedup_skips_commented_import_line():
    # Inline comment prevents matching; both lines are left alone
    src = "from typing import Any  # noqa\nfrom typing import Any\n"
    assert _lift_and_dedup_imports(src) == src


def test_lift_and_dedup_skips_indented_imports():
    # Indented imports (TYPE_CHECKING blocks, try/except, etc.) are not touched
    src = "    from typing import Any\n    from typing import Dict\n"
    assert _lift_and_dedup_imports(src) == src


def test_lift_and_dedup_empty_names_skipped():
    # Malformed import with no names: left unchanged
    src = "from typing import ,\nfrom typing import ,\n"
    assert _lift_and_dedup_imports(src) == src


def test_lift_and_dedup_non_import_lines_preserved():
    src = "from typing import Any\nx = 1\nfrom typing import Dict\ny = 2\n"
    result = _lift_and_dedup_imports(src)
    assert result == "from typing import Any, Dict\nx = 1\ny = 2\n"


def test_lift_and_dedup_lifts_misplaced_existing_module():
    # Helper inserted before second_fn lands after def first_fn → misplaced
    # The import merges into the block and the misplaced copy is removed.
    src = (
        "from typing import Any\n"
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "from typing import Optional\n"  # misplaced — helper preamble
        "def _helper():\n"
        "    pass\n"
        "\n"
        "def second_fn():\n"
        "    pass\n"
    )
    result = _lift_and_dedup_imports(src)
    assert result == (
        "from typing import Any, Optional\n"
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "def _helper():\n"
        "    pass\n"
        "\n"
        "def second_fn():\n"
        "    pass\n"
    )


def test_lift_and_dedup_lifts_misplaced_new_module():
    # Helper introduces a brand-new import mid-file → moved to after block.
    src = (
        "from typing import Any\n"
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "from collections import OrderedDict\n"  # misplaced — new module
        "def _helper():\n"
        "    pass\n"
        "\n"
        "def second_fn():\n"
        "    pass\n"
    )
    result = _lift_and_dedup_imports(src)
    assert result == (
        "from typing import Any\n"
        "from collections import OrderedDict\n"  # lifted after last block import
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "def _helper():\n"
        "    pass\n"
        "\n"
        "def second_fn():\n"
        "    pass\n"
    )


def test_lift_and_dedup_lifts_misplaced_plain_import_new_module():
    # Covers: misplaced plain 'import X' (i >= first_funcdef_idx branch) and
    # the new_plain_modules emission path inside _emit_new_imports.
    src = (
        "from typing import Any\n"
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "import os\n"  # misplaced plain import — new module
        "def _helper():\n"
        "    pass\n"
    )
    result = _lift_and_dedup_imports(src)
    assert result == (
        "from typing import Any\n"
        "import os\n"  # lifted after last block import
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "def _helper():\n"
        "    pass\n"
    )


def test_lift_and_dedup_sorts_new_imports_by_pep8_section():
    # New lifted imports are sorted future→stdlib→third-party→local regardless
    # of the order they were encountered.
    src = (
        "from typing import Any\n"  # block stdlib import
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "import requests\n"  # misplaced third-party
        "from collections import OrderedDict\n"  # misplaced stdlib
        "def _helper():\n"
        "    pass\n"
    )
    result = _lift_and_dedup_imports(src)
    assert result == (
        "from typing import Any\n"
        "from collections import OrderedDict\n"  # stdlib before third-party
        "import requests\n"
        "\n"
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "def _helper():\n"
        "    pass\n"
    )


def test_lift_and_dedup_blank_lines_in_block_dropped():
    # Blank lines between import lines in the block are removed when the block
    # is rebuilt — covers the blank-line-dropping branch in pass 5.
    src = (
        "import os\n"
        "\n"  # blank between block imports → dropped on rebuild
        "from typing import Any\n"
        "from typing import Dict\n"  # duplicate module → merged
        "x = 1\n"
    )
    result = _lift_and_dedup_imports(src)
    # PEP 8 sort: both are stdlib (group 1); from_order precedes plain_order in
    # all_final_imports so stable sort keeps 'from typing' before 'import os'.
    assert result == ("from typing import Any, Dict\n" "import os\n" "x = 1\n")


def test_lift_and_dedup_no_block_imports_inserts_before_first_funcdef():
    # File has no imports at all; helper adds one mid-file → moved to very top.
    src = (
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "from collections import OrderedDict\n"  # misplaced
        "def _helper():\n"
        "    pass\n"
        "\n"
        "def second_fn():\n"
        "    pass\n"
    )
    result = _lift_and_dedup_imports(src)
    assert result == (
        "from collections import OrderedDict\n"  # inserted before first funcdef
        "def first_fn():\n"
        "    pass\n"
        "\n"
        "def _helper():\n"
        "    pass\n"
        "\n"
        "def second_fn():\n"
        "    pass\n"
    )
