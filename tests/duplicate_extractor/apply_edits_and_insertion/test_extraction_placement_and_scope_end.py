from unittest.mock import MagicMock, patch
import textwrap
from libcst.metadata import MetadataWrapper
from crispen.refactors.duplicate_extractor import (
    DuplicateExtractor,
    _SeqInfo,
    _SequenceCollector,
    _normalize_replacement_indentation,
    _scope_end_line,
)
import libcst as cst
from ..test_duplicate_extractor_integration_core import (
    _make_extract_response,
    _make_verify_response,
    _make_veto_response,
)


def _make_seq_with_source(source: str) -> _SeqInfo:
    return _SeqInfo(
        stmts=[], start_line=1, end_line=1, scope="f", source=source, fingerprint=""
    )


def test_normalize_indentation_already_correct():
    # Replacement already matches the block's indentation — unchanged.
    seq = _make_seq_with_source("    x = compute()\n    y = finalize(x)\n")
    replacement = "    result = helper()\n"
    assert (
        _normalize_replacement_indentation(seq, replacement)
        == "    result = helper()\n"
    )


def test_normalize_indentation_col0_to_indented():
    # Replacement at column 0 is re-indented to match the original block.
    seq = _make_seq_with_source("    x = compute()\n    y = finalize(x)\n")
    replacement = "result = helper()\n"
    assert (
        _normalize_replacement_indentation(seq, replacement)
        == "    result = helper()\n"
    )


def test_normalize_indentation_multiline():
    # Multi-line replacement at column 0 gets uniformly re-indented.
    seq = _make_seq_with_source("        x = a()\n        y = b(x)\n")
    replacement = "x = helper()\nif x is None:\n    x = default()\n"
    expected = (
        "        x = helper()\n        if x is None:\n            x = default()\n"
    )
    assert _normalize_replacement_indentation(seq, replacement) == expected


def test_normalize_indentation_module_level_block():
    # Module-level block (no indent) — replacement is just dedented.
    seq = _make_seq_with_source("x = compute()\ny = finalize(x)\n")
    replacement = "result = helper()\n"
    assert _normalize_replacement_indentation(seq, replacement) == "result = helper()\n"


def test_normalize_indentation_empty_source():
    # Empty source — no indentation can be inferred; replacement returned as-is.
    seq = _make_seq_with_source("")
    replacement = "result = helper()\n"
    assert _normalize_replacement_indentation(seq, replacement) == replacement


def test_successful_extraction_has_two_blank_lines(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Each function has 4 statements. The first statement is STRUCTURALLY different
    # between them (if-block vs assignment), so the normalizer produces different
    # fingerprints for the full 4-stmt body. Only the trailing 3-stmt block
    # (compute/transform/finalize) is duplicated, so the proxy-wrapper guard
    # does not trigger (3 stmts < body_stmt_count 4).
    source = textwrap.dedent(
        """\
        import os

        def foo():
            if debug:
                validate(data)
            x = compute(data)
            y = transform(x)
            z = finalize(y)

        def bar():
            result = validate(data)
            x = compute(data)
            y = transform(x)
            z = finalize(y)
        """
    )
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "    _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor([(12, 14)], source=source)

    assert de._new_source is not None
    # Exactly 2 blank lines before and after the inserted helper.
    assert "\n\n\ndef _helper" in de._new_source
    assert "\n\n\n\ndef _helper" not in de._new_source
    assert "def _helper(data):\n    pass\n\n\ndef foo" in de._new_source


def test_helper_placed_before_class_not_inside(monkeypatch):
    """Helper extracted from class methods must be placed BEFORE the class.

    When duplicate blocks live inside class methods, inserting a module-level
    helper before the method (inside the class body) ends the class definition
    prematurely and turns the remaining methods into nested functions.  The fix
    in _find_insertion_point walks backwards to the enclosing class.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        import os

        class MyClass:
            def method_a(self, x):
                if self.debug:
                    pass
                a = compute(x)
                b = transform(a)
                c = finalize(b)
                return c

            def method_b(self, x):
                result = None
                a = compute(x)
                b = transform(a)
                c = finalize(b)
                return c
        """
    )
    helper = "def _do_work(x):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_do_work",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        return _do_work(x)\n",
                        "        return _do_work(x)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        de = DuplicateExtractor([(1, 100)], source=source)

    assert de._new_source is not None
    compile(de._new_source, "<test>", "exec")
    # Helper must appear BEFORE the class definition, not inside it.
    helper_pos = de._new_source.find("def _do_work")
    class_pos = de._new_source.find("class MyClass")
    assert (
        helper_pos < class_pos
    ), "helper was placed after/inside class instead of before it"
    # The class structure must be intact: MyClass still has both methods.
    import ast as _ast

    tree = _ast.parse(de._new_source)
    classes = [n for n in _ast.walk(tree) if isinstance(n, _ast.ClassDef)]
    assert len(classes) == 1
    assert classes[0].name == "MyClass"
    methods = [n.name for n in classes[0].body if isinstance(n, _ast.FunctionDef)]
    assert "method_a" in methods
    assert "method_b" in methods


def test_successful_extraction_module_level(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        import os

        def foo():
            if debug:
                pass
            x = compute(data)
            y = transform(x)
            z = finalize(y)

        def bar():
            result = None
            x = compute(data)
            y = transform(x)
            z = finalize(y)
        """
    )
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True, "same logic"),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "    _helper(data)\n",
                        "    _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]

        de = DuplicateExtractor([(12, 14)], source=source)

    assert de._new_source is not None
    assert "_helper" in de._new_source
    assert len(de.changes_made) == 1
    assert "'_helper'" in de.changes_made[0]
    assert de.get_rewritten_source() == de._new_source


def test_staticmethod_placement(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class MyClass:
            def foo(self):
                if self.debug:
                    pass
                x = compute(data)
                y = transform(x)
                z = finalize(y)

            def bar(self):
                result = None
                x = compute(data)
                y = transform(x)
                z = finalize(y)
        """
    )
    helper = "    @staticmethod\n    def _helper(data):\n        pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:MyClass",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]

        de = DuplicateExtractor([(11, 13)], source=source)

    assert de._new_source is not None


def test_staticmethod_placement_zero_indent_helper_auto_indented(monkeypatch):
    """0-indent helper with staticmethod: placement is auto-indented into the class."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class MyClass:
            def foo(self):
                if self.debug:
                    pass
                x = compute(data)
                y = transform(x)
                z = finalize(y)

            def bar(self):
                result = None
                x = compute(data)
                y = transform(x)
                z = finalize(y)
        """
    )
    # LLM generates a 0-indent (module-level) def even though it requested
    # staticmethod:MyClass placement.  Without auto-indent this would end the
    # class body at the docstring, making foo/bar nested inside the helper.
    helper_zero_indent = "def _helper(self, data):\n    return compute(data)\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        mock_client.messages.create.side_effect = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:MyClass",
                    "helper_source": helper_zero_indent,
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]

        de = DuplicateExtractor([(11, 13)], source=source)

    assert de._new_source is not None
    # foo and bar must remain real class methods, not nested inside the helper.
    import ast as _ast

    tree = _ast.parse(de._new_source)
    class_def = next(
        n
        for n in _ast.walk(tree)
        if isinstance(n, _ast.ClassDef) and n.name == "MyClass"
    )
    top_level_methods = {
        n.name for n in class_def.body if isinstance(n, _ast.FunctionDef)
    }
    assert "foo" in top_level_methods
    assert "bar" in top_level_methods
    assert "_helper" in top_level_methods


def test_cross_class_duplicates_use_module_level_placement(monkeypatch):
    """Duplicates in different classes must be extracted as module-level functions."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class ClassA:
            def foo(self):
                if self.debug:
                    pass
                x = compute(data)
                y = transform(x)
                z = finalize(y)

        class ClassB:
            def bar(self):
                result = None
                x = compute(data)
                y = transform(x)
                z = finalize(y)
        """
    )
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        responses = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        _helper(data)\n",
                        "        _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        mock_client.messages.create.side_effect = responses
        de = DuplicateExtractor([(3, 5)], source=source)

    assert de._new_source is not None
    # The extraction call prompt should tell the LLM to use module_level
    extract_prompt = mock_client.messages.create.call_args_list[1][1]["messages"][0][
        "content"
    ]
    assert "module_level" in extract_prompt
    assert "staticmethod" not in extract_prompt


def test_cross_class_staticmethod_placement_rejected(monkeypatch):
    """LLM returning staticmethod placement for a cross-class group is rejected."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class ClassA:
            def foo(self):
                if self.debug:
                    pass
                x = compute(data)
                y = transform(x)
                z = finalize(y)

        class ClassB:
            def bar(self):
                result = None
                x = compute(data)
                y = transform(x)
                z = finalize(y)
        """
    )
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        # First extraction attempt: LLM ignores prompt and returns staticmethod
        # placement for a cross-class group → rejected; second attempt: correct.
        responses = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:ClassA",
                    "helper_source": (
                        "    @staticmethod\n    def _helper(data):\n        pass\n"
                    ),
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        _helper(data)\n",
                        "        _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        mock_client.messages.create.side_effect = responses
        de = DuplicateExtractor([(3, 5)], source=source)

    assert de._new_source is not None
    # Three LLM calls: veto + two extraction attempts
    assert mock_client.messages.create.call_count == 4


def test_cross_class_staticmethod_placement_rejected_non_verbose(monkeypatch):
    """Defensive cross-class check works when verbose=False (no print side-effect)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class ClassA:
            def foo(self):
                if self.debug:
                    pass
                x = compute(data)
                y = transform(x)
                z = finalize(y)

        class ClassB:
            def bar(self):
                result = None
                x = compute(data)
                y = transform(x)
                z = finalize(y)
        """
    )
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        responses = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:ClassA",
                    "helper_source": (
                        "    @staticmethod\n    def _helper(data):\n        pass\n"
                    ),
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        _helper(data)\n",
                        "        _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        mock_client.messages.create.side_effect = responses
        de = DuplicateExtractor([(3, 5)], source=source, verbose=False)

    assert de._new_source is not None


def test_same_class_module_level_placement_rejected(monkeypatch):
    """module_level placement with self.<name>() call sites is rejected and retried."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class MyClass:
            def foo(self):
                if self.debug:
                    pass
                x = compute(data)
                y = transform(x)
                z = finalize(y)

            def bar(self):
                result = None
                x = compute(data)
                y = transform(x)
                z = finalize(y)
        """
    )
    helper_module = "def _helper(data):\n    pass\n"
    helper_static = "    @staticmethod\n    def _helper(data):\n        pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        responses = [
            _make_veto_response(True),
            # First attempt: module_level placement but call sites use self._helper()
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper_module,
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            # Second attempt: correct staticmethod placement
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:MyClass",
                    "helper_source": helper_static,
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        mock_client.messages.create.side_effect = responses
        de = DuplicateExtractor([(11, 13)], source=source)

    assert de._new_source is not None
    # veto + two extraction attempts + verify
    assert mock_client.messages.create.call_count == 4


def test_same_class_module_level_placement_rejected_non_verbose(monkeypatch):
    """Same inconsistency rejection works when verbose=False."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class MyClass:
            def foo(self):
                if self.debug:
                    pass
                x = compute(data)
                y = transform(x)
                z = finalize(y)

            def bar(self):
                result = None
                x = compute(data)
                y = transform(x)
                z = finalize(y)
        """
    )
    helper_static = "    @staticmethod\n    def _helper(data):\n        pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        responses = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": "def _helper(data):\n    pass\n",
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:MyClass",
                    "helper_source": helper_static,
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        mock_client.messages.create.side_effect = responses
        de = DuplicateExtractor([(11, 13)], source=source, verbose=False)

    assert de._new_source is not None


def test_cross_class_module_level_self_call_rejected(monkeypatch):
    """module_level with self.<name>() call sites in a cross-class group is rejected."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class ClassA:
            def foo(self):
                if self.debug:
                    pass
                x = compute(data)
                y = transform(x)
                z = finalize(y)

        class ClassB:
            def bar(self):
                result = None
                x = compute(data)
                y = transform(x)
                z = finalize(y)
        """
    )
    helper = "def _helper(data):\n    pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        responses = [
            _make_veto_response(True),
            # First attempt: module_level but call sites use self._helper()
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            # Second attempt: correct call sites
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "module_level",
                    "helper_source": helper,
                    "call_site_replacements": [
                        "        _helper(data)\n",
                        "        _helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        mock_client.messages.create.side_effect = responses
        de = DuplicateExtractor([(3, 5)], source=source)

    assert de._new_source is not None
    assert mock_client.messages.create.call_count == 4


def test_staticmethod_wrong_class_rejected(monkeypatch):
    """LLM naming the wrong class in staticmethod:X is rejected and retried."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class ClassA:
            def setup(self):
                pass

        class ClassB:
            def foo(self):
                if self.debug:
                    pass
                x = compute(data)
                y = transform(x)
                z = finalize(y)

            def bar(self):
                result = None
                x = compute(data)
                y = transform(x)
                z = finalize(y)
        """
    )
    helper_static = "    @staticmethod\n    def _helper(data):\n        pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        responses = [
            _make_veto_response(True),
            # First attempt: LLM names the wrong class
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:ClassA",
                    "helper_source": helper_static,
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            # Second attempt: correct class name
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:ClassB",
                    "helper_source": helper_static,
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        mock_client.messages.create.side_effect = responses
        de = DuplicateExtractor([(14, 16)], source=source)

    assert de._new_source is not None
    assert mock_client.messages.create.call_count == 4


def test_staticmethod_wrong_class_rejected_non_verbose(monkeypatch):
    """Wrong-class staticmethod rejection works when verbose=False."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = textwrap.dedent(
        """\
        class ClassA:
            def setup(self):
                pass

        class ClassB:
            def foo(self):
                if self.debug:
                    pass
                x = compute(data)
                y = transform(x)
                z = finalize(y)

            def bar(self):
                result = None
                x = compute(data)
                y = transform(x)
                z = finalize(y)
        """
    )
    helper_static = "    @staticmethod\n    def _helper(data):\n        pass\n"
    with patch("crispen.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.APIError = Exception
        responses = [
            _make_veto_response(True),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:ClassA",
                    "helper_source": helper_static,
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_extract_response(
                {
                    "function_name": "_helper",
                    "placement": "staticmethod:ClassB",
                    "helper_source": helper_static,
                    "call_site_replacements": [
                        "        self._helper(data)\n",
                        "        self._helper(data)\n",
                    ],
                }
            ),
            _make_verify_response(True, []),
        ]
        mock_client.messages.create.side_effect = responses
        de = DuplicateExtractor([(14, 16)], source=source, verbose=False)

    assert de._new_source is not None


def test_sequence_collector_class_scope():
    """_SequenceCollector sets class_scope for sequences inside class methods."""

    source = textwrap.dedent(
        """\
        x = 1
        y = 2
        z = 3

        class MyClass:
            def method(self):
                a = 1
                b = 2
                c = 3
        """
    )
    lines = source.splitlines(keepends=True)
    tree = cst.parse_module(source)
    collector = _SequenceCollector(lines, max_seq_len=8)
    MetadataWrapper(tree).visit(collector)

    module_seqs = [s for s in collector.sequences if s.class_scope is None]
    class_seqs = [s for s in collector.sequences if s.class_scope == "MyClass"]
    assert module_seqs, "expected module-level sequences with class_scope=None"
    assert class_seqs, "expected class-method sequences with class_scope='MyClass'"


def _make_source_lines(src: str):
    return src.splitlines(keepends=True)


def test_scope_end_line_module_returns_full_length():
    lines = _make_source_lines("x = 1\ny = 2\n")
    assert _scope_end_line(lines, "<module>", 1) == len(lines)


def test_scope_end_line_function_scope():
    src = "def foo():\n    x = 1\n    y = 2\n\ndef bar():\n    z = 3\n"
    lines = _make_source_lines(src)
    # Block ends at line 2 (inside foo). foo ends at line 3.
    assert _scope_end_line(lines, "foo", 2) == 3


def test_scope_end_line_does_not_bleed_into_next_function():
    src = "def foo():\n    x = 1\n\ndef bar():\n    x = 2\n"
    lines = _make_source_lines(src)
    # Searching for `x` after line 2 should stop at end of foo (line 2), not
    # reach bar where `x` also appears.
    end = _scope_end_line(lines, "foo", 2)
    assert end == 2  # foo ends at line 2; bar's x is excluded


def test_scope_end_line_picks_innermost_matching_scope():
    # Two functions named "inner" — one nested inside outer, one at module level.
    src = (
        "def outer():\n"
        "    def inner():\n"
        "        a = 1\n"
        "    inner()\n"
        "\n"
        "def inner():\n"
        "    b = 2\n"
    )
    lines = _make_source_lines(src)
    # Block at line 3 is inside the nested inner (lines 2-3). That is the
    # smallest matching span, so end_lineno == 3 is returned.
    assert _scope_end_line(lines, "inner", 3) == 3


def test_scope_end_line_class_scope():
    src = "class Foo:\n    x = 1\n    y = 2\n\nclass Bar:\n    x = 3\n"
    lines = _make_source_lines(src)
    assert _scope_end_line(lines, "Foo", 2) == 3


def test_scope_end_line_no_match_returns_full_length():
    src = "def foo():\n    x = 1\n"
    lines = _make_source_lines(src)
    # Scope name doesn't match any definition.
    assert _scope_end_line(lines, "bar", 1) == len(lines)


def test_scope_end_line_syntax_error_returns_full_length():
    lines = _make_source_lines("def (\n    x = 1\n")
    assert _scope_end_line(lines, "foo", 1) == len(lines)
