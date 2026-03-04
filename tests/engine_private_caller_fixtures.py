import textwrap


def _write_private_caller_fixture_file(tmp_path, filename="code.py"):
    source = textwrap.dedent(
        """\
        def _make_result():
            return (1, 2, 3)

        def use_it():
            a, b, c = _make_result()
        """
    )
    f = tmp_path / filename
    f.write_text(source, encoding="utf-8")
    return f
