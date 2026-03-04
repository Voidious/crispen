import textwrap


def _write_if_not_else_fixture(tmp_path, filename="code.py", source=None):
    if source is None:
        source = textwrap.dedent(
            """\
            if not x:
                a()
            else:
                b()
            """
        )
    f = tmp_path / filename
    f.write_text(source, encoding="utf-8")
    return f, source
