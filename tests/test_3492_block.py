import textwrap

# Source that already defines _helper AND has duplicate blocks.
_COLLISION_SOURCE = textwrap.dedent(
    """\
    def _helper(x):
        return x

    def foo():
        x = compute(data)
        y = transform(x)
        z = finalize(y)

    def bar():
        x = compute(data)
        y = transform(x)
        z = finalize(y)
    """
)
_COLLISION_RANGES = [(9, 11)]  # overlaps bar's body
