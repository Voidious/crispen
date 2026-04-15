from __future__ import annotations


_VALID_REWRITE = (
    '@patch("crispen.after.X")\n'
    '@patch("crispen.after.Y")\n'
    "def test_f(mock_x, mock_y):\n"
    "    pass\n"
)
